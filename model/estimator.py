from enum import Enum
from typing import Dict, Optional, Sequence

import torch
from fast_forward.index import Index
from fast_forward.ranking import Ranking
from transformers import AutoModel, AutoTokenizer

EncodingModelBatch = Dict[str, torch.LongTensor]


class WEIGHT_METHOD(Enum):
    """
    Enumeration for different types of probability distributions used to assign weights to tokens in the WeightedAvgEncoder.

    Attributes:
        UNIFORM: all tokens are weighted equally.
        LEARNED: weights are learned during training.
    """

    UNIFORM = "UNIFORM"
    LEARNED = "LEARNED"


class AvgEmbQueryEstimator(torch.nn.Module):
    """
    Estimate query embeddings as the weighted average of:
        - lightweight semantic query estimation.
            - based on the weighted average of query's (fine-tuned) token embeddings.
        - its top-ranked document embeddings.

    Note that the optimal values for these values are learned during fine-tuning:
    - `self.tok_embs`: the token embeddings
    - `self.tok_embs_avg_weights`: token embedding weighted averages
    - `self.embs_avg_weights`: embedding weighted averages

    """

    def __init__(
        self,
        n_docs: int,
        tok_w_method: str = "LEARNED",
        docs_only: bool = False,
        q_only: bool = False,
        normalize_q_emb_1: bool = False,
        normalize_q_emb_2: bool = False,
    ) -> None:
        """Constructor.

        Args:
            n_docs (int): The number of top-ranked documents to average.
            tok_w_method (TOKEN_WEIGHT_METHOD): The method to use for token weighting.
            docs_only (bool): Whether to disable the lightweight query estimation and only use the top-ranked documents.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
            normalize_q_emb_1 (bool): Whether to normalize the lightweight query estimation.
            normalize_q_emb_2 (bool): Whether to normalize the final query embedding.
        """
        super().__init__()
        self.n_docs = n_docs
        self.tok_w_method = WEIGHT_METHOD(tok_w_method)
        self.docs_only = docs_only
        self.q_only = q_only
        self.normalize_q_emb_1 = normalize_q_emb_1
        self.normalize_q_emb_2 = normalize_q_emb_2
        self.pretrained_model = "bert-base-uncased"
        self.doc_encoder = None
        self._ranking = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        vocab_size = self.tokenizer.vocab_size

        model = AutoModel.from_pretrained(self.pretrained_model, return_dict=True)
        self.tok_embs = model.get_input_embeddings()

        self.tok_embs_weights = torch.nn.Parameter(torch.ones(vocab_size) / vocab_size)

        n_embs = n_docs + 1
        self.embs_weights = torch.nn.Parameter(torch.ones(n_embs) / n_embs)

    def _get_top_docs(self, queries: Sequence[str]):
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.doc_encoder is not None, "Provide a doc_encoder before encoding."

        # Create tensors for padding and total embedding counts
        d_embs_pad = torch.zeros((len(queries), self.n_docs, 768)).to(self.device)
        n_embs_per_q = torch.ones((len(queries)), dtype=torch.int).to(self.device)

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)]
        query_to_idx = {query: idx for idx, query in enumerate(queries)}

        for query, group in top_docs.groupby("query"):
            # TODO: Get the embeddings for the top-ranked documents from doc_encoder
            d_embs, d_idxs = self.index._get_vectors(group["id"].unique())
            if self.index.quantizer is not None:
                d_embs = self.index.quantizer.decode(d_embs)
            d_embs = torch.tensor(d_embs[[x[0] for x in d_idxs]])

            # Pad and count embeddings for this query
            query_idx = query_to_idx[str(query)]
            d_embs_pad[query_idx, : len(d_embs)] = d_embs
            n_embs_per_q[query_idx] += len(d_embs)

        return d_embs_pad, n_embs_per_q

    def forward(self, q_tokens: EncodingModelBatch) -> torch.Tensor:
        input_ids = q_tokens["input_ids"]
        attention_mask = q_tokens["attention_mask"]
        batch_size = len(input_ids)

        if self.docs_only:
            q_emb_1 = torch.zeros((batch_size, 768))
        else:

            # estimate lightweight query as weighted average of q_tok_embs
            q_tok_embs = self.tok_embs(input_ids)
            q_tok_embs_masked = q_tok_embs * attention_mask.unsqueeze(-1)
            match self.tok_w_method:
                case WEIGHT_METHOD.UNIFORM:
                    q_emb_1 = torch.mean(q_tok_embs_masked, 1)
                case WEIGHT_METHOD.LEARNED:
                    q_tok_weights = torch.nn.functional.softmax(
                        self.tok_embs_weights[input_ids], -1
                    )
                    q_emb_1 = torch.sum(
                        q_tok_embs_masked * q_tok_weights.unsqueeze(-1), 1
                    )
            if self.normalize_q_emb_1:
                q_emb_1 = torch.nn.functional.normalize(q_emb_1)

        if self.q_only:
            return q_emb_1

        # lookup embeddings of top-ranked documents in (in-memory) self.index
        queries = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        d_embs_pad, n_embs_per_q = self._get_top_docs(queries)

        # estimate query embedding as weighted average of q_emb and d_embs
        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs_pad), -2).to(self.device)
        embs_weights = torch.zeros((batch_size, embs.shape[-2])).to(self.device)
        # assign self.embs_weights to embs_weights, but only up to the number of top-ranked documents per query
        for i, n_embs in enumerate(n_embs_per_q):
            if self.docs_only:
                embs_weights[i, 0] = 0.0
                embs_weights[i, 1:n_embs] = torch.nn.functional.softmax(
                    self.embs_weights[1:n_embs], 0
                )
            else:
                embs_weights[i, :n_embs] = torch.nn.functional.softmax(
                    self.embs_weights[:n_embs], 0
                )
        q_emb_2 = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        if self.normalize_q_emb_2:
            q_emb_2 = torch.nn.functional.normalize(q_emb_2)

        return q_emb_2

    @property
    def ranking(self) -> Optional[Ranking]:
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: Ranking):
        self._ranking = ranking.cut(self.n_docs)

    @property
    def embedding_dimension(self) -> int:
        return self.tok_embs.embedding_dim

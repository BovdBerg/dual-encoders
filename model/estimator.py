from enum import Enum
from typing import Dict, Optional, Sequence

import numpy as np
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
        self.n_embs = n_docs + 1
        self.tok_w_method = WEIGHT_METHOD(tok_w_method)
        self.docs_only = docs_only
        self.q_only = q_only
        self.normalize_q_emb_1 = normalize_q_emb_1
        self.normalize_q_emb_2 = normalize_q_emb_2
        self.pretrained_model = "bert-base-uncased"
        self.doc_encoder = None
        self.d_text_index = None
        self._ranking = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        vocab_size = self.tokenizer.vocab_size

        model = AutoModel.from_pretrained(self.pretrained_model, return_dict=True)
        self.tok_embs = model.get_input_embeddings()

        self.tok_embs_avg_weights = torch.nn.Parameter(torch.ones(vocab_size) / vocab_size)

        self.embs_avg_weights = torch.nn.Parameter(torch.ones(self.n_embs) / self.n_embs)

    def _get_top_docs(self, queries: Sequence[str]):
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.doc_encoder is not None, "Provide a doc_encoder before encoding."

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)]
        print(f"top_docs:\n{top_docs}")
        top_docs_ids = torch.tensor(top_docs["id"].unique().astype(np.int64), dtype=torch.long)
        if len(top_docs_ids) < self.n_docs:
            # Repeat top_docs_ids until reaching length n_docs
            top_docs_ids = torch.cat([top_docs_ids] * self.n_docs, dim=0)[: self.n_docs]
        print(f"top_docs_ids:\n{top_docs_ids}")

        # self.d_text_index is a pt.IndexFactory.of(index_ref) mapping doc_ids to doc_texts
        d_texts = self.d_text_index(top_docs_ids)
        print(f"d_texts: {d_texts}")

        # TODO: probably requires a EncodingModelBatch, which is tokenized.
        d_embs = self.doc_encoder(d_texts)
        print(f"d_embs: {d_embs}")

        return d_embs

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
                        self.tok_embs_avg_weights[input_ids], -1
                    )
                    q_emb_1 = torch.sum(
                        q_tok_embs_masked * q_tok_weights.unsqueeze(-1), 1
                    )
            if self.normalize_q_emb_1:
                q_emb_1 = torch.nn.functional.normalize(q_emb_1)

        if self.q_only:
            return q_emb_1

        # find embeddings of top-ranked documents
        queries = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        d_embs = self._get_top_docs(queries)

        # estimate query embedding as weighted average of q_emb and d_embs
        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs), -2).to(self.device)
        embs_weights = torch.zeros((self.n_embs), device=self.device)
        if self.docs_only:
            embs_weights[0] = 0.0
            embs_weights[1:self.n_embs] = torch.nn.functional.softmax(self.embs_avg_weights[1:self.n_embs], 0)
        else:
            embs_weights[:self.n_embs] = torch.nn.functional.softmax(self.embs_avg_weights[:self.n_embs], 0)
        embs_weights = embs_weights.unsqueeze(0).expand(batch_size, -1)

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

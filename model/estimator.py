from enum import Enum
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pyterrier as pt
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
        pretrained_model: str = "bert-base-uncased",
        tok_w_method: str = "LEARNED",
        q_only: bool = False,
        docs_only: bool = False,
        normalize_q_emb_1: bool = False,
        normalize_q_emb_2: bool = False,
    ) -> None:
        """Constructor.

        Args:
            n_docs (int): The number of top-ranked documents to average.
            tok_w_method (TOKEN_WEIGHT_METHOD): The method to use for token weighting.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
            docs_only (bool): Whether to disable the lightweight query estimation and only use the top-ranked documents.
            normalize_q_emb_1 (bool): Whether to normalize the lightweight query estimation.
            normalize_q_emb_2 (bool): Whether to normalize the final query embedding.
        """
        assert not (q_only and docs_only), "Cannot use both q_only and docs_only."

        super().__init__()
        pt.init()

        self.n_docs = n_docs
        self.n_embs = n_docs + 1
        self.pretrained_model = pretrained_model
        self.tok_w_method = WEIGHT_METHOD(tok_w_method)
        self.docs_only = docs_only
        self.q_only = q_only
        self.normalize_q_emb_1 = normalize_q_emb_1
        self.normalize_q_emb_2 = normalize_q_emb_2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.doc_encoder = None
        self.sparse_index = pt.BatchRetrieve.from_dataset(
            "msmarco_passage",
            "terrier_stemmed_text",
            wmodel="BM25",
            metadata=["text"],  # Add doc text to the output
            num_results=self.n_docs,  # Only return n_docs top-ranked documents
        )

        model = AutoModel.from_pretrained(self.pretrained_model, return_dict=True)
        self.tok_embs = model.get_input_embeddings()
        self.tok_embs_avg_weights = torch.nn.Parameter(
            torch.ones(self.tokenizer.vocab_size)
        )
        self.embs_avg_weights = torch.nn.Parameter(
            torch.ones(self.n_embs)
        )

    def _get_top_docs_embs(self, queries: pd.DataFrame):
        assert self.doc_encoder is not None, "Provide a doc_encoder before encoding."

        # Retrieve top-ranked documents for all queries in batch
        top_docs = self.sparse_index.transform(queries)

        # Tokenize top_docs texts
        # TODO: use TransformerTokenizer (from estimator.yaml) directly
        d_toks = self.tokenizer(
            top_docs["text"].tolist(),
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=True,  # TODO: might make sense to disable special tokens, e.g. [CLS] will learn a generic embedding
        ).to(self.device)

        # Encode top_docs_toks with doc_encoder and reshape
        d_embs = self.doc_encoder(d_toks)
        d_embs = d_embs.view(len(queries), self.n_docs, -1)

        return d_embs

    def forward(self, q_tokens: EncodingModelBatch) -> torch.Tensor:
        input_ids = q_tokens["input_ids"]
        attention_mask = q_tokens["attention_mask"]

        if self.docs_only:
            q_emb_1 = torch.zeros((len(input_ids), 768))
        else:
            # estimate lightweight query as weighted average of q_tok_embs
            q_tok_embs = self.tok_embs(input_ids)
            # TODO: does the masking even do anything? q_tok_embs is already 0 where attention_mask is 0 right?
            match self.tok_w_method:
                case WEIGHT_METHOD.UNIFORM:
                    q_emb_1 = torch.mean(q_tok_embs, 1)
                case WEIGHT_METHOD.LEARNED:
                    q_tok_weights = torch.nn.functional.softmax(
                        self.tok_embs_avg_weights[input_ids], -1
                    )
                    q_emb_1 = torch.sum(
                        q_tok_embs * q_tok_weights.unsqueeze(-1), 1
                    )
            if self.normalize_q_emb_1:
                q_emb_1 = torch.nn.functional.normalize(q_emb_1)

        if self.q_only:
            return q_emb_1

        # Validate queries to prevent QueryParserException
        def validate_query(query):
            # Check if query is empty or only whitespace
            if not query or query.strip() == "":
                return False
            # Check if query ends with an incomplete phrase or special character
            if query.strip().endswith(":") or query.strip().endswith("?"):
                return False
            # Check for special characters that might cause parsing issues
            if any(char in query for char in [';', '"', "'"]):
                return False
            # Check for minimum length
            if len(query.strip()) < 5:
                return False
            return True

        # find embeddings of top-ranked documents
        queries = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        queries = pd.DataFrame({"query": queries, "qid": np.arange(len(queries))})
        queries = queries[queries["query"].apply(validate_query)]
        if queries.empty:
            raise ValueError("All queries in batch are invalid.")
        d_embs = self._get_top_docs_embs(queries)

        # estimate query embedding as weighted average of q_emb and d_embs
        print("q_emb_1 shape v d_embs shape:", q_emb_1.shape, d_embs.shape)
        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs), -2).to(self.device)
        embs_weights = torch.zeros((self.n_embs), device=self.device)
        if self.docs_only:
            embs_weights[0] = 0.0
            embs_weights[1 : self.n_embs] = torch.nn.functional.softmax(
                self.embs_avg_weights[1 : self.n_embs], 0
            )
        else:
            embs_weights[: self.n_embs] = torch.nn.functional.softmax(
                self.embs_avg_weights[: self.n_embs], 0
            )
        embs_weights = embs_weights.unsqueeze(0).expand(len(queries), -1)

        q_emb_2 = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        if self.normalize_q_emb_2:
            q_emb_2 = torch.nn.functional.normalize(q_emb_2)

        return q_emb_2

    @property
    def embedding_dimension(self) -> int:
        return self.tok_embs.embedding_dim

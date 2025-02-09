import logging
import re
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
        WEIGHTED: weights are learned during training.
    """

    UNIFORM = "UNIFORM"
    WEIGHTED = "WEIGHTED"


class AvgEmbQueryEstimator(torch.nn.Module):
    """
    Estimate query embeddings as the weighted average of:
        - lightweight semantic query estimation.
            - based on the weighted average of query's (fine-tuned) token embeddings.
        - its top-ranked document embeddings.

    Note that the optimal values for these values are learned during fine-tuning:
    - `self.tok_embs`: the token embeddings
    - `self.tok_embs_weights`: token embedding weighted averages
    - `self.embs_weights`: embedding weighted averages
    """

    def __init__(
        self,
        n_docs: int,
        pretrained_model: str = "bert-base-uncased",
        tok_embs_w_method: str = "WEIGHTED",
        embs_w_method: str = "WEIGHTED",
        q_only: bool = False,
    ) -> None:
        """Constructor.

        Args:
            n_docs (int): The number of top-ranked documents to average.
            tok_embs_w_method (TOKEN_WEIGHT_METHOD): The method to use for token weighting.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
        """
        super().__init__()
        pt.init()

        self.n_docs = n_docs
        self.n_embs = n_docs + 1
        self.pretrained_model = pretrained_model
        self.tok_embs_w_method = WEIGHT_METHOD(tok_embs_w_method)
        self.embs_w_method = WEIGHT_METHOD(embs_w_method)
        self.q_only = q_only

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.doc_tokenizer = None
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

        vocab_size = self.tokenizer.vocab_size
        self.tok_embs_weights = torch.nn.Parameter(torch.ones(vocab_size))

        self.embs_weights = torch.nn.Parameter(torch.ones(self.n_embs) / self.n_embs)

        self.to(self.device)
        self.eval()

    def compute_weighted_average(
        self,
        embs: torch.tensor,
        init_weights: torch.tensor,
        mask: torch.tensor,
    ) -> torch.Tensor:
        print(f"init_weights: {init_weights.shape}\n{init_weights}")
        weights = init_weights * mask  # Mask padding
        print(f"weights (masked): {weights.shape}\n{weights}")

        # Normalize the weights per emb
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize
        print(f"weights (normalized): {weights.shape}\n{weights}")

        embs = embs * weights.unsqueeze(-1)  # Apply weights
        q_estimation = embs.sum(-2)  # Compute weighted sum
        return q_estimation

    def _get_top_docs_embs(self, queries: Sequence[str]) -> torch.Tensor:
        assert self.doc_tokenizer is not None, "Provide a doc_tokenizer training."
        assert self.doc_encoder is not None, "Provide a doc_encoder before training."

        # Retrieve top-ranked documents for all queries in batch
        top_docs_embs = torch.zeros((len(queries), self.n_docs, 768), device=self.device)
        for q_no, query in enumerate(queries):
            try:
                q_top_docs = self.sparse_index.search(query)
                if "text" not in q_top_docs.keys():
                    continue
                q_top_docs_texts = q_top_docs["text"].tolist()
                q_top_docs_toks = self.doc_tokenizer(q_top_docs_texts).to(self.device)
            except Exception as e:
                continue
            q_top_docs_embs = torch.zeros((self.n_docs, 768), device=self.device)
            q_top_docs_embs[: len(q_top_docs)] = self.doc_encoder(q_top_docs_toks)
            top_docs_embs[q_no] = q_top_docs_embs

        return top_docs_embs

    def forward(self, q_tokens: EncodingModelBatch) -> torch.Tensor:
        input_ids = q_tokens["input_ids"]  # (batch_size, max_length)
        batch_size, max_length = input_ids.size()

        # Estimate lightweight query as (weighted) average of q_tok_embs, excluding padding
        q_tok_embs = self.tok_embs(input_ids)  # Get token embeddings
        match self.tok_embs_w_method:  # q_tok_weights: (batch_size, max_length)
            case WEIGHT_METHOD.UNIFORM:
                q_tok_weights = torch.ones_like(input_ids, dtype=torch.float) / max_length
            case WEIGHT_METHOD.WEIGHTED:
                q_tok_weights = self.tok_embs_weights[input_ids]
        q_emb_1 = self.compute_weighted_average(q_tok_embs, q_tok_weights, q_tokens["attention_mask"])
        if self.q_only:
            return q_emb_1

        # Find top-ranked document embeddings
        queries = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        d_embs = self._get_top_docs_embs(queries)

        # Estimate final query as (weighted) average of q_emb_1 ++ top_docs_embs
        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs), -2) # (batch_size, n_embs, emb_dim)
        match self.embs_w_method:  # embs_weights: (batch_size, n_embs)
            case WEIGHT_METHOD.UNIFORM:
                embs_weights = torch.ones_like((batch_size, self.n_embs), dtype=torch.float) / self.n_embs
            case WEIGHT_METHOD.WEIGHTED:
                embs_weights = self.embs_weights # (n_embs)
                embs_weights = embs_weights.unsqueeze(0).expand(batch_size, -1) # (batch_size, n_embs), repeated values
        embs_mask = torch.zeros((batch_size, self.n_embs), device=self.device)  # (batch_size, n_embs), 1 for each non-zero emb
        embs_mask[:, 0] = 1  # q_emb_1 is always non-zero
        embs_mask[:, 1:] = torch.any(d_embs != 0, dim=-1)  # Check if any doc embs are non-zero
        q_emb_2 = self.compute_weighted_average(embs, embs_weights, embs_mask)
        return q_emb_2

    @property
    def embedding_dimension(self) -> int:
        return self.tok_embs.embedding_dim

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        keys_to_remove = [
            key
            for key in sd.keys()
            if key.startswith("doc_encoder.")
            or key.startswith("query_encoder.doc_encoder.")
        ]
        for key in keys_to_remove:
            del sd[key]
        return sd

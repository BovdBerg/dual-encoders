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
        self.tok_embs_weights = torch.nn.Parameter(torch.ones(vocab_size) / vocab_size)

        self.embs_weights = torch.nn.Parameter(torch.ones(self.n_embs) / self.n_embs)

        self.to(self.device)
        self.eval()

    def _get_top_docs_embs(self, queries: Sequence[str]) -> torch.Tensor:
        assert self.doc_tokenizer is not None, "Provide a doc_tokenizer training."
        assert self.doc_encoder is not None, "Provide a doc_encoder before training."

        # Retrieve top-ranked documents for all queries in batch
        d_embs = torch.zeros((len(queries), self.n_docs, 768), device=self.device)
        for q_no, query in enumerate(queries):
            try:
                top_docs = self.sparse_index.search(query)
                d_texts = top_docs["text"].tolist()
                d_toks = self.doc_tokenizer(d_texts).to(self.device)
            except Exception as e:
                continue
            d_emb = torch.zeros((self.n_docs, 768), device=self.device)
            d_emb[: len(top_docs)] = self.doc_encoder(d_toks)
            d_embs[q_no] = d_emb

        return d_embs

    def forward(self, q_tokens: EncodingModelBatch) -> torch.Tensor:
        input_ids = q_tokens["input_ids"]
        mask = q_tokens["attention_mask"]

        # Estimate lightweight query as (weighted) average of q_tok_embs, excluding padding
        q_tok_embs = self.tok_embs(input_ids)
        q_tok_embs = q_tok_embs * mask.unsqueeze(-1)  # Mask padding tokens

        # Apply weights to the q_tok_embs
        if self.tok_embs_w_method == WEIGHT_METHOD.WEIGHTED:
            q_tok_weights = self.tok_embs_weights[input_ids]
            q_tok_weights = q_tok_weights * mask  # Mask padding weights
            q_tok_weights = q_tok_weights / q_tok_weights.sum(dim=1, keepdim=True)  # Normalize
            q_tok_embs = q_tok_embs * q_tok_weights.unsqueeze(-1)

        # Compute the mean of the masked embeddings, excluding padding
        n_masked = mask.sum(dim=1, keepdim=True)
        q_emb_1 = q_tok_embs.sum(dim=1) / n_masked

        if self.q_only:
            return q_emb_1

        # Find top-ranked document embeddings
        queries = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        d_embs = self._get_top_docs_embs(queries)

        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs), -2)
        mask = (embs != 0).float().sum(dim=-1)  # (batch_size, n_embs, dim) -> (batch_size, n_embs)

        # Apply weights to the embs
        match self.embs_w_method:
            case WEIGHT_METHOD.UNIFORM:
                embs_weights = torch.ones(self.n_embs) / self.n_embs
            case WEIGHT_METHOD.WEIGHTED:
                embs_weights = self.embs_weights[: self.n_embs]
                embs_weights = embs_weights.unsqueeze(0).expand(len(queries), -1) # (n_embs) -> (batch_size, n_embs)
                embs_weights = embs_weights * mask  # Mask zeros
        embs_weights = embs_weights / embs_weights.sum(dim=-1, keepdim=True)  # Normalize
        embs = embs * embs_weights.unsqueeze(-1)

        # Compute the mean of the masked embeddings, excluding padding
        n_masked = mask.sum(dim=-1, keepdim=True)
        q_emb_2 = embs.sum(dim=-2) / n_masked
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

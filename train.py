#! /usr/bin/env python3


import os
import pickle
from math import ceil
from pathlib import Path

import hydra
import pandas as pd
import pyterrier as pt
import torch
from fast_forward.ranking import Ranking
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from ranking_utils.model import TrainingMode
from tqdm import tqdm

from model.estimator import AvgEmbQueryEstimator


def create_lexical_ranking(n_docs):
    cache_n_docs = 50
    dataset_cache_path = Path("/scratch/bovandenberg/fast-forward-indexes/data/q-to-rep/tct")
    cache_dir = dataset_cache_path / f"ranking_cache_{cache_n_docs}docs"
    os.makedirs(cache_dir, exist_ok=True)
    chunk_size = 10_000

    train_topics = pt.get_dataset("irds:msmarco-passage/train").get_topics()
    eval_topics = pt.get_dataset("irds:msmarco-passage/eval").get_topics()
    eval_head = eval_topics.sample(n=1000, random_state=42)  # Because my local data used 1k val samples first
    dev_topics = pt.get_dataset("irds:msmarco-passage/dev").get_topics()
    all_topics = pd.concat([eval_head, train_topics, eval_topics, dev_topics])
    queries_path = dataset_cache_path / f"{len(all_topics)}_topics.csv"
    all_topics.to_csv(queries_path, index=False)

    queries = pd.read_csv(queries_path)
    res_df = pd.DataFrame()
    for i, chunk in enumerate(
        tqdm(
            pd.read_csv(queries_path, chunksize=chunk_size),
            desc="Loading/creating Ranking in chunks",
            total=ceil(len(queries) / chunk_size),
        )
    ):
        cache_file = cache_dir / f"{i * chunk_size}-{(i + 1) * chunk_size}.pt"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                chunk_ranking = pickle.load(f)
        else:
            print(f"Creating new ranking for {cache_file}")
            sys_bm25 = (
                pt.BatchRetrieve.from_dataset(
                    "msmarco_passage",
                    "terrier_stemmed",
                    wmodel="BM25",
                    memory=True,
                    verbose=True,
                    num_results=cache_n_docs,
                )
                % cache_n_docs
            )
            chunk["query"] = chunk["query"].astype(str)
            chunk_df = sys_bm25.transform(chunk)
            chunk_ranking = Ranking(
                chunk_df.rename(columns={"qid": "q_id", "docno": "id"})
            )

            with open(cache_file, "wb") as f:
                pickle.dump(chunk_ranking, f)

        chunk_ranking = chunk_ranking.cut(n_docs)
        res_df = pd.concat([res_df, chunk_ranking._df])

    ranking = Ranking(res_df)
    print(f"Created ranking with {len(ranking._df)} rows")

    return ranking


def create_index(tokenizer):
    index_path = "/home/bvdb9/indices/msm-psg/d_tokens2_index_msmpsg"

    if os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        d_tokens_index = pt.IndexFactory.of(index_path)
    else:
        print(f"Creating new index at {index_path}")
        dataset = pt.get_dataset("irds:msmarco-passage")
        indexer = pt.IterDictIndexer(index_path)
        index_ref = indexer.index(
            [
                {
                    "docno": doc["docno"],
                    "d_tokens": tokenizer(str(doc["text"]))
                }
                for doc in dataset.get_corpus_iter()
            ],
            fields=["docno", "d_tokens"]
        )
        d_tokens_index = pt.IndexFactory.of(index_ref)
    print("Index is now available")
    
    return d_tokens_index


@hydra.main(config_path="config", config_name="training", version_base="1.3")
def main(config: DictConfig) -> None:
    seed_everything(config.random_seed, workers=True)
    data_processor = instantiate(config.ranker.data_processor)
    data_module = instantiate(
        config.training_data,
        data_processor=data_processor,
    )
    trainer = instantiate(config.trainer)
    model = instantiate(config.ranker.model)

    q_enc = model.query_encoder
    if isinstance(q_enc, AvgEmbQueryEstimator):
        pt.init()
        q_enc.d_tokens_index = create_index(data_processor.doc_tokenizer)
        q_enc.ranking = create_lexical_ranking(q_enc.n_docs)

    if config.ckpt_path is not None:
        model.load_state_dict(torch.load(config.ckpt_path)["state_dict"])
    data_module.training_mode = model.training_mode = TrainingMode.CONTRASTIVE
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

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


def create_lexical_ranking(n_docs, val_samples = None):
    cache_n_docs = 50
    dataset_cache_path = Path("/home/bvdb9/fast-forward-indexes/data/q-to-rep/tct")
    cache_dir = dataset_cache_path / f"ranking_cache_{cache_n_docs}docs"
    os.makedirs(cache_dir, exist_ok=True)
    chunk_size = 10_000

    train_topics = pt.get_dataset("irds:msmarco-passage/train").get_topics()
    val_topics = pt.get_dataset("irds:msmarco-passage/eval").get_topics()
    if val_samples is not None:
        val_topics = val_topics.sample(n=val_samples, random_state=42)
    all_topics = pd.concat([val_topics, train_topics])
    queries_path = dataset_cache_path / f"{len(all_topics)}_topics.csv"
    all_topics.to_csv(queries_path, index=False)

    ranking = None
    queries = pd.read_csv(queries_path)
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
        if ranking is None:
            res_df = chunk_ranking._df
        else:
            res_df = pd.concat([res_df, chunk_ranking._df])

    return Ranking(res_df).cut(n_docs)


def load_index(index_path):
    index = OnDiskIndex.load(index_path)
    if args.storage == "mem":
        index = index.to_memory(2**15)

    return index


@hydra.main(config_path="config", config_name="training", version_base="1.3")
def main(config: DictConfig) -> None:
    seed_everything(config.random_seed, workers=True)
    data_module = instantiate(
        config.training_data,
        data_processor=instantiate(config.ranker.data_processor),
    )
    trainer = instantiate(config.trainer)
    model = instantiate(config.ranker.model)

    q_enc = model.query_encoder
    if isinstance(q_enc, AvgEmbQueryEstimator):
        pt.init()
        ranking = create_lexical_ranking(q_enc.n_docs, trainer.limit_val_batches)
        q_enc.ranking = ranking
        index = load_index(q_enc.index_path)
        q_enc.index = index

    if config.ckpt_path is not None:
        model.load_state_dict(torch.load(config.ckpt_path)["state_dict"])
    data_module.training_mode = model.training_mode = TrainingMode.CONTRASTIVE
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

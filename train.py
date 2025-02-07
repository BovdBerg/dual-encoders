#! /usr/bin/env python3

import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from model.estimator import AvgEmbQueryEstimator
from ranking_utils.model import TrainingMode

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)


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
        q_enc.doc_encoder = model.doc_encoder
        q_enc.doc_tokenizer = data_processor.doc_tokenizer
        q_enc.encode_docs = model.encode_docs

    if config.ckpt_path is not None:
        model.load_state_dict(torch.load(config.ckpt_path)["state_dict"])
    data_module.training_mode = model.training_mode = TrainingMode.CONTRASTIVE
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

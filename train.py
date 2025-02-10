#! /usr/bin/env python3

import logging
import os
import warnings
from operator import call

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from ranking_utils.model import TrainingMode

from model.estimator import AvgEmbQueryEstimator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)
logging.getLogger("org.terrier").setLevel(logging.ERROR)


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

    if config.ckpt_path is not None:
        model.load_state_dict(torch.load(config.ckpt_path)["state_dict"])
    data_module.training_mode = model.training_mode = TrainingMode.CONTRASTIVE
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

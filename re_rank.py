#! /usr/bin/env python3


import csv
import logging
from pathlib import Path

import hydra
import ir_datasets
import ir_measures
from fast_forward import Mode, OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util import to_ir_measures
from omegaconf import DictConfig

from util import StandaloneEncoder

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="re_ranking", version_base="1.3")
def main(config: DictConfig) -> None:
    LOGGER.info("loading %s", config.ckpt_path)
    query_encoder = StandaloneEncoder(
        config.query_encoder, config.ckpt_path, "query_encoder", config.device
    )

    index_dir = Path(config.index_dir)
    LOGGER.info("loading %s", index_dir)
    ff_index = OnDiskIndex.load(index_dir / "ff_index.h5", query_encoder, Mode.MAXP)

    LOGGER.info("loading %s", config.dataset)
    dataset = ir_datasets.load(config.dataset)

    LOGGER.info("reading %s", config.sparse_runfile)
    sparse_ranking = Ranking.from_file(
        Path(config.sparse_runfile),
        {
            query.query_id: getattr(query, config.query_attribute)
            for query in dataset.queries_iter()
        },
    )
    if config.cutoff_sparse is not None:
        sparse_ranking.cut(config.cutoff_sparse)

    LOGGER.info("computing scores")
    ff_out = ff_index(sparse_ranking)

    if len(config.metrics) > 0:
        measures = list(map(ir_measures.parse_measure, config.metrics))
        LOGGER.info("computing %s", measures)

        results_file = Path.cwd() / f"ff_results_{config.name}.csv"
        LOGGER.info("writing %s", results_file)
        with open(results_file, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, ["type", "name", "dataset", "alpha"] + measures)
            writer.writeheader()

            for alpha in config.alpha:
                interpolated_ranking = ff_out.interpolate(sparse_ranking, alpha)
                run_file = Path.cwd() / f"ff_results_{config.name}_{alpha}.tsv"
                LOGGER.info("writing %s", run_file)
                interpolated_ranking.save(run_file)

                results = ir_measures.calc_aggregate(
                    measures, dataset.qrels_iter(), to_ir_measures(interpolated_ranking)
                )
                results["type"] = "fast-forward"
                results["name"] = config.name
                results["dataset"] = config.dataset
                results["alpha"] = alpha
                writer.writerow(results)
                LOGGER.info("headers were:\n%s", writer.fieldnames)
                LOGGER.info("results for alpha %f:\n%s", alpha, results)


if __name__ == "__main__":
    main()

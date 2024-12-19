import logging

import hydra
import omegaconf
import wandb
from utilpy import log_init

from src.configs import Config
from src.fastdtm import DTM


@hydra.main(version_base=None, config_path="src/configs/", config_name="demo")
def main(cfg: Config):
    log_init()
    logger = logging.getLogger("main")

    docs = [[[1, 2, 5], [3, 4], [2]], [[1, 3]], [[1], [5, 2]]]
    vocabulary = ["func", "yellow", "prefix", "func1", "yellow1", "prefix1"]
    try:
        dtm = DTM(docs, vocabulary, cfg.model)
        dtm.initialize(True)
        dtm.estimate(cfg.data.epochs)
        dtm.save_data(cfg.data.output_dir)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    main()

import logging

import equinox as eqx
import hydra
import omegaconf
import wandb
from utilpy import log_init

from src.configs import Config
from src.fastdtm import DTMJax


@hydra.main(version_base=None, config_path="src/configs/", config_name="demo")
def main(cfg: Config):
    log_init()
    logger = logging.getLogger("main")

    docs = [[[1, 2, 5], [3, 4], [2]], [[1, 3]], [[1], [5, 2]]]
    vocabulary = ["func", "yellow", "prefix", "func1", "yellow1", "prefix1"]
    try:
        dtm, state = eqx.nn.make_with_state(DTMJax)(docs, vocabulary, cfg.model)
        state = dtm.initialize(state, True)
        state = dtm.estimate(state, cfg.data.epochs)
        # dtm.save_data(state, cfg.data.output_dir)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    main()

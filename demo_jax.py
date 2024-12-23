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

    with open("data/vocabulary.txt") as f:
        vocabulary = [line.replace("\n", "") for line in f.readlines()]
    docs = []
    with open("data/nips.conf") as f:
        years = [line.replace("\n", "") for line in f.readlines()]
    for year in years:
        with open(year) as f:
            docs.append(
                [
                    [int(numstr) for numstr in filter(lambda a: a != "", line.replace("\n", "").split(" "))]
                    for line in f.readlines()
                ]
            )
    # docs = [[[1, 2, 5], [3, 4], [2]], [[1, 3]], [[1], [5, 2]]]
    # vocabulary = ["func", "yellow", "prefix", "func1", "yellow1", "prefix1"]
    try:
        dtm, state = eqx.nn.make_with_state(DTMJax)(docs, vocabulary, cfg.model)
        logger.info("start initialize")
        state = dtm.initialize(state, True)
        logger.info("start estimate")
        state = dtm.estimate(state, cfg.data.epochs)
        # dtm.save_data(state, cfg.data.output_dir)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    main()

import logging

import hydra
import omegaconf
import wandb
from utilpy import log_init

from src.configs import Config
from src.fastdtm import DTM


@hydra.main(version_base=None, config_path="src/configs/", config_name="nips")
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

    try:
        dtm = DTM(docs, vocabulary, cfg.model)
        dtm.initialize(True)
        dtm.estimate(cfg.data.epochs)
        dtm.save_data(cfg.data.output_dir)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    main()

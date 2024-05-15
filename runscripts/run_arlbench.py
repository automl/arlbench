"""Console script for arlbench."""

from __future__ import annotations

import csv
import logging
import sys
import traceback

import hydra
import jax
from arlbench.arlbench import run_arlbench
from codecarbon import track_emissions
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="base")
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg: DictConfig):
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Logging configured")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX device count: {jax.local_device_count()}")
    logger.info(f"JAX default backend: {jax.default_backend()}")

    if cfg.jax_enable_x64:
        logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)
    try:
        return run(cfg, logger)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def run(cfg: DictConfig, logger: logging.Logger):
    """Console script for arlbench."""
    logger.info("Starting run with config:")
    logger.info(str(OmegaConf.to_yaml(cfg)))

    # check if file done exists and if so, return
    try:
        with open("./done.txt") as f:
            logger.info("Job already done, returning.")

        with open("./performance.csv") as pf:
            csvreader = csv.reader(pf)
            performance = next(csvreader)
            performance = float(performance[0])
            logger.info(f"Returning performance {performance}.")
            return performance
    except FileNotFoundError:
        pass

    objectives = run_arlbench(cfg, logger=logger)

    with open("./performance.csv", "w+") as f:
        f.write(str(objectives))
    with open("./done.txt", "w+") as f:
        f.write("yes")

    return objectives


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

"""Console script for arlbench."""

from __future__ import annotations
import traceback
import sys

import hydra
from codecarbon import track_emissions
import jax
import logging
from omegaconf import OmegaConf, DictConfig
from arlbench.arlbench import run_arlbench


@hydra.main(version_base=None, config_path="configs", config_name="runtime_experiments")
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg: DictConfig):
    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def run(cfg: DictConfig):
    """Console script for arlbench."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Starting run with config:")
    logger.info(OmegaConf.to_yaml(cfg))

    logger.info("Enabling x64 support for JAX.")
    jax.config.update("jax_enable_x64", True)

    objectives = run_arlbench(cfg)

    with open("./performance.csv", "w+") as f:
        f.write(str(objectives))
    with open("./done.txt", "w+") as f:
        f.write("yes")

    return objectives


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

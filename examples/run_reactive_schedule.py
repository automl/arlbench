"""Console script for arlbench."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import logging
import sys
import traceback
from typing import TYPE_CHECKING

import hydra
import jax
from arlbench.autorl import AutoRLEnv

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run(cfg: DictConfig, logger: logging.Logger):
    """Gradient-based learning rate schedule. Spike the learning rate for one step if gradients stagnate."""
    # Initialize environment with general config
    env = AutoRLEnv(cfg.autorl)

    # Reset environment and run for 10 steps
    _ = env.reset()

    # Remember grad norm and if we currently spike the learning rate
    last_grad_norm = None
    spiked = False

    # define a tolerance for the gradient norm
    tolerance = 1e-4
    for i in range(100):

        # Statistics here contain the number of steps and gradient information
        statistics, objectives, te, tr, _ = env.step(cfg.hp_config)
        grad_norm, _ = statistics["grad_info"]

        # If grad norm doesn't change much, spike the learning rate
        if last_grad_norm is not None:
            print(i)
            print(abs(grad_norm - last_grad_norm))
        if last_grad_norm is not None and abs(grad_norm - last_grad_norm) < tolerance:
            last_lr = cfg.hp_config.learning_rate
            cfg.hp_config.learning_rate *= 10
            logger.info(f"Gradients stagnated, spiking learning rate to {cfg.hp_config.learning_rate}")
            spiked = True

        # Reset learning rate if we spiked it in the last step
        if spiked:
            cfg.hp_config.learning_rate = last_lr
            spiked = False
            logger.info(f"Resetting learning rate to {cfg.hp_config.learning_rate}")
        last_grad_norm = grad_norm
    logger.info(f"Training finished with a total reward of {objectives['reward_mean']}")


@hydra.main(version_base=None, config_path="configs", config_name="gradient_lr")
def execute(cfg: DictConfig):
    """Helper function for nice logging and error handling."""
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if cfg.jax_enable_x64:
        logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)
    try:
        return run(cfg, logger)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(execute())  # pragma: no cover

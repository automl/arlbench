"""Console script for arlbench."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import logging
import sys
import traceback

import hydra
import jax
from arlbench.autorl import AutoRLEnv
from omegaconf import DictConfig

def run(cfg: DictConfig, logger: logging.Logger):
    """Heuristic-based exploration schedule. Decrease epsilon in DQN when evaluation performance reaches a certain threshold."""
    logger.info(f"Starting run with epsilon value {str(cfg.hp_config.initial_epsilon)}")

    # Initialize environment with general config
    env = AutoRLEnv(cfg.autorl)

    # Reset environment and run for 10 steps
    _ = env.reset()
    for _ in range(10):
        # The objectives are configured to return the mean reward
        _, objectives, _, _, _ = env.step(cfg.hp_config)
        if objectives["reward_mean"] > 30 and cfg.hp_config.initial_epsilon > 0.7:
            # We can change epsilon by changing which config we run in the next step
            cfg.hp_config.target_epsilon = 0.7
            cfg.hp_config.initial_epsilon = 0.7
            logger.info("Agent reached performance threshold, decreasing epsilon to 0.7")
    
    logger.info(f"Training finished with a total reward of {objectives['reward_mean']}")

@hydra.main(version_base=None, config_path="configs", config_name="epsilon_heuristic")
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

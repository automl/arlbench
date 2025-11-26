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
from arlbench.arlbench import run_arlbench

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(
    version_base=None, config_path="configs", config_name="random_smac"
)
def execute(cfg: DictConfig):
    """Helper function for nice logging and error handling."""
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rng_key = jax.random.PRNGKey(0)

    randomized_cfgs = []
    for _ in range(cfg.n_configs):
        new_cfg = cfg.copy()
        for p, bounds in cfg.randomized_params.items():
            rng_key, subkey = jax.random.split(rng_key)
            #logger.info(f"Randomizing parameter {p} in bounds {bounds}")
            if bounds[0] == "int":
                new_value = jax.random.randint(
                    subkey, shape=(), minval=bounds[1], maxval=bounds[2]
                ).astype(jax.numpy.int32).item()
            elif bounds[0] == "float":
                new_value = jax.random.uniform(
                    subkey, shape=(), minval=bounds[1], maxval=bounds[2]
                ).astype(jax.numpy.float32).item()
            elif bounds[0] == "cat":
                index = jax.random.randint(
                    subkey, shape=(), minval=0, maxval=len(bounds) - 1
                ).item()
                new_value = bounds[index + 1]

            #print(f"Setting parameter {p} to value {new_value}")
            keys = p.split(".")
            sub_cfg = new_cfg[keys[0]]
            if len(keys) == 3:
                sub_cfg2 = sub_cfg[keys[1]]
                sub_cfg2[keys[-1]] = new_value
                sub_cfg[keys[1]] = sub_cfg2
            else:
                sub_cfg[keys[-1]] = new_value
            new_cfg[keys[0]] = sub_cfg
        randomized_cfgs.append(new_cfg)
        if len(randomized_cfgs) >= cfg.n_configs:
            break
    scores = []
    for random_cfg in randomized_cfgs:
        try:
            score = run(random_cfg, logger)
            scores.append(score)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            scores.append(-1000)

    return jax.numpy.mean(jax.numpy.array(scores))


def run(cfg: DictConfig, logger: logging.Logger):
    """Console script for arlbench."""
    #logger.info(f"Running ARLBench with config: {cfg}")
    objectives = run_arlbench(cfg, logger=logger)

    with open("./performance.csv", "w+") as f:
        f.write(str(objectives))
    with open("./done.txt", "w+") as f:
        f.write("yes")

    return objectives


if __name__ == "__main__":
    sys.exit(execute())  # pragma: no cover
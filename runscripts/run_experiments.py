"""Console script for runtime experiments."""
from __future__ import annotations

import sys

import hydra
from codecarbon import track_emissions
import jax
from arlbench.core.environments import make_env
from arlbench.core.algorithms import DQN, PPO, SAC
import time
import pandas as pd
import logging
import os
from datetime import timedelta

def format_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    h, m, s = str(td).split(':')
    return f"{int(h):d}:{m:2d}:{s:2d}"


ARLBENCH_ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
}


def train_arlbench(cfg, logger: logging.Logger):
    env = make_env(cfg.environment.env_framework, cfg.environment.env_name, n_envs=cfg.environment.n_envs, seed=cfg.seed, env_kwargs=cfg.environment.env_kwargs)
    rng = jax.random.PRNGKey(cfg.seed)

    algorithm_cls = ARLBENCH_ALGORITHMS[cfg.algorithm.algorithm]

    # override NAS config
    nas_config = algorithm_cls.get_default_nas_config()
    if cfg.nas_config:
        for k, v in cfg.nas_config:
            nas_config[k] = v

    agent = algorithm_cls(cfg.hp_config, env, nas_config=nas_config, cnn_policy=cfg.environment.cnn_policy)
    algorithm_state = agent.init(rng)

    start = time.time()
    logger.info("training started")
    (algorithm_state, results) = agent.train(*algorithm_state, n_total_timesteps=cfg.env.n_total_timesteps)
    logger.info("training finished")
    training_time = time.time() - start

    logging.info(f"Finished in {format_time(training_time)}s.")

    train_info_df = pd.DataFrame()
    for i in range(len(results.eval_rewards)):
        train_info_df[f"return_{i}"] = results.eval_rewards[i]

    return train_info_df, training_time


# TODO implement
def train_purejaxrl(cfg, logger: logging.Logger):
    pass


# TODO implement
def train_sbx(cfg, logger: logging.Logger):
    pass


@hydra.main(version_base=None, config_path="configs", config_name="base")
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg):
    dir_name= "./"      # TODO add to config
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if True:    # arlbench algorithm
        train_info_df, training_time = train_arlbench(cfg, logger)
    elif False: # purejaxrl algorithm
        train_info_df, training_time = train_purejaxrl(cfg, logger)
    elif False: # sbx algorithm
        train_info_df, training_time = train_sbx(cfg, logger)
    else:
        pass

    # os.makedirs(
    #     os.path.join("dqn_results", f"{framework}_{env_name}", dir_name), exist_ok=True
    # )
    # train_info_df.to_csv(
    #     os.path.join(
    #         f"{cfg.algorithm.algorithm}_results", f"{framework}_{env_name}", dir_name, f"{seed}_results.csv"
    #     )
    # )
    # with open(
    #     os.path.join(
    #         "dqn_results", f"{framework}_{env_name}", dir_name, f"{seed}_info"
    #     ),
    #     "w",
    # ) as f:
    #     f.write(f"config: {str(cfg)}\n")
    #     f.write(f"time: {training_time}\n")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
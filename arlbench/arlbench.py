from __future__ import annotations

from .autorl import AutoRLEnv
from omegaconf import DictConfig, OmegaConf
from logging import Logger
import os
import numpy as np
import sys


def run_arlbench(cfg: DictConfig, logger: Logger | None = None) -> float | tuple | list:
    """Run ARLBench using the given config and return objective(s)."""

    if "load" in cfg and cfg.load:
        checkpoint_path = os.path.join(cfg.load, cfg.autorl.checkpoint_name, "default_checkpoint_c_episode_1_step_1")
    else:
        checkpoint_path = None
    
    if "save" in cfg and cfg.save:
        cfg.autorl.checkpoint_dir = str(cfg.save).replace(".pt", "")
        if cfg.algorithm == "PPO":
            cfg.autorl.checkpoint = ["opt_state", "params"]
        else:
            cfg.autorl.checkpoint = ["opt_state", "params", "buffer"]

    env = AutoRLEnv(cfg.autorl)
    _ = env.reset()

    if logger:
        logger.info("Your AutoRL config is:")
        logger.info(OmegaConf.to_yaml(cfg.autorl))
        logger.info("Training started.")
    _, objectives, _, _, info = env.step(cfg.hp_config, checkpoint_path=checkpoint_path)
    if logger:
        logger.info("Training finished.")

    info["train_info_df"].to_csv("evaluation.csv", index=False)

    if "reward_curves" in cfg and cfg.reward_curves:
        return list(info["train_info_df"]["returns"])
    
    for k, v in objectives.items():
        if np.isnan(v):
            objectives[k] = sys.float_info.min

    if len(objectives) == 1:
        return objectives[list(objectives.keys())[0]]
    else:
        return tuple(objectives.values())

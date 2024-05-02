from __future__ import annotations

from .autorl import AutoRLEnv
from .utils import HandleTermination
from omegaconf import DictConfig


def run_arlbench(cfg: DictConfig):
    """Run ARLBench using the given config and return objective(s)."""
    env = AutoRLEnv(cfg.autorl)

    with HandleTermination(env):
        print(f"Your current config is: {cfg}")

        checkpoint_path = (
            cfg["load_checkpoint"]
            if "load_checkpoint" in cfg and cfg["load_checkpoint"] != ""
            else None
        )

        _ = env.reset(checkpoint_path=checkpoint_path)
        _, objectives, _, _, _ = env.step(cfg.hp_config)

        if len(objectives) == 1:
            return objectives[list(objectives.keys())[0]]
        else:
            return tuple(objectives.values())

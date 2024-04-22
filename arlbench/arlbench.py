from __future__ import annotations

from .autorl import AutoRLEnv
from .utils import HandleTermination


def run_arlbench(cfg: dict):
    """Run ARLBench using the given config and return objective(s)."""
    env = AutoRLEnv(cfg.autorl)

    with HandleTermination(env):
        print(f"Your current config is: {cfg}")

        checkpoint_path = cfg["load_checkpoint"] if "load_checkpoint" in cfg and cfg["load_checkpoint"] != "" else None

        _ = env.reset(checkpoint_path=checkpoint_path)
        obs, reward, term, trunc, info = env.step(cfg.hp_config)

        if len(reward) == 1:
            return reward[list(reward.keys())[0]]
        else:
            return tuple(reward.values())


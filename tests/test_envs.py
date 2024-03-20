import jax
import time

from arlbench.agents import DQN

from arlbench.utils import (
    make_env,
)

DQN_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}
DQN_CONFIG = DQN.get_default_configuration()

# TODO implement env tests

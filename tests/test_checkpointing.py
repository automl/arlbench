from arlbench.autorl_env import AutoRLEnv
import gymnax
import pytest
import jax
import numpy as np
import jax.numpy as jnp
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState
from arlbench.agents import TimeStep
import flashbax as fbx

from arlbench.agents import (
    PPO,
    DQN
)

from arlbench.utils.checkpointing import Checkpointer
from arlbench.utils import make_env


def test_read_write_buffer():
    CHECKPOINT_DIR = "/tmp/test_saves"

    DQN_OPTIONS = {
        "n_total_timesteps": 1e6,
        "n_envs": 10,
        "n_env_steps": 500,
        "n_eval_episodes": 10,
        "track_metrics": False,
        "track_traj": False,
    }

    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)

    (runner_state, buffer_state), _ = agent.train(runner_state, buffer_state)
    # TODO save buffer state

    disk_buffer_state = ...     # TODO load buffer state

    assert np.allclose(buffer_state.experience.obs,  disk_buffer_state.experience.obs, atol=1e5)

    (runner_state, disk_buffer_state), _ = agent.train(runner_state, disk_buffer_state)
    reward = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    assert reward > 400
    print(reward)


test_read_write_buffer()
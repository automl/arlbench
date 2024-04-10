
import time

import jax
import numpy as np

from arlbench.autorl.objectives import (EmissionsObjective, RewardObjective,
                                        RuntimeObjective)
from arlbench.core.algorithms import DQN
from arlbench.core.environments import make_env

DQN_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_eval_steps": 1e5,
    "n_eval_episodes": 10,
    "n_envs": 10,
    "n_env_steps": 500,
    "reward_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}

def test_reward():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    objectives = {}
    train_func = agent.train
    train_func = RewardObjective(train_func, objectives, env)

    (runner_state, _), _ = train_func(runner_state, buffer_state)
    rewards = agent.eval(runner_state, DQN_OPTIONS["reward_eval_episodes"])
    reward = np.mean(rewards)

    assert np.abs(reward - objectives["reward"]) < 0.01

def test_runtime():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    objectives = {}
    train_func = agent.train
    train_func = RuntimeObjective(train_func, objectives, None)

    start = time.time()
    train_func(runner_state, buffer_state)
    runtime = time.time() - start

    assert np.abs(runtime - objectives["runtime"]) < 0.05

def test_emissions():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    objectives = {}
    train_func = agent.train
    train_func = EmissionsObjective(train_func, objectives, None)
    train_func(runner_state, buffer_state)
    assert objectives["emissions"] > 0
    assert objectives["emissions"] < 1


if __name__ == "__main__":
    test_reward()
from __future__ import annotations

import numpy as np
from arlbench import AutoRLEnv
from arlbench.core.algorithms import DQN


def test_autorl_env_dqn_single_objective_upper():
    CONFIG = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 1e6,
        "n_eval_steps": 10,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "optimize_objectives": "upper",
        "state_features": [],
        "grad_obs": False,
        "n_steps": 10,
    }

    env = AutoRLEnv(CONFIG)
    _, _ = env.reset()

    action = DQN.get_default_hpo_config()

    _, objectives, done, _, _ = env.step(action)
    assert done is False
    assert isinstance(objectives, dict)
    assert objectives["reward_mean"] > 0

    rewards = env.eval(env.config["n_eval_episodes"])
    reward = np.mean(rewards)

    assert np.abs(objectives["reward_mean"] - reward) < 5


def test_autorl_env_dqn_single_objective_lower():
    CONFIG = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 1e6,
        "n_eval_steps": 10,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "optimize_objectives": "lower",
        "state_features": [],
        "grad_obs": False,
        "n_steps": 10,
    }

    env = AutoRLEnv(CONFIG)
    _, _ = env.reset()

    action = DQN.get_default_hpo_config()

    _, objectives, done, _, _ = env.step(action)
    assert done is False
    assert isinstance(objectives, dict)
    assert objectives["reward_mean"] < 0

    rewards = env.eval(env.config["n_eval_episodes"])
    reward = np.mean(rewards)

    assert np.abs(-objectives["reward_mean"] - reward) < 5


def test_autorl_env_dqn_multi_objective():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward_mean", "reward_std", "runtime", "emissions"],
        "state_features": [],
        "optimize_objectives": "upper",
        "checkpoint": [],
        "n_steps": 10,
        "n_eval_episodes": 10,
    }

    env = AutoRLEnv(CONFIG)
    init_obs, _ = env.reset()
    assert len(init_obs.keys()) == 0

    action = dict(DQN.get_default_hpo_config())
    n_obs, objectives, done, _, _ = env.step(action)

    assert len(n_obs.keys()) == 1
    assert done is False
    assert isinstance(objectives, dict)
    assert objectives["reward_mean"] > 0

    rewards = env.eval(env.config["n_eval_episodes"])
    reward = np.mean(rewards)

    assert np.abs(objectives["reward_mean"] - reward) < 5
    assert objectives["runtime"] < 0  # TODO improve? How can we estimate inner runtime?
    assert objectives["emissions"] < 0


if __name__ == "__main__":
    test_autorl_env_dqn_multi_objective()

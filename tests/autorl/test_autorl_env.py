from __future__ import annotations

import pytest

from arlbench import AutoRLEnv
from arlbench.core.algorithms import DQN


def test_autorl_env_dqn_default_obs():
    config = {
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
        "state_features": [],
        "n_steps": 10,
    }

    env = AutoRLEnv(config=config)
    init_obs, _ = env.reset()
    assert len(init_obs.keys()) == 0

    action = env.config_space.sample_configuration()
    obs, objectives, _, trunc, _ = env.step(action)
    assert len(obs.keys()) == 1
    assert obs["steps"].shape == (2,)
    assert trunc is False
    assert objectives["reward_mean"] > 0


def test_autorl_env_dqn_grad_obs():
    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 1e5,
        "n_eval_steps": 10,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "state_features": ["grad_info"],
        "n_steps": 10,
    }

    env = AutoRLEnv(config=config)
    init_obs, _ = env.reset()
    assert len(init_obs.keys()) == 0

    action = env.config_space.get_default_configuration()
    obs, objectives, _, trunc, _ = env.step(action)
    assert len(obs.keys()) == 2
    assert obs["steps"].shape == (2,)
    assert obs["grad_info"].shape == (2,)
    assert trunc is False
    assert objectives["reward_mean"] > 0


def test_autorl_env_ppo_grad_obs():
    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "ppo",
        "cnn_policy": False,
        "n_total_timesteps": 1e5,
        "n_eval_steps": 10,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "state_features": ["grad_info"],
        "n_steps": 10,
    }

    env = AutoRLEnv(config=config)
    init_obs, _ = env.reset()
    assert len(init_obs.keys()) == 0

    action = env.config_space.get_default_configuration()
    obs, objectives, _, trunc, _ = env.step(action)
    assert len(obs.keys()) == 2
    assert obs["steps"].shape == (2,)
    assert obs["grad_info"].shape == (2,)
    assert trunc is False
    assert objectives["reward_mean"] > 0


def test_autorl_env_sac_grad_obs():
    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "Pendulum-v1",
        "n_envs": 10,
        "algorithm": "sac",
        "cnn_policy": False,
        "n_total_timesteps": 5e4,
        "n_eval_steps": 10,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "state_features": ["grad_info"],
        "n_steps": 10,
    }

    env = AutoRLEnv(config=config)
    init_obs, _ = env.reset()
    assert len(init_obs.keys()) == 0

    action = env.config_space.get_default_configuration()
    obs, objectives, _, trunc, _ = env.step(action)
    assert len(obs.keys()) == 2
    assert obs["steps"].shape == (2,)
    assert obs["grad_info"].shape == (2,)
    assert trunc is False
    assert objectives["reward_mean"] > -2000


def test_autorl_env_dqn_per_switch():
    config = {
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
        "state_features": [],
        "n_steps": 10,
    }

    env = AutoRLEnv(config)
    _, _ = env.reset()
    action = env.config_space.get_default_configuration()

    action["buffer_prio_sampling"] = True
    _, objectives, _, _, _ = env.step(action)
    assert objectives["reward_mean"] > 400

    action["buffer_prio_sampling"] = False
    _, objectives, _, _, _ = env.step(action)
    assert objectives["reward_mean"] > 450

    action["buffer_prio_sampling"] = True
    _, objectives, _, _, _ = env.step(action)
    assert objectives["reward_mean"] > 490

    _, _ = env.reset()
    action["buffer_prio_sampling"] = False
    _, objectives, _, _, _ = env.step(action)
    assert objectives["reward_mean"] > 400

    action["buffer_prio_sampling"] = True
    _, objectives, _, _, _ = env.step(action)
    assert objectives["reward_mean"] > 450

    action["buffer_prio_sampling"] = False
    _, objectives, _, _, _ = env.step(action)
    assert objectives["reward_mean"] > 490


def test_autorl_env_dqn_dac():
    config = {
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
        "state_features": [],
        "n_steps": 3,
    }

    env = AutoRLEnv(config)
    # perform 3 HPO steps
    for _ in range(3):
        _, _ = env.reset()
        steps = 0
        trunc = False
        while not trunc:
            action = env.config_space.sample_configuration()

            obs, objectives, _, trunc, _ = env.step(action)
            steps += 1
            assert len(obs.keys()) == 1
            assert obs["steps"].shape == (2,)
            assert objectives["reward_mean"] > 0
        assert trunc is True
        assert steps == 3


def test_autorl_env_dqn_hpo():
    config = {
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
        "state_features": [],
        "n_steps": 1,  # Classic (static) HPO
    }

    env = AutoRLEnv(config)

    _, _ = env.reset()
    action = env.config_space.sample_configuration()
    obs, objectives, _, trunc, _ = env.step(action)
    assert len(obs.keys()) == 1
    assert obs["steps"].shape == (2,)
    assert objectives["reward_mean"] > 0
    assert trunc is True


def test_autorl_env_step_before_reset():
    config = {
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
        "state_features": [],
        "n_steps": 1,  # Classic HPO
    }

    env = AutoRLEnv(config)

    with pytest.raises(ValueError) as excinfo:
        action = dict(DQN.get_hpo_config_space().sample_configuration())
        env.step(action)

    assert "Called step() before reset()" in str(excinfo.value)


def test_autorl_env_forbidden_step():
    config = {
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
        "state_features": [],
        "n_steps": 1,  # Classic HPO
    }

    env = AutoRLEnv(config)
    env.reset()
    action = env.config_space.sample_configuration()
    env.step(action)

    with pytest.raises(ValueError) as excinfo:
        env.step(action)

    assert "Called step() before reset()" in str(excinfo.value)


if __name__ == "__main__":
    test_autorl_env_dqn_grad_obs()
    test_autorl_env_ppo_grad_obs()
    test_autorl_env_sac_grad_obs()

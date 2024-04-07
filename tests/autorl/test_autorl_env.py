import gymnax
import numpy as np
import pytest

from arlbench import AutoRLEnv
from arlbench.core.algorithms import DQN, PPO
from arlbench.core.environments import make_env


def test_autorl_env_dqn_single_objective():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward"],
        "checkpoint": [],
        "n_steps": 10,
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": True
    }

    envs = {
        0: {
            "env": make_env("gymnax", "CartPole-v1", n_envs=10, seed=42),
            "env_options": {
                "n_total_timesteps": 1e6,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    init_obs, _ = env.reset()
    assert init_obs.shape == (4,)

    action = dict(DQN.get_default_hpo_config())

    n_obs, objectives, done, _, _ = env.step(action)
    assert n_obs.shape == (4,)
    assert done is False
    assert isinstance(objectives, dict)
    assert objectives["reward_mean"] > 0

    rewards = env.algorithm.eval(env.runner_state, CONFIG["n_eval_episodes"])
    reward = np.mean(rewards)

    assert np.abs(objectives["reward_mean"] - reward) < 5

def test_autorl_env_dqn_multi_objective():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward", "runtime", "emissions"],
        "checkpoint": [],
        "n_steps": 10,
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": True
    }

    envs = {
        0: {
            "env": make_env("gymnax", "CartPole-v1", n_envs=10, seed=42),
            "env_options": {
                "n_total_timesteps": 1e6,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    init_obs, _ = env.reset()
    assert init_obs.shape == (4,)

    action = dict(DQN.get_default_hpo_config())
    n_obs, objectives, done, _, _ = env.step(action)

    assert n_obs.shape == (4,)
    assert done is False
    assert isinstance(objectives, dict)
    assert objectives["reward_mean"] > 0

    rewards = env.algorithm.eval(env.runner_state, CONFIG["n_eval_episodes"])
    reward = np.mean(rewards)

    assert np.abs(objectives["reward_mean"] - reward) < 5
    assert objectives["runtime"] > 0    # TODO improve? How can we estimate inner runtime?
    assert objectives["emissions"] > 0

def test_autorl_env_dqn_grad_obs():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward"],
        "checkpoint": [],
        "n_steps": 10,
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": True
    }

    envs = {
        0: {
            "env": make_env("gymnax", "CartPole-v1", n_envs=10, seed=42),
            "env_options": {
                "n_total_timesteps": 1e6,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    init_obs, _ = env.reset()
    assert init_obs.shape == (4,)

    action = dict(DQN.get_hpo_config_space().sample_configuration())
    n_obs, objectives, done, _, _ = env.step(action)
    assert n_obs.shape == (4,)
    assert done is False
    assert objectives["reward_mean"] > 0

def test_autorl_env_dqn_per_switch():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward"],
        "checkpoint": [],
        "n_steps": 10,
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": False
    }

    envs = {
        0: {
            "env": make_env("gymnax", "CartPole-v1", n_envs=10, seed=42),
            "env_options": {
                "n_total_timesteps": 1e6,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    _, _ = env.reset()
    _, objectives, _, _, _ = env.step({ "buffer_per": True })
    assert objectives["reward_mean"] > 400
    _, objectives, _, _, _ = env.step({ "buffer_per": False })
    assert objectives["reward_mean"] > 450
    _, objectives, _, _, _ = env.step({ "buffer_per": True })
    assert objectives["reward_mean"] > 490

    _, _ = env.reset()
    _, objectives, _, _, _ = env.step({ "buffer_per": False })
    assert objectives["reward_mean"] > 400
    _, objectives, _, _, _ = env.step({ "buffer_per": True })
    assert objectives["reward_mean"] > 450
    _, objectives, _, _, _ = env.step({ "buffer_per": False })
    assert objectives["reward_mean"] > 490


def test_autorl_env_dqn_hpo():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward"],
        "checkpoint": [],
        "n_steps": 1,   # Classic HPO
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": False
    }

    envs = {
        0: {
            "env": make_env("gymnax", "CartPole-v1", n_envs=10, seed=42),
            "env_options": {
                "n_total_timesteps": 1e6,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    # perform 3 HPO steps
    for _ in range(3):
        _, _ = env.reset()
        steps = 0
        done = False
        while not done:
            action = dict(DQN.get_hpo_config_space().sample_configuration())

            n_obs, objectives, done, _, _ = env.step(action)
            steps += 1
            assert n_obs.shape == (2,)
            assert objectives["reward_mean"] > 0
            assert done is True
        assert steps == 1


def test_autorl_env_step_before_reset():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward"],
        "checkpoint": [],
        "n_steps": 1,   # Classic HPO
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": False
    }

    envs = {
        0: {
            "env": make_env("gymnax", "CartPole-v1", n_envs=10, seed=42),
            "env_options": {
                "n_total_timesteps": 1e6,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    
    with pytest.raises(ValueError) as excinfo:
        action = dict(DQN.get_hpo_config_space().sample_configuration())
        env.step(action)
    
    assert "Called step() before reset()" in str(excinfo.value)


def test_autorl_env_forbidden_step():
    CONFIG = {
        "seed": 0,
        "algorithm": "dqn",
        "objectives": ["reward"],
        "checkpoint": [],
        "n_steps": 1,   # Classic HPO
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": False
    }

    envs = {
        0: {
            "env": make_env("gymnax", "CartPole-v1", n_envs=10, seed=42),
            "env_options": {
                "n_total_timesteps": 1e6,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    env.reset()
    action = dict(DQN.get_hpo_config_space().sample_configuration())
    env.step(action)
    
    with pytest.raises(ValueError) as excinfo:
        env.step(action)
    
    assert "Called step() before reset()" in str(excinfo.value)


if __name__ == "__main__":
    test_autorl_env_dqn_single_objective()
from arlbench.autorl_env import AutoRLEnv
import gymnax

from arlbench.agents import (
    PPO,
    DQN
)

def test_autorl_env_ppo_grad_obs():
    CONFIG = {
        "algorithm": "ppo",
        "objective": "reward",
        "n_steps": 10,
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": True
    }

    cartpole = gymnax.make("CartPole-v1")

    envs = {
        0: {
            "env": cartpole[0],
            "env_params": cartpole[1],
            "env_options": {
                "n_total_timesteps": 1e5,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    init_obs, _ = env.reset()
    assert init_obs.shape == (4,)

    action = dict(PPO.get_default_configuration())

    n_obs, reward, done, _, _ = env.step(action)
    assert n_obs.shape == (4,)
    assert done is False
    assert reward > 0

def test_autorl_env_dqn_grad_obs():
    CONFIG = {
        "algorithm": "dqn",
        "objective": "reward",
        "n_steps": 10,
        "n_eval_episodes": 10,
        "track_trajectories": False,
        "grad_obs": True
    }

    cartpole = gymnax.make("CartPole-v1")

    envs = {
        0: {
            "env": cartpole[0],
            "env_params": cartpole[1],
            "env_options": {
                "n_total_timesteps": 1e5,
                "n_env_steps": 500,
                "n_envs": 10,
            }
        }
    }

    env = AutoRLEnv(CONFIG, envs)
    init_obs, _ = env.reset()
    assert init_obs.shape == (4,)

    done = False
    steps = 0
    while not done:
        action = dict(DQN.get_configuration_space().sample_configuration())

        n_obs, reward, done, _, _ = env.step(action)
        steps += 1
        assert n_obs.shape == (4,)
        assert reward > 0
        if steps < 10:
            assert done is False
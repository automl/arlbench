from arlbench.autorl_env import AutoRLEnv
import gymnax

from arlbench.agents import (
    PPO,
    DQN
)
from arlbench.utils import (
    make_env,
)

def test_autorl_env_ppo_cartpole():
    CONFIG = {
        "algorithm": "ppo",
        "objective": "reward",
        "n_steps": 10,
        "n_eval_episodes": 10,
        "track_metrics": False,
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

    action = dict(PPO.get_default_configuration())

    n_obs, n_state, reward, done, _ = env.step(action)
    print(n_obs)
    print(reward)

test_autorl_env_ppo_cartpole()

from __future__ import annotations

import time

import jax
import numpy as np

from arlbench.core.algorithms import DQN, PPO
from arlbench.core.environments import make_env


def test_atari_ppo():
    training_kw_args = {
        "n_total_timesteps": 1e5,
    }

    env = make_env("envpool", "Adventure-v5", cnn_policy=True, n_envs=10, seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed
    config = PPO.get_default_hpo_config()
    config["buffer_size"] = 512
    algorithm = PPO(config, env, cnn_policy=True)
    algorithm_state = algorithm.init(rng)

    start = time.time()
    algorithm_state, result = algorithm.train(*algorithm_state, **training_kw_args)
    training_time = time.time() - start
    reward = np.mean(result.eval_rewards[-1])

    #  assert reward > -500
    print(reward, training_time)


def test_atari_dqn():
    training_kw_args = {
        "n_total_timesteps": 1e5,
    }

    env = make_env("envpool", "Adventure-v5", cnn_policy=True, n_envs=10, seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed
    config = DQN.get_default_hpo_config()
    algorithm = DQN(config, env, cnn_policy=True)
    algorithm_state = algorithm.init(rng)

    start = time.time()
    algorithm_state, result = algorithm.train(*algorithm_state, **training_kw_args)
    training_time = time.time() - start
    reward = np.mean(result.eval_rewards[-1])

    # assert reward > -500
    print(reward, training_time)


if __name__ == "__main__":
    test_atari_ppo()
    # test_atari_dqn()

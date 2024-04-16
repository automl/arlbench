import time

import jax
import numpy as np

from arlbench.core.algorithms import SAC
from arlbench.core.environments import make_env

N_TOTAL_TIMESTEPS = 1e5
EVAL_STEPS = 10
EVAL_EPISODES = 1
N_ENVS = 10


def test_default_sac_continuous():
    env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=N_ENVS)
    rng = jax.random.PRNGKey(42)

    config = SAC.get_default_hpo_config()
    agent = SAC(config, env)
    algorithm_state = agent.init(rng)
    
    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_TOTAL_TIMESTEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = results.eval_rewards.mean(axis=1)

    print(reward, training_time, algorithm_state.runner_state.global_step)
    assert reward > -1200    


if __name__ == "__main__":
    test_default_sac_continuous()
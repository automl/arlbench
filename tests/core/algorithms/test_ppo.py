import time
import warnings

import jax
import numpy as np

from arlbench.core.algorithms import PPO
from arlbench.core.environments import make_env

N_TOTAL_TIMESTEPS = 1e5
EVAL_STEPS = 1
EVAL_EPISODES = 1
N_ENVS = 1


def test_default_ppo_discrete(n_envs=10):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    runner_state, buffer_state, _ = agent.train(
        runner_state,
        buffer_state,
        n_total_timesteps=N_TOTAL_TIMESTEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    rewards = agent.eval(runner_state, EVAL_EPISODES)
    reward = np.mean(rewards)

    print(f"n_envs = {n_envs}, time = {training_time:.2f}, steps = {runner_state.global_step}, reward = {reward:.2f}")
    # assert reward > 450    


def test_default_ppo_continuous():
    env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=N_ENVS)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    runner_state, buffer_state, _ = agent.train(
        runner_state,
        buffer_state,
        n_total_timesteps=N_TOTAL_TIMESTEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    rewards = agent.eval(runner_state, EVAL_EPISODES)
    reward = np.mean(rewards)

    print(reward, training_time, runner_state.global_step)
    assert reward > -1200    


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_default_ppo_discrete(n_envs=1)
        test_default_ppo_discrete(n_envs=10)
    # test_default_ppo_continuous()
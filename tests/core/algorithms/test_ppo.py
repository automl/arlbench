import time

import jax
import numpy as np

from arlbench.core.algorithms import PPO
from arlbench.core.environments import make_env


N_TOTAL_TIMESTEPS = 1e5
EVAL_STEPS = 10
EVAL_EPISODES = 1
N_ENVS = 10


def test_default_ppo_discrete():
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=N_ENVS)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _, _, _) = agent.train(
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
    assert reward > 450    


def test_default_ppo_continuous():
    env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=N_ENVS)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _, _, _) = agent.train(
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
    test_default_ppo_discrete()
    test_default_ppo_continuous()
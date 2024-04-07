import time

import jax
import numpy as np

from arlbench.core.algorithms import PPO
from arlbench.core.environments import make_env

PPO_OPTIONS = {
    "n_total_timesteps": 1e5,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}


def test_default_ppo_discrete():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, PPO_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)

    assert reward > 450    
    print(reward, training_time)


def test_default_ppo_continuous():
    env = make_env("gymnax", "Pendulum-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, PPO_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)

    assert reward > -1200    
    print(reward, training_time)


if __name__ == "__main__":
    test_default_ppo_discrete()
    test_default_ppo_continuous()
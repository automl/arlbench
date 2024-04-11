import time

import jax
import numpy as np

from arlbench.core.algorithms import DQN
from arlbench.core.environments import make_env

N_TOTAL_TIMESTEPS = 1e6
EVAL_STEPS = 10
EVAL_EPISODES = 10
N_ENVS = 10


# Default hyperparameter configuration
def test_default_dqn():
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=N_ENVS)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, env)
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
    assert reward > 400

# uniform experience replay
def test_uniform_dqn():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed

    config = DQN.get_default_hpo_config()
    config["buffer_prio_sampling"] = False
    agent = DQN(config, env)
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
    
    assert reward > 400
    print(reward, training_time, runner_state.global_step)

# no target network
def test_no_target_dqn():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed

    config = DQN.get_default_hpo_config()
    config["use_target_network"] = False
    agent = DQN(config, env)
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
    
    assert reward > 400
    print(reward, training_time, runner_state.global_step)

# ReLU activation
def test_relu_dqn():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed

    config = DQN.get_default_hpo_config()
    config["activation"] = "relu"
    agent = DQN(config, env)
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
    
    assert reward > 400
    print(reward, training_time, runner_state.global_step)


if __name__ == "__main__":
    test_default_dqn()
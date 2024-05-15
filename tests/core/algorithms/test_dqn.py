from __future__ import annotations
import time
import warnings

import jax
from arlbench.core.algorithms import DQN
from arlbench.core.environments import make_env

N_UPDATES = 1e6
EVAL_STEPS = 10
EVAL_EPISODES = 128
N_ENVS = 10
GRADIENT_STEPS = 10


# Default hyperparameter configuration
def test_default_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    eval_env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=EVAL_EPISODES)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["gradient_steps"] = GRADIENT_STEPS
    config["target_update_interval"] = 10
    config["buffer_batch_size"] = 64
    config["learning_rate"] = 1e-3
    agent = DQN(config, env, eval_env=eval_env)
    algorithm_state = agent.init(rng)

    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES,
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()

    print(
        f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}"
    )
    assert reward > 400


# Default hyperparameter configuration with prioritised experience replay
def test_prioritised_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    eval_env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=EVAL_EPISODES)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["gradient_steps"] = GRADIENT_STEPS
    config["buffer_prio_sampling"] = True
    config["target_update_interval"] = 10
    config["buffer_batch_size"] = 64
    config["learning_rate"] = 1e-3
    agent = DQN(config, env, eval_env=eval_env)
    algorithm_state = agent.init(rng)

    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES,
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()

    print(
        f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}"
    )
    assert reward > 400


# Normalise observations
def test_normalise_obs_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    eval_env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=EVAL_EPISODES)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["gradient_steps"] = GRADIENT_STEPS
    config["target_update_interval"] = 10
    config["buffer_batch_size"] = 64
    config["learning_rate"] = 1e-3
    config["normalize_observations"] = True
    agent = DQN(config, env, eval_env=eval_env)
    algorithm_state = agent.init(rng)

    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES,
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()

    print(
        f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}"
    )
    assert reward > 400


# no target network
def test_no_target_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    eval_env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=EVAL_EPISODES)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["gradient_steps"] = GRADIENT_STEPS
    config["buffer_batch_size"] = 64
    config["use_target_network"] = False
    agent = DQN(config, env, eval_env=eval_env)
    algorithm_state = agent.init(rng)

    start = time.time()
    algorithm_state, result = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES,
    )
    training_time = time.time() - start
    reward = result.eval_rewards[-1].mean()

    print(
        f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}"
    )
    assert reward > 400


# ReLU activation
def test_relu_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    eval_env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=EVAL_EPISODES)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["gradient_steps"] = GRADIENT_STEPS
    config["target_update_interval"] = 10
    config["buffer_batch_size"] = 64
    config["learning_rate"] = 1e-3
    nas_config = DQN.get_default_nas_config()
    nas_config["activation"] = "relu"
    agent = DQN(config, env, eval_env=eval_env, nas_config=nas_config)
    algorithm_state = agent.init(rng)

    start = time.time()
    algorithm_state, result = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES,
    )
    training_time = time.time() - start
    reward = result.eval_rewards[-1].mean()

    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")

    assert reward > 400


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_default_dqn()

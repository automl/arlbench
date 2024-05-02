from __future__ import annotations

import time
import warnings

import jax
from arlbench.core.algorithms import SAC
from arlbench.core.environments import make_env

TRAINING_STEPS = 5e4
EVAL_STEPS = 10
EVAL_EPISODES = 128
N_ENVS = 1


# Normalize observations
def test_normalize_observations_sac(n_envs=N_ENVS):
    env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=n_envs)
    eval_env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=EVAL_EPISODES)
    rng = jax.random.PRNGKey(42)

    config = SAC.get_default_hpo_config()
    config["normalize_observations"] = True
    nas_config = SAC.get_default_nas_config()
    nas_config["hidden_size"] = 64
    agent = SAC(config, env, eval_env=eval_env, nas_config=nas_config)
    algorithm_state = agent.init(rng)

    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=TRAINING_STEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES,
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()

    print(
        f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}"
    )
    assert reward > -300


# Default SAC
#def test_default_sac(n_envs=N_ENVS):
#    env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=n_envs)
#    eval_env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=EVAL_EPISODES)
#    rng = jax.random.PRNGKey(42)
#
#    config = SAC.get_default_hpo_config()
#    nas_config = SAC.get_default_nas_config()
#    nas_config["hidden_size"] = 64
#    agent = SAC(config, env, eval_env=eval_env, nas_config=nas_config)
#    algorithm_state = agent.init(rng)
#
#    start = time.time()
#    algorithm_state, results = agent.train(
#        *algorithm_state,
#        n_total_timesteps=TRAINING_STEPS,
#        n_eval_steps=EVAL_STEPS,
#        n_eval_episodes=EVAL_EPISODES,
#    )
#    training_time = time.time() - start
#    reward = results.eval_rewards[-1].mean()
#
#    print(
#        f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}"
#    )
#    assert reward > -300
#
#
## Gradient steps
#def test_gradient_steps_sac(n_envs=N_ENVS):
#    env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=n_envs)
#    eval_env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=EVAL_EPISODES)
#    rng = jax.random.PRNGKey(42)
#
#    config = SAC.get_default_hpo_config()
#    config["gradient_steps"] = 4
#    nas_config = SAC.get_default_nas_config()
#    nas_config["hidden_size"] = 64
#    agent = SAC(config, env, eval_env=eval_env, nas_config=nas_config)
#    algorithm_state = agent.init(rng)
#
#    start = time.time()
#    algorithm_state, results = agent.train(
#        *algorithm_state,
#        n_total_timesteps=TRAINING_STEPS,
#        n_eval_steps=EVAL_STEPS,
#        n_eval_episodes=EVAL_EPISODES,
#    )
#    training_time = time.time() - start
#    reward = results.eval_rewards[-1].mean()
#
#    print(
#        f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}"
#    )
#    assert reward > -300

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default_sac_continuous(n_envs=1)
        default_sac_continuous(n_envs=10)

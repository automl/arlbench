import time

import jax
import warnings

from arlbench.core.algorithms import DQN
from arlbench.core.environments import make_env

N_UPDATES = 1e6
EVAL_STEPS = 10
EVAL_EPISODES = 5
N_ENVS = 10


# Default hyperparameter configuration
def test_default_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, env)
    algorithm_state = agent.init(rng)
    
    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES * n_envs,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()
    
    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")
    assert reward > 400


def test_gradient_steps_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["gradient_steps"] = 4
    agent = DQN(config, env)
    algorithm_state = agent.init(rng)

    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES * n_envs,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()

    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")
    assert reward > 400

# uniform experience replay
def test_uniform_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["buffer_prio_sampling"] = False
    agent = DQN(config, env)
    algorithm_state = agent.init(rng)
    
    start = time.time()
    algorithm_state, result = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES * n_envs,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = result.eval_rewards[-1].mean()
    
    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")
    assert reward > 400

# no target network
def test_no_target_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["use_target_network"] = False
    agent = DQN(config, env)
    algorithm_state = agent.init(rng)
    
    start = time.time()
    algorithm_state, result = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES * n_envs,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = result.eval_reward[-1].mean()
    
    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")
    assert reward > 400

# ReLU activation
def test_relu_dqn(n_envs=N_ENVS):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["activation"] = "relu"
    agent = DQN(config, env)
    algorithm_state = agent.init(rng)
    
    start = time.time()
    algorithm_state, result = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES * n_envs,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = result.eval_rewards[-1].mean()
    
    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")
    assert reward > 400


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_default_dqn(n_envs=1)
        test_default_dqn(n_envs=10)

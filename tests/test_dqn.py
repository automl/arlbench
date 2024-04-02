import jax
import time
import numpy as np

from arlbench.algorithms import DQN

from arlbench.utils import (
    make_env,
)

DQN_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}

# Default hyperparameter configuration
def test_default_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > 400
    print(reward, training_time)

# uniform experience replay
def test_per_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["buffer_prio_sampling"] = True
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)

    print(reward, training_time)
    assert reward > 300

# no target network
def test_no_target_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["use_target_network"] = False
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)

    print(reward, training_time)
    assert reward > 300

# ReLU activation
def test_relu_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    config["activation"] = "relu"
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)

    print(reward, training_time)
    assert reward > 300

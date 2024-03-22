import jax
import time

from arlbench.agents import DQN

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

    config = DQN.get_default_configuration()
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    assert reward > 400
    print(reward, training_time)

# uniform experience replay
def test_uniform_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_configuration()
    config["buffer_alpha"] = 0.
    config["buffer_beta"] = 1.
    config["buffer_epsilon"] = 0.
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    assert reward > 300
    print(reward, training_time)

# no target network
def test_no_target_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_configuration()
    config["use_target_network"] = False
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    assert reward > 300
    print(reward, training_time)

# ReLU activation
def test_relu_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_configuration()
    config["activation"] = 1    # + ReLU
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    assert reward > 300
    print(reward, training_time)


if __name__ == "__main__":
    test_default_dqn()
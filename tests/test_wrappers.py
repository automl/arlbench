import jax
import time
import numpy as np
import jax.numpy as jnp

from arlbench.algorithms import PPO
from arlbench.algorithms import DQN
from arlbench.environments import make_env


PPO_OPTIONS = {
    "n_total_timesteps": 1e5,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}

DQN_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}


def test_vw_gymnax():
    env = make_env("gymnax", "CartPole-v1", n_envs=PPO_OPTIONS["n_envs"], seed=42)
    rng = jax.random.PRNGKey(42)
    env_state, obs = env.reset(rng)
    action = jnp.array(
                [
                    env.action_space.sample(rng)
                    for _ in range(PPO_OPTIONS["n_envs"])
                ]
    )

    env_state, (obs, reward, done, info) = env.step(env_state, action, rng)

    assert(obs.shape == (PPO_OPTIONS["n_envs"], 4,))
    assert(reward.shape == (PPO_OPTIONS["n_envs"],))
    assert(done.shape == (PPO_OPTIONS["n_envs"],))
    assert(isinstance(info, dict))


def test_vw_envpool():
    env = make_env("envpool", "CartPole-v1", n_envs=PPO_OPTIONS["n_envs"], seed=42)
    rng = None      # we don't need rng here, envpool handles seeding 
    env_state, obs = env.reset(rng)
    action = jnp.array(
                [
                    env.action_space.sample(rng)
                    for _ in range(PPO_OPTIONS["n_envs"])
                ]
    )
    
    env_state, (obs, reward, done, info) = env.step(env_state, action, rng)

    assert(obs.shape == (PPO_OPTIONS["n_envs"], 4,))
    assert(reward.shape == (PPO_OPTIONS["n_envs"],))
    assert(done.shape == (PPO_OPTIONS["n_envs"],))
    assert(isinstance(info, dict))


def test_vw_envpool_ppo():
    env = make_env("envpool", "Pendulum-v1", n_envs=PPO_OPTIONS["n_envs"], seed=42)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, PPO_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start

    rewards = agent.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > -1500      # TODO check if is this reasonable?
    print(reward, training_time)


def test_vw_gymnax_ppo():
    env = make_env("gymnax", "Pendulum-v1", n_envs=PPO_OPTIONS["n_envs"], seed=42)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, PPO_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start

    rewards = agent.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)

    assert reward > -1500      # TODO check if is this reasonable?
    print(reward, training_time)


def test_vw_envpool_dqn():
    env = make_env("envpool", "CartPole-v1", n_envs=PPO_OPTIONS["n_envs"], seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start

    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > 100
    print(reward, training_time)


def test_vw_gymnax_dqn():
    env = make_env("gymnax", "CartPole-v1", n_envs=PPO_OPTIONS["n_envs"], seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start

    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > 400
    print(reward, training_time)


if __name__ == "__main__":
    #test_vw_gymnax()
    #test_vw_envpool() 
    #test_vw_envpool_ppo()
    #test_vw_gymnax_ppo()
    #test_vw_envpool_dqn()
    test_vw_gymnax_dqn()
import jax
import time
import jax.numpy as jnp
import numpy as np

from arlbench.core.algorithms import DQN, PPO, SAC

from arlbench.core.environments import make_env


def test_gymnasium_dqn():
    options = {
    "n_total_timesteps": 1e6,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}
    
    env = make_env("gymnasium", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed

    config = DQN.get_default_hpo_config()
    agent = DQN(config, options, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, options["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > 400
    print(reward, training_time)


def test_gymnasium_ppo():
    options = {
        "n_total_timesteps": 1e6,
        "n_env_steps": 200,
        "n_eval_episodes": 10,
        "track_metrics": False,
        "track_traj": False,
    }
        
    env = make_env("gymnasium", "Pendulum-v1", seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed

    config = PPO.get_default_hpo_config()
    agent = PPO(config, options, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, options["n_eval_episodes"])
    reward = np.mean(rewards)
    
    #assert reward > -500
    print(reward, training_time)


def test_gymnasium_sac():
    options = {
        "n_total_timesteps": 1e5,
        "n_env_steps": 200,
        "n_eval_episodes": 10,
        "track_metrics": False,
        "track_traj": False,
        "n_eval_steps": 1e4
    }
        
    env = make_env("gymnasium", "Pendulum-v1", seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed

    config = SAC.get_default_hpo_config()
    agent = SAC(config, options, env)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, options["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > -500
    print(reward, training_time)

if __name__ == "__main__":
    test_gymnasium_ppo()
    

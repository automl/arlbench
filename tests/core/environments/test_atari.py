import time

import jax
import jax.numpy as jnp
import numpy as np

from arlbench.core.algorithms import DQN, PPO, SAC
from arlbench.core.environments import make_env


def test_atari_ppo():
    options = {
        "n_total_timesteps": 1e5,
        "n_env_steps": 200,
        "n_eval_episodes": 10,
        "track_metrics": True,
        "track_traj": True,
    }
        
    env = make_env("gymnasium", "ALE/Adventure-v5", cnn_policy=True, n_envs=10, seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed
    config = PPO.get_default_hpo_config()
    agent = PPO(config, options, env, cnn_policy=True, track_metrics=options["track_metrics"], track_trajectories=options["track_traj"])
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, options["n_eval_episodes"])
    reward = np.mean(rewards)
    
    #assert reward > -500
    print(reward, training_time)

def test_atari_dqn():
    options = {
        "n_total_timesteps": 1e5,
        "n_env_steps": 200,
        "n_eval_episodes": 10,
        "track_metrics": True,
        "track_traj": True,
    }
        
    env = make_env("gymnasium", "ALE/Adventure-v5", cnn_policy=True, n_envs=10, seed=42)
    rng = jax.random.PRNGKey(43)  # todo: fix this seed
    config = DQN.get_default_hpo_config()
    agent = DQN(config, options, env, cnn_policy=True, track_metrics=options["track_metrics"], track_trajectories=options["track_traj"])
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, options["n_eval_episodes"])
    reward = np.mean(rewards)
    
    #assert reward > -500
    print(reward, training_time)


if __name__ == "__main__":
    test_atari_ppo()
    test_atari_dqn()
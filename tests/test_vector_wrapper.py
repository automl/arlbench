from arlbench.utils.wrappers import EnvpoolToGymnaxWrapper
import envpool
import jax
import time
import numpy as np
import gymnax

from arlbench.algorithms import PPO as DEFAULTPPO
from arlbench.algorithms.ppo_envpool import PPO as ENVPOOLPPO


PPO_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}


def test_vw_envpool_ppo():
    env = envpool.make("Pendulum-v1", env_type="gymnasium", num_envs=PPO_OPTIONS["n_envs"], seed=42)
    rng = jax.random.PRNGKey(42)

    config = ENVPOOLPPO.get_default_hpo_config()
    agent = ENVPOOLPPO(config, PPO_OPTIONS, env, None)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start

    eval_env = envpool.make("Pendulum-v1", env_type="gymnasium", num_envs=1)
    rewards = agent.eval_(runner_state, PPO_OPTIONS["n_eval_episodes"], eval_env)
    reward = np.mean(rewards)
    
    #assert reward > 400
    print(reward, training_time)

if __name__ == "__main__":
    test_envpool_ppo() 
    test_gymnax_ppo()
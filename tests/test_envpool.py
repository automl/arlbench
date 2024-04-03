from arlbench.utils.wrappers import EnvpoolToGymnaxWrapper
import envpool
import jax
import time
import numpy as np

from arlbench.algorithms.ppo_envpool import PPO


PPO_OPTIONS = {
    "n_total_timesteps": 1e5,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}


def test_envpool_ppo():
    env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=PPO_OPTIONS["n_envs"])
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, PPO_OPTIONS, env, None)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > 400
    print(reward, training_time)

if __name__ == "__main__":
    test_envpool_ppo() 
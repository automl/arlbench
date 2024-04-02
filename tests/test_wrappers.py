from arlbench.utils.wrappers import EnvpoolToGymnaxWrapper
import envpool
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
    env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=100)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env, None)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    
    assert reward > 400
    print(reward, training_time)
import jax
import time

from arlbench.agents import (
    PPO,
    DQN
)

from arlbench.utils import (
    make_env,
)

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

PPO_CONFIG = PPO.get_default_configuration()
DQN_CONFIG = DQN.get_default_configuration()

def test_oop_ppo():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    agent = PPO(PPO_CONFIG, PPO_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    assert reward > 400
    print(reward, training_time)

    

def test_oop_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    agent = DQN(DQN_CONFIG, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    assert reward > 400
    print(reward, training_time)

test_oop_dqn()


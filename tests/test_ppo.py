import jax
import time

from arlbench.agents import PPO

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

def test_default_ppo():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, PPO_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    assert reward > 450    
    print(reward, training_time)

test_default_ppo()


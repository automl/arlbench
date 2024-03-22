import jax
import time

from arlbench.agents import SAC

from arlbench.utils import (
    make_env,
)

SAC_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}

# Default hyperparameter configuration
def test_default_sac():
    env, env_params = make_env("gymnax", "Pendulum-v1")
    rng = jax.random.PRNGKey(42)

    config = SAC.get_default_configuration()
    agent = SAC(config, SAC_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)
    
    start = time.time()
    (runner_state, _), _ = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    reward = agent.eval(runner_state, SAC_OPTIONS["n_eval_episodes"])
    assert reward > 400
    print(reward, training_time)


if __name__ == "__main__":
    test_default_sac()
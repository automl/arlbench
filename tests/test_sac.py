import jax
import time

from arlbench.agents import SAC

from stable_baselines3 import SAC as SB3SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

from arlbench.utils import (
    make_env,
)

SAC_OPTIONS = {
    "n_total_timesteps": 1e5,
    "n_envs": 1,
    "n_env_steps": 200,
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
    reward = agent.sac_eval(runner_state, SAC_OPTIONS["n_eval_episodes"])
    assert reward > -200
    print(reward, training_time)


def test_sb3_sac():
    env = gym.make("Pendulum-v1")
    policy_kwargs = dict(
        net_arch=[64, 64]  # Two hidden layers with 64 units each
    )
    model = SB3SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    start = time.time()
    model.learn(total_timesteps=int(SAC_OPTIONS["n_total_timesteps"]))
    training_time = time.time() - start

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=SAC_OPTIONS["n_eval_episodes"])
    assert mean_reward > -200
    print(mean_reward, training_time)


if __name__ == "__main__":
    #test_sb3_sac()
    with jax.disable_jit(disable=False):
        test_default_sac()
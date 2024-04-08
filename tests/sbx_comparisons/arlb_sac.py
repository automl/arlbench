import jax
import time

from arlbench.algorithms import SAC

from stable_baselines3 import SAC as SB3SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
import gymnasium as gym

from brax import envs as brax_envs

from brax.envs.wrappers import gym as gym_wrapper

from sbx.sac import SAC as SBXSAC
from arlbench.environments import make_env

SAC_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_envs": 1,
    "n_env_steps": 1000,
    "n_eval_steps": 100,
    "n_eval_episodes": 100,
    "track_metrics": False,
    "track_traj": False,
}

# Default hyperparameter configuration
def test_default_sac():
    #env, env_params = make_env("gym", "LunarLanderContinuous-v2")
    env = make_env("brax", "ant", n_envs=SAC_OPTIONS["n_envs"], seed=42)
    #env, env_params = make_env("brax", "ant")
    rng = jax.random.PRNGKey(42)

    hpo_config = SAC.get_default_hpo_config()
    hpo_config["tau"] = 0.005
    hpo_config["learning_starts"] = 10000
    hpo_config["buffer_alpha"] = 0.0
    hpo_config["buffer_beta"] = 0.0
    nas_config = SAC.get_default_nas_config()
    nas_config["activation"] = "relu"
    nas_config["hidden_size"] = 256

    agent = SAC(hpo_config, SAC_OPTIONS, env, nas_config)
    runner_state, buffer_state = agent.init(rng)

    start = time.time()
    (runner_state, _), (reward, _) = agent.train(runner_state, buffer_state)
    training_time = time.time() - start
    #reward = agent.sac_eval(runner_state, SAC_OPTIONS["n_eval_episodes"])
    #assert reward > -200
    print(reward, training_time)


def test_sb3_sac():
    # Function to create the environment
    def make_env():
        def _init():
            #return gym.make("LunarLanderContinuous-v2")
            return env
        return _init

    # Create multiple environments
    env = brax_envs.create("ant",
                           episode_length=SAC_OPTIONS["n_env_steps"],
                           backend='spring')
    env = gym_wrapper.GymWrapper(env)
    #env = torch_wrapper.TorchWrapper(env, device="cpu")


    policy_kwargs = dict(
        net_arch=[256, 256]  # Two hidden layers with 64 units each
    )
    model = SB3SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=5)

    start = time.time()
    model.learn(total_timesteps=int(SAC_OPTIONS["n_total_timesteps"]))
    training_time = time.time() - start

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=SAC_OPTIONS["n_eval_episodes"])
    #assert mean_reward > -200
    print(mean_reward, training_time)


def test_sb3_jax_sac():
    # Function to create the environment
    #def make_env():
    #    def _init():
    #        return GymWrapper(brax_envs.training.wrap(
    #            brax_envs.get_environment("ant", backend="spring"), episode_length=SAC_OPTIONS["n_env_steps"], action_repeat=1)
    #        )
    #        #return gym.make("Pendulum-v1")
    #    return _init

    # Create multiple environments
    #env = VecMonitor(SubprocVecEnv([make_env() for _ in range(1)]))
    env = brax_envs.create("ant", batch_size=None,
                           episode_length=SAC_OPTIONS["n_env_steps"],
                           backend='spring')
    env = gym_wrapper.GymWrapper(env)
    # env = torch_wrapper.TorchWrapper(env, device="cpu")
    #env = VecMonitor(env)
    # automatically convert between jax ndarrays and torch tensors:
    #env = torch_wrapper.TorchWrapper(env, device=device)

    policy_kwargs = dict(
        net_arch=[256, 256]  # Two hidden layers with 64 units each
    )
    model = SBXSAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=5)

    start = time.time()
    model.learn(total_timesteps=int(SAC_OPTIONS["n_total_timesteps"]))
    training_time = time.time() - start

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=SAC_OPTIONS["n_eval_episodes"])
    #assert mean_reward > -200
    print(mean_reward, training_time)


if __name__ == "__main__":
    #test_sb3_sac()
    with jax.disable_jit(disable=False):
        #test_sb3_jax_sac()
        test_default_sac()

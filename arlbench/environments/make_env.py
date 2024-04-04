
from gymnax.wrappers.purerl import FlattenObservationWrapper
from arlbench.environments import Environment
from arlbench.environments.wrappers import GymnaxWrapper, GymToGymnaxWrapper, BraxToGymnaxWrapper, EnvpoolWrapper


def make_env(env_framework, env_name, n_envs=10, seed=0) -> Environment:
    if env_framework == "gymnasium":
        import gymnasium

        env = gymnasium.make(env_name, autoreset=True)
        env = GymToGymnaxWrapper(env)
        env = FlattenObservationWrapper(env)
        env = GymnaxWrapper(env, n_envs, None)
    elif env_framework == "gymnax":
        import gymnax

        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)
        env = GymnaxWrapper(env, n_envs, env_params)
    elif env_framework == "envpool":
        import envpool

        env = envpool.make(env_name, env_type="gymnasium", num_envs=n_envs, seed=seed)
        env = FlattenObservationWrapper(env)
        env = EnvpoolWrapper(env, n_envs)
    elif env_framework == "brax":
        from brax import envs

        env = envs.get_environment(env_name, backend="generalized")
        env = envs.training.wrap(env)
        env = BraxToGymnaxWrapper(env)
        env = FlattenObservationWrapper(env)
        env = GymnaxWrapper(env, n_envs, None)
    else:
        raise ValueError(f"Invalid framework: {env_framework}")
    
    return env


from arlbench.environments import AutoRLEnv, GymnaxEnv, BraxEnv, GymnasiumEnv, EnvpoolEnv
from arlbench.wrappers import FlattenObservationWrapper, AutoRLWrapper
from typing import Union


def make_env(env_framework, env_name, n_envs=10, seed=0) -> Union[AutoRLEnv, AutoRLWrapper]:
    if env_framework == "gymnasium":
        import gymnasium

        env = gymnasium.vector.make(env_name, num_envs=n_envs)
        env = GymnasiumEnv(env, n_envs)
    elif env_framework == "gymnax":
        import gymnax

        env, env_params = gymnax.make(env_name)
        env = GymnaxEnv(env, n_envs, env_params)
    elif env_framework == "envpool":
        import envpool

        env = envpool.make(env_name, env_type="gymnasium", num_envs=n_envs, seed=seed)
        env = EnvpoolEnv(env, n_envs)
    elif env_framework == "brax":
        from brax import envs

        env = envs.get_environment(env_name, backend="generalized")
        env = envs.training.wrap(env)
        env = BraxEnv(env, n_envs)
    else:
        raise ValueError(f"Invalid framework: {env_framework}")
    
    env = FlattenObservationWrapper(env)
    return env

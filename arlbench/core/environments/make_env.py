from __future__ import annotations

from typing import TYPE_CHECKING
import warnings
from arlbench.core.wrappers import AutoRLWrapper, FlattenObservationWrapper

from .brax_env import BraxEnv
from .envpool_env import EnvpoolEnv
from .gymnasium_env import GymnasiumEnv
from .gymnax_env import GymnaxEnv

if TYPE_CHECKING:
    from .autorl_env import AutoRLEnv


def make_env(env_framework, env_name, n_envs=1, seed=0) -> AutoRLEnv | AutoRLWrapper:
    if env_framework == "gymnasium":
        if n_envs > 1:
            warnings.warn(f"For gymnasium only n_envs must be set to 1 but actual value is {n_envs}. n_envs will be set to 1.")
        import gymnasium

        env = gymnasium.make(env_name)
        env = GymnasiumEnv(env, seed)
    elif env_framework == "gymnax":
        import gymnax

        env, env_params = gymnax.make(env_name)
        env = GymnaxEnv(env, n_envs, env_params)
    elif env_framework == "envpool":
        from arlbench.core.environments import envpool_env

        env = envpool_env.make(env_name, env_type="gymnasium", num_envs=n_envs, seed=seed)
        env = EnvpoolEnv(env, n_envs)
    elif env_framework == "brax":
        from brax import envs

        env = envs.get_environment(env_name, backend="generalized")
        env = envs.training.wrap(env)
        env = BraxEnv(env, n_envs)
    else:
        raise ValueError(f"Invalid framework: {env_framework}")

    return FlattenObservationWrapper(env)

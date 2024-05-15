from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from arlbench.core.wrappers import AutoRLWrapper, FlattenObservationWrapper

if TYPE_CHECKING:
    from typing import Any

    from .autorl_env import Environment


def make_env(
    env_framework: str,
    env_name: str,
    cnn_policy: bool = False,
    n_envs: int = 1,
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
) -> Environment | AutoRLWrapper:
    # todo: add env_kwargs to all different environments
    if env_kwargs is None:
        env_kwargs = {}
    if env_framework == "gymnasium":
        if n_envs > 1:
            warnings.warn(
                f"For gymnasium only n_envs must be set to 1 but actual value is {n_envs}. n_envs will be set to 1."
            )
        from .gymnasium_env import GymnasiumEnv

        env = GymnasiumEnv(env_name, seed, env_kwargs=env_kwargs)
    elif env_framework == "gymnax":
        from .gymnax_env import GymnaxEnv

        env = GymnaxEnv(env_name, n_envs, env_kwargs=env_kwargs)
    elif env_framework == "envpool":
        from .envpool_env import EnvpoolEnv

        env = EnvpoolEnv(env_name, n_envs, seed, env_kwargs=env_kwargs)
    elif env_framework == "brax":
        from .brax_env import BraxEnv

        env = BraxEnv(env_name, n_envs, env_kwargs=env_kwargs)
    elif env_framework == "xland":
        from .xland_env import XLandEnv

        env = XLandEnv(env_name, n_envs, env_kwargs=env_kwargs, cnn_policy=cnn_policy)
    else:
        raise ValueError(f"Invalid framework: {env_framework}")

    if cnn_policy:
        return env
    else:
        return FlattenObservationWrapper(env)

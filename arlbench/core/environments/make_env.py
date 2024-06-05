"""Environment creation function for ARLBench."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from arlbench.core.wrappers import FlattenObservationWrapper, Wrapper

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
) -> Environment | Wrapper:
    """ARLBench equivalent to make_env in gymnasium/gymnax etc.
    Creates a JAX-compatible RL environment.

    Args:
        env_framework (str): Environment framework to use.
            Must be one of the following: brax, envpool, gymnasium, gymnax, xland
        env_name (str): Name/id of the environment. Has to match the env_framework.
        cnn_policy (bool, optional): _description_. Defaults to False.
        n_envs (int, optional): Number of environments. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 0.
        env_kwargs (dict[str, Any] | None, optional): Keyword arguments
            to pass to the environment. Defaults to None.

    Returns:
        Environment | Wrapper: JAX-compatible RL environment.
    """
    if env_kwargs is None:
        env_kwargs = {}
    if env_framework == "gymnasium":
        if n_envs > 1:
            warnings.warn(
                f"""For gymnasium only n_envs must be set to 1 but
                actual value is {n_envs}. n_envs will be set to 1."""
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

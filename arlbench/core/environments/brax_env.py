from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs

from .autorl_env import Environment

if TYPE_CHECKING:
    import chex
    from brax.envs.base import State as BraxEnvState
    from chex import PRNGKey
    from typing import Any


class BraxEnv(Environment):
    def __init__(self, env_name: str, n_envs: int, env_kwargs: dict[str, Any] = {}):
        env = envs.create(env_name, batch_size=n_envs, **env_kwargs)
        super().__init__(env_name, env, n_envs)
        self.max_steps_in_episode = 1000

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey) -> tuple[BraxEnvState, chex.Array]:
        env_state = self._env.reset(rng)
        return env_state, env_state.obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(
            self,
            env_state: BraxEnvState,
            action: chex.Array,
            rng: PRNGKey
        ) -> tuple[BraxEnvState, tuple[chex.Array, chex.Array, chex.Array, dict]]:
        env_state = self._env.step(env_state, action)
        return env_state, (env_state.obs, env_state.reward, env_state.done, {})

    @property
    def action_space(self):
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.action_size,)
        )

    @property
    def observation_space(self):
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.observation_size,)
        )


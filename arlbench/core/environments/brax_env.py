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


class BraxEnv(Environment):
    def __init__(self, env_name: str, n_envs: int):
        env = envs.get_environment(env_name, backend="spring")
        env = envs.training.wrap(env)
        super().__init__(env_name, env, n_envs)
        self.max_steps_in_episode = 1000

    @functools.partial(jax.jit, static_argnums=0)
    def __reset(self, rng: chex.PRNGKey) -> tuple[BraxEnvState, chex.Array]:
        """Internal reset in brax environment."""
        env_state = self._env.reset(rng=jnp.array([rng]))
        return env_state, env_state.obs[0]

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey) -> tuple[BraxEnvState, chex.Array]:
        reset_rng = jax.random.split(rng, self._n_envs)
        env_state, obs = jax.vmap(self.__reset, in_axes=(0))(
            reset_rng
        )
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=0)
    def __step(
        self,
        env_state: BraxEnvState,
        action: chex.Array
        ) -> tuple[BraxEnvState, tuple[chex.Array, chex.Array, chex.Array, dict]]:
        """Internal step in brax environment."""
        env_state = self._env.step(env_state, jnp.array([action]))
        return env_state, (env_state.obs[0], env_state.reward[0], env_state.done[0], {})

    @functools.partial(jax.jit, static_argnums=0)
    def step(
            self,
            env_state: BraxEnvState,
            action: chex.Array,
            rng: PRNGKey
        ) -> tuple[BraxEnvState, tuple[chex.Array, chex.Array, chex.Array, dict]]:
        env_state, (obs, reward, done, info) = jax.vmap(
            self.__step, in_axes=(0, 0)
        )(env_state, action)

        return env_state, (obs, reward, done, info)

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


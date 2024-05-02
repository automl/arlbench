from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnasium
import jax
import jax.numpy as jnp

from arlbench.utils import gymnasium_space_to_gymnax_space

from .autorl_env import Environment

if TYPE_CHECKING:
    import chex
    from chex import PRNGKey
    from typing import Any

class GymnasiumEnv(Environment):
    def __init__(self, env_name: str, seed: int, env_kwargs: dict[str, Any] = {}):
        env = gymnasium.make(env_name, **env_kwargs)
        super().__init__(env_name, env, 1, seed)

        self._reset_result = jnp.array(self._env.observation_space.sample())

        self._step_result = (
            jnp.array(self._env.observation_space.sample(), dtype=jnp.float32),
            jnp.array([1.], dtype=jnp.float32),
            jnp.array([False], dtype=jnp.bool),
            {}
        )

    @functools.partial(jax.jit, static_argnums=0)
    def __reset(self, _) -> chex.Array:
        def reset_env():
            obs, _ = self._env.reset(seed=self._seed)
            return obs

        return jax.pure_callback(
            reset_env, self._reset_result
        )

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey) -> tuple[None, chex.Array]:
        obs = jax.vmap(
            self.__reset, in_axes=(0)
        )(jnp.arange(self._n_envs))
        return None, obs

    @functools.partial(jax.jit, static_argnums=0)
    def __step(self, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, dict]:
        def step(_action):
            obs, reward, term, trunc, _ = self._env.step(_action)

            # Autoreset
            # https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/autoreset/
            if term or trunc:
                obs, _ = self._env.reset(seed=self._seed)

            obs = jnp.array(obs, dtype=jnp.float32)
            reward = jnp.array([reward], dtype=jnp.float32)
            done = jnp.array([term or trunc], dtype=jnp.bool)
            return obs, reward, done, {}

        obs, reward, done, _ = jax.pure_callback(
            step, self._step_result, action)
        return obs, reward[0], done[0], {}

    @functools.partial(jax.jit, static_argnums=0)
    def step(
            self,
            env_state: None,
            action: chex.Array,
            rng: PRNGKey
        ) -> tuple[None, tuple[chex.Array, chex.Array, chex.Array, dict]]:
        obs, reward, done, info = jax.vmap(
            self.__step, in_axes=(0)
        )(action)

        return None, (obs, reward, done, info)

    @property
    def action_space(self):
        return gymnasium_space_to_gymnax_space(self._env.action_space)

    @property
    def observation_space(self):
        return gymnasium_space_to_gymnax_space(self._env.observation_space)


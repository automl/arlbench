"""Gymnasium environment adapter."""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnasium
import jax
import jax.numpy as jnp

from arlbench.utils import gymnasium_space_to_gymnax_space

from .autorl_env import Environment

if TYPE_CHECKING:
    from typing import Any

    import chex
    from chex import PRNGKey


class GymnasiumEnv(Environment):
    """A gymnasium-based RL environment."""

    def __init__(
        self, env_name: str, seed: int, env_kwargs: dict[str, Any] | None = None
    ):
        """Creates a gymnasium environment for JAX-based RL training.

        Args:
            env_name (str): Name/id of the brax environment.
            seed (int): Random seed.
            env_kwargs (dict[str, Any] | None, optional): Keyword arguments
                to pass to the brax environment. Defaults to None.
        """
        if env_kwargs is None:
            env_kwargs = {}
        env = gymnasium.make(env_name, **env_kwargs)
        super().__init__(env_name, env, 1, seed)

        # For the JAX IO callback we need the shapes of the values returned
        # by step() and reset()
        self._reset_result = jnp.array(self._env.observation_space.sample())
        self._step_result = (
            jnp.array(self._env.observation_space.sample(), dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([False], dtype=jnp.bool),
            {},
        )

    @functools.partial(jax.jit, static_argnums=0)
    def __reset(self, _: None) -> jnp.ndarray:
        """Wraps the internal reset() function.

        Args:
            _ (None): Unused parameter.

        Returns:
            jnp.ndarray: Reset() result.
        """
        def reset_env():
            obs, _ = self._env.reset(seed=self._seed)
            return obs

        return jax.pure_callback(reset_env, self._reset_result)

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey) -> tuple[None, chex.Array]: # noqa: ARG002
        """Wraps actual reset for jitting."""
        obs = jax.vmap(self.__reset, in_axes=(0))(jnp.arange(self._n_envs))
        return None, obs

    @functools.partial(jax.jit, static_argnums=0)
    def __step(
        self, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Wraps the internal step() function.

        Args:
            action (jnp.ndarray): Action to take.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
                Step result: (obs, reward, done, info).
        """
        def step(_action: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
            """Internal step to match function data types and perform autoreset.

            Args:
                _action (jnp.ndarray): Action to take.

            Returns:
                tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
                    Step result: (obs, reward, done, info).
            """
            obs, reward, term, trunc, _ = self._env.step(_action)

            # Autoreset
            # https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/autoreset/
            if term or trunc:
                obs, _ = self._env.reset(seed=self._seed)

            obs = jnp.array(obs, dtype=jnp.float32)
            reward = jnp.array([reward], dtype=jnp.float32)
            done = jnp.array([term or trunc], dtype=jnp.bool)
            return obs, reward, done, {}

        obs, reward, done, _ = jax.pure_callback(step, self._step_result, action)
        return obs, reward[0], done[0], {}

    @functools.partial(jax.jit, static_argnums=0)
    def step(
        self, env_state: None, action: chex.Array, rng: PRNGKey # noqa: ARG002
    ) -> tuple[None, tuple[chex.Array, chex.Array, chex.Array, dict]]:
        """Wraps actual step for jitting."""
        obs, reward, done, info = jax.vmap(self.__step, in_axes=(0))(action)

        return None, (obs, reward, done, info)

    @property
    def action_space(self):
        """Action space of the environment."""
        return gymnasium_space_to_gymnax_space(self._env.action_space)

    @property
    def observation_space(self):
        """Observation space of the environment."""
        return gymnasium_space_to_gymnax_space(self._env.observation_space)

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces

from .wrapper import Wrapper

if TYPE_CHECKING:
    import chex

    from arlbench.core.environments import Environment


# TODO add test cases
class FlattenObservationWrapper(Wrapper):
    """Wraps the given environment to flatten its observations."""

    def __init__(self, env: Environment):
        """Wraps the given environment to flatten its observations.

        Args:
            env (Environment): Environment to wrap.
        """
        super().__init__(env)

    @property
    def observation_space(self) -> spaces.Box:
        """The flattened observation space of the environment.

        Returns:
            spaces.Box: Flattened obseration space.
        """
        assert isinstance(
            self._env.observation_space, spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space.low,
            high=self._env.observation_space.high,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def __flatten(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Flattens a batch of observations.

        Args:
            obs (jnp.ndarray): The observations to flatten.

        Returns:
            jnp.ndarray: The flattened observations.
        """
        # Since we have a stack of <n_envs> observations,
        # we want to keep the first dimension
        return obs.reshape(obs.shape[0], -1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: chex.PRNGKey) -> tuple[Any, jnp.ndarray]:
        """Calls the reset() function of the environment and flattens the returned observations.


        Args:
            rng (chex.PRNGKey): Random number generator key.

        Returns:
            tuple[Any, jnp.ndarray]: Result of the step function but observations are flattened.
        """
        env_state, obs = self._env.reset(rng)

        obs = self.__flatten(obs)
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        env_state: Any,
        action:
        jnp.ndarray,
        rng: chex.PRNGKey
    ) -> tuple[Any, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]]:
        """Calls the step() function of the environment and flattens the returned
        observations.

        Args:
            env_state (Any): The internal environment state.
            action (jnp.ndarray): The actions to take.
            rng (chex.PRNGKey): Random number generator key.

        Returns:
            tuple[Any, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]]: Result of
            the step function but observations are flattened.
        """
        env_state, (obs, reward, done, info) = self._env.step(env_state, action, rng)

        obs = self.__flatten(obs)
        return env_state, (obs, reward, done, info)

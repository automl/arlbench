from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import gymnax
    from chex import PRNGKey


class Environment(ABC):
    """An abstract environment class to support various kinds of RL environments.

    Sub-classes need to implement the following methods:
      - reset()
      - step()
    Note: Both functions need to be fully jittable to support JAX-based RL algorithms!

    As well as the properties:
      - action_space
      - observation_space
    Note: These need to be gymnax spaces, not gymnasium spaces.
    """
    def __init__(self, env_name: str, env: Any, n_envs: int, seed: int | None = None):
        """Creates a JAX-compatible RL environment. It is automatically vectorized to compute multiple environments simultaneously.

        Args:
            env_name (str): Name/id of the environment.
            env (Any): Inner/wrapped environment.
            n_envs (int): Number of vectorized environments.
            seed (int | None, optional): Random seed. Defaults to None.
        """
        self._env_name = env_name
        self._env = env
        self._n_envs = n_envs
        self._seed = seed

    @abstractmethod
    def reset(self, rng: PRNGKey) -> tuple[Any, Any]:
        """Environment reset() function. Resets the internal environment state.

        Args:
            rng (PRNGKey): Random number generator key.

        Returns:
            tuple[Any, Any]: Returns a tuple containing the environment state as well as the actual return of the reset() function.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, env_state: Any, action: Any, rng: PRNGKey
    ) -> tuple[Any, Any]:
        """Environment step() function. Performs a step in the environment given an action.

        Args:
            env_state (Any): Internal environment state.
            action (Any): Action to take.
            rng (PRNGKey): Random number generator key.

        Returns:
            tuple[Any, Any]:  Returns a tuple containing the environment state as well as the actual return of the step() function.
        """
        raise NotImplementedError

    @abstractmethod
    def action_space(self) -> gymnax.environments.spaces.Space:
        """The action space of the environment (gymnax space).

        Returns:
            gymnax.environments.spaces.Space: Action space of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def observation_space(self) -> gymnax.environments.spaces.Space:
        """The observation space of the environment (gymnax space).

        Returns:
            gymnax.environments.spaces.Space: Observation space of the environment.
        """
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=0)
    def sample_actions(self, rng: PRNGKey) -> jnp.ndarray:
        """Samples a random action for each environment.

        Args:
            rng (PRNGKey): Random number generator key.

        Returns:
            jnp.ndarray: Array of sampled actions, one for each environment.
        """
        _rngs = jax.random.split(rng, self._n_envs)
        return jnp.array(
            [self.action_space.sample(_rngs[i]) for i in range(self._n_envs)]
        )

    @property
    def n_envs(self) -> int:
        """The number of environments.

        Returns:
            int: _description_
        """
        return self._n_envs

    @property
    def env_name(self) -> str:
        """Returns the name/id of the environments.

        Returns:
            str: Environment name.
        """
        return self._env_name

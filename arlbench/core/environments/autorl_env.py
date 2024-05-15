from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from chex import PRNGKey


class Environment(ABC):
    def __init__(self, env_name: str, env: Any, n_envs: int, seed: int | None = None):
        self._env_name = env_name
        self._env = env
        self._n_envs = n_envs
        self._seed = seed

    @abstractmethod
    def reset(self, rng: PRNGKey) -> tuple[Any, Any]:  # TODO improve typing
        raise NotImplementedError

    @abstractmethod
    def step(
        self, env_state: Any, action: Any, rng: PRNGKey
    ) -> tuple[Any, Any]:  # TODO improve typing
        raise NotImplementedError

    @abstractmethod
    def action_space(self):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=0)
    def sample_actions(self, rng):
        _rngs = jax.random.split(rng, self._n_envs)
        return jnp.array(
            [self.action_space.sample(_rngs[i]) for i in range(self._n_envs)]
        )

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def env_name(self):
        return self._env_name

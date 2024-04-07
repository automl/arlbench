from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
import chex
import jax.numpy as jnp
import jax
import functools

if TYPE_CHECKING:
    from chex import PRNGKey


class AutoRLEnv(ABC):
    def __init__(self, env: Any, n_envs: int):
        self.env = env
        self.n_envs = n_envs

    @abstractmethod
    def reset(self, rng: PRNGKey) -> tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod
    def step(self, env_state: Any, action: Any, rng: PRNGKey) -> tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod
    def action_space(self):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self):
        raise NotImplementedError
    
    @functools.partial(jax.jit, static_argnums=0)
    def sample_actions(self, rng):
        _rngs = jax.random.split(rng, self.n_envs)
        return jnp.array(
            [
                self.action_space.sample(_rngs[i])
                for i in range(self.n_envs)
            ]
        )
    
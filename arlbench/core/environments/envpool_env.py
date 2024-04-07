from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp

from arlbench.utils import gymnasium_space_to_gymnax_space

from .autorl_env import AutoRLEnv


class EnvpoolEnv(AutoRLEnv):
    def __init__(self, env: Any, n_envs: int):
        super().__init__(env, n_envs)
        self.handle0_, _, _, self.xla_step_ = self.env.xla()

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, _):
        obs, _ = self.env.reset()
        return self.handle0_, obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, _):
        handle1, (obs, reward, term, trunc, info) = self.xla_step_(env_state, action)
        done = jnp.logical_or(term, trunc)
        return handle1, (obs, reward, done, info)

    @property
    def action_space(self):
        return gymnasium_space_to_gymnax_space(self.env.action_space)

    @property
    def observation_space(self):
        return gymnasium_space_to_gymnax_space(self.env.observation_space)


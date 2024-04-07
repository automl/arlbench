from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax

from .autorl_env import AutoRLEnv

if TYPE_CHECKING:
    from chex import PRNGKey
    from gymnax.environments.environment import Environment


class GymnaxEnv(AutoRLEnv):
    def __init__(self, env: Environment, n_envs: int, env_params: Any):
        super().__init__(env, n_envs)

        self.env_params = env_params

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey):
        reset_rng = jax.random.split(rng, self.n_envs)
        obs, env_state = jax.vmap(self._env.reset, in_axes=(0, None))(
            reset_rng, self.env_params
        )
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: PRNGKey):
        step_rng = jax.random.split(rng, self.n_envs)
        obs, env_state, reward, done, info = jax.vmap(
            self._env.step, in_axes=(0, 0, 0, None)
        )(step_rng, env_state, action, self.env_params)

        return env_state, (obs, reward, done, info)

    @property
    def action_space(self):
        return self._env.action_space(self.env_params)

    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, rng: PRNGKey):
        return self.action_space.sample(rng)

    @property
    def observation_space(self):
        return self._env.observation_space(self.env_params)


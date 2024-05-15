from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp

from .autorl_env import Environment

if TYPE_CHECKING:
    from chex import PRNGKey


class XLandEnv(Environment):
    def __init__(
        self,
        env_name: str,
        n_envs: int,
        env_kwargs: dict[str, Any] | None = None,
        cnn_policy: bool = False,
    ):
        if env_kwargs is None:
            env_kwargs = {}
        try:
            import xminigrid
            from xminigrid.experimental.img_obs import RGBImgObservationWrapper
            from xminigrid.wrappers import GymAutoResetWrapper
        except ImportError:
            raise ValueError(
                "Failed to import XLand. Please install the package first."
            )
        env, env_params = xminigrid.make(env_name, **env_kwargs)
        env = GymAutoResetWrapper(env)

        if cnn_policy:
            env = RGBImgObservationWrapper(env)
        super().__init__(env_name, env, n_envs)

        self.env_params = env_params

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey):
        reset_rng = jax.random.split(rng, self.n_envs)
        timestep = jax.vmap(self._env.reset, in_axes=(None, 0))(
            self.env_params, reset_rng
        )
        return timestep, timestep.observation

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: PRNGKey):
        timestep = jax.vmap(self._env.step, in_axes=(None, 0, 0))(
            self.env_params, env_state, action
        )  # env_state = timestep

        return timestep, (timestep.observation, timestep.reward, timestep.last(), {})

    @property
    def action_space(self):
        return gymnax.environments.spaces.Discrete(
            self._env.num_actions(self.env_params)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, rng: PRNGKey):
        return self.action_space.sample(rng)

    @property
    def observation_space(self):
        obs_shape = self._env.observation_shape(self.env_params)
        return gymnax.environments.spaces.Box(
            low=jnp.array([0 for _ in obs_shape]),
            high=jnp.array([s - 1 for s in obs_shape]),
            shape=self._env.observation_shape(self.env_params),
        )

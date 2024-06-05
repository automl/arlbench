"""XLand environment adapter."""
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
    """A xland-based RL environment."""
    def __init__(
        self,
        env_name: str,
        n_envs: int,
        env_kwargs: dict[str, Any] | None = None,
        cnn_policy: bool = False,
    ):
        """Creates an xland environment for JAX-based RL training.

        Args:
            env_name (str): Name/id of the brax environment.
            n_envs (int): Number of environments.
            env_kwargs (dict[str, Any] | None, optional): Keyword arguments to pass
                to the gymnax environment. Defaults to None.
            cnn_policy (bool, optional): Use a CNN-based policy (instead of MLP-based).
                Defaults to False.
        """
        if env_kwargs is None:
            env_kwargs = {}
        try:
            import xminigrid
            from xminigrid.experimental.img_obs import RGBImgObservationWrapper
            from xminigrid.wrappers import GymAutoResetWrapper
        except ImportError:
            raise ValueError(
                "Failed to import XLand. Please make sure the package is installed."
            )
        env, env_params = xminigrid.make(env_name, **env_kwargs)
        env = GymAutoResetWrapper(env)

        if cnn_policy:
            env = RGBImgObservationWrapper(env)
        super().__init__(env_name, env, n_envs)

        self.env_params = env_params

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey):
        """Resets the environment."""
        reset_rng = jax.random.split(rng, self.n_envs)
        timestep = jax.vmap(self._env.reset, in_axes=(None, 0))(
            self.env_params, reset_rng
        )
        return timestep, timestep.observation

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: PRNGKey): #noqa: ARG002
        """Steps the environment forward by one step."""
        # here, the env_state is equal to the timestep
        # (as referred to in the xland documentation)
        timestep = jax.vmap(self._env.step, in_axes=(None, 0, 0))(
            self.env_params, env_state, action
        )

        return timestep, (timestep.observation, timestep.reward, timestep.last(), {})

    @property
    def action_space(self):
        """Action space of the environment."""
        return gymnax.environments.spaces.Discrete(
            self._env.num_actions(self.env_params)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, rng: PRNGKey):
        """Samples a random action from the action space."""
        return self.action_space.sample(rng)

    @property
    def observation_space(self):
        """Observation space of the environment."""
        obs_shape = self._env.observation_shape(self.env_params)
        return gymnax.environments.spaces.Box(
            low=jnp.array([0 for _ in obs_shape]),
            high=jnp.array([s - 1 for s in obs_shape]),
            shape=self._env.observation_shape(self.env_params),
        )

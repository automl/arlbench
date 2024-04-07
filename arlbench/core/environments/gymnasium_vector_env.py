from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnasium
import jax
import numpy as np

from .autorl_env import AutoRLEnv

if TYPE_CHECKING:
    from gymnasium.experimental.vector import AsyncVectorEnv


class JaxifyGymOutput(gymnasium.Wrapper):
    def step(self, action):
        s, r, te, tr, _ = self.env.step(action)
        r = np.ones(s.shape) * r
        d = np.ones(s.shape) * int(te or tr)
        return np.stack([s, r, d]).astype(np.float32)


def make_bool(data):
    return np.array([bool(data)])


class GymnasiumVectorEnv(AutoRLEnv):
    def __init__(self, env: AsyncVectorEnv, n_envs: int):
        super().__init__(env, n_envs)

        # TODO implement


    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng):
        # TODO implement
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state, action, rng):
        # TODO implement
        raise NotImplementedError

    @property
    def action_space(self):
        # TODO implement
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, rng):
        # TODO implement
        raise NotImplementedError

    @property
    def observation_space(self):
        # TODO implement
        raise NotImplementedError



# class GymToGymnaxWrapper(gymnax.environments.environment.Environment):
#     def __init__(self, env):
#         super().__init__()
#         self.done = False
#         self.env = JaxifyGymOutput(env)
#         self.state = None
#         self.state_type = None

#     def step_env(
#         self,
#         key: chex.PRNGKey,
#         state: EnvState,
#         action: Union[int, float],
#         params: EnvParams,
#     ):
#         """Environment-specific step transition."""
#         # TODO: this is obviously super wasteful for large state spaces - how can we fix it?
#         result_shape = jax.core.ShapedArray(
#             np.repeat(self.state[None, ...], 3, axis=0).shape, jnp.float32
#         )
#         result = jax.pure_callback(self.env.step, result_shape, action)
#         s = result[0].astype(self.state_type)
#         r = result[1].mean()
#         d = result[2].mean()
#         result_shape = jax.core.ShapedArray((1,), bool)
#         self.done = jax.pure_callback(make_bool, result_shape, d)[0]
#         return s, {}, r, self.done, {}

#     def reset_env(self, key: chex.PRNGKey, params: EnvParams):
#         """Environment-specific reset."""
#         self.done = False
#         self.state, _ = self.env.reset()
#         self.state_type = self.state.dtype
#         return self.state, {}

#     def get_obs(self, state: EnvState) -> chex.Array:
#         """Applies observation function to state."""
#         return state

#     def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
#         """Check whether state transition is terminal."""
#         return self.done

#     @property
#     def num_actions(self) -> int:
#         """Number of actions possible in environment."""
#         if isinstance(self.env.action_space, gymnasium.spaces.Box):
#             return len(self.env.action_space.low)
#         else:
#             return self.env.action_space.n

#     def action_space(self, params: EnvParams):
#         """Action space of the environment."""
#         if isinstance(self.env.action_space, gymnasium.spaces.Box):
#             return gymnax.environments.spaces.Box(
#                 self.env.action_space.low,
#                 self.env.action_space.high,
#                 self.env.action_space.low.shape,
#             )
#         elif isinstance(self.env.action_space, gymnasium.spaces.Discrete):
#             return gymnax.environments.spaces.Discrete(self.env.action_space.n)
#         else:
#             raise NotImplementedError(
#                 "Only Box and Discrete action spaces are supported."
#             )

#     def observation_space(self, params: EnvParams):
#         """Observation space of the environment."""
#         return gymnax.environments.spaces.Box(
#             self.env.observation_space.low,
#             self.env.observation_space.high,
#             self.env.observation_space.low.shape,
#         )

#     def state_space(self, params: EnvParams):
#         """State space of the environment."""
#         return gymnax.environments.spaces.Dict({})

#     @property
#     def default_params(self) -> EnvParams:
#         return EnvParams(500)


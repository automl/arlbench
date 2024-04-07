from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env

from arlbench.utils import gymnasium_space_to_gymnax_space

from .autorl_env import AutoRLEnv

if TYPE_CHECKING:
    import chex
    from chex import PRNGKey


class JaxifyGymOutput(gymnasium.Wrapper):
    def step(self, action):
        s, r, te, tr, _ = self.env.step(action)
        r = np.ones(s.shape) * r
        d = np.ones(s.shape) * int(te or tr)
        return np.stack([s, r, d]).astype(np.float32)


def make_bool(data):
    return np.array([bool(data)])


class GymnasiumEnv(AutoRLEnv):
    def __init__(self, env: Env, n_envs: int, seed: int):
        super().__init__(env, n_envs)

        self.seed = seed

        self.reset_result = jnp.array(self.env.observation_space.sample())

        self.step_result = (
            jnp.array(self.env.observation_space.sample()),
            1.,
            False,
            False,
            {}
        )

    @functools.partial(jax.jit, static_argnums=0)
    def __reset(self, _) -> chex.Array:
        def reset_env():
            return self.env.reset(seed=self.seed)
    
        return jax.pure_callback(
            reset_env, self.reset_result
        )

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey) -> tuple[None, chex.Array]:
        obs = jax.vmap(
            self.__reset, in_axes=(0)
        )(jnp.arange(self.n_envs))
        return None, obs

    @functools.partial(jax.jit, static_argnums=0)
    def __step(self, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, dict]:
        def step():
            obs, reward, term, trunc, info = self.env.step(action)
            return obs, reward, term or trunc, info

        return jax.pure_callback(
            step, self.step_result)

    @functools.partial(jax.jit, static_argnums=0)
    def step(
            self,
            env_state: None,
            action: chex.Array,
            rng: PRNGKey
        ) -> tuple[None, tuple[chex.Array, chex.Array, chex.Array, dict]]:
        step_rng = jax.random.split(rng, self.n_envs)
        obs, reward, done, info = jax.vmap(
            self.__step, in_axes=(0)
        )(action)

        return None, (obs, reward, done, info)

    @property
    def action_space(self):
        return gymnasium_space_to_gymnax_space(self.env.action_space)

    @property
    def observation_space(self):
        return gymnasium_space_to_gymnax_space(self.env.observation_space)



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


from .autorl_wrapper import AutoRLWrapper
from ..environments import AutoRLEnv
from gymnax.environments import spaces
import numpy as np
import jax
import functools


# TODO add test cases
class FlattenObservationWrapper(AutoRLWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: AutoRLEnv):
        super().__init__(env)

    @property
    def observation_space(self) -> spaces.Box:
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
    def __flatten(self, obs):
        # since we have a stack of <n_envs> observations,
        # we want to keep the first dimension
        return obs.reshape(obs.shape[0], -1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, rng
    ):  # TODO improve typing
        env_state, obs = self._env.reset(rng)

        obs = self.__flatten(obs)
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
       self, env_state, action, rng
    ):  # TODO improve typing
        env_state, (obs, reward, done, info) = self._env.step(env_state, action, rng)

        obs = self.__flatten(obs)
        return env_state, (obs, reward, done, info)
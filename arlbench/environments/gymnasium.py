import jax
import gymnax
import numpy as np
import jax.numpy as jnp
from gymnasium.vector import VectorEnv
from arlbench.environments import AutoRLEnv
import functools


class GymnasiumEnv(AutoRLEnv):
    def __init__(self, env: VectorEnv, n_envs: int):
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
    

import jax
import gymnax
import numpy as np
import jax.numpy as jnp
import chex
from typing import Union
from gymnax.environments import EnvState, EnvParams
import gymnasium


class BraxToGymnaxWrapper(gymnax.environments.environment.Environment):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.max_steps_in_episode = 1000
        self.step = jax.jit(self.internal_step)
        self.reset = jax.jit(self.internal_reset)

    def internal_step(self, key, state, action, params):
        state = self.env.step(state, jnp.array([action]))
        return state.obs[0], state, state.reward[0], state.done[0], {}

    def internal_reset(self, key: chex.PRNGKey, params: EnvParams):
        state = self.env.reset(rng=jnp.array([key]))
        return state.obs[0], state

    @property
    def default_params(self):
        return None

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.action_size,)
        )

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.observation_size,)
        )

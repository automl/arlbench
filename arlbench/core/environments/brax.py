import jax
import numpy as np
import jax.numpy as jnp
from .autorl_env import AutoRLEnv
import functools
from brax import BraxEnv
import gymnax
from chex import PRNGKey
from typing import Any


class BraxEnv(AutoRLEnv):
    def __init__(self, env: BraxEnv, n_envs: int):
        super().__init__(env, n_envs)
        self.max_steps_in_episode = 1000

    @functools.partial(jax.jit, static_argnums=0)
    def __reset(self, rng):
        """Internal reset in brax environment."""
        env_state = self.env.reset(rng=jnp.array([rng]))
        return env_state.obs[0], env_state
        
    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey):
        reset_rng = jax.random.split(rng, self.n_envs)
        obs, env_state = jax.vmap(self.__reset, in_axes=(0, None))(
            reset_rng, None
        )
        return env_state, obs
    
    @functools.partial(jax.jit, static_argnums=0)
    def __step(self, env_state, action):
        """Internal step in brax environment."""
        env_state = self.env.step(env_state, jnp.array([action]))
        return env_state.obs[0], env_state, env_state.reward[0], env_state.done[0], {}    

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: PRNGKey):
        step_rng = jax.random.split(rng, self.n_envs)
        obs, env_state, reward, done, info = jax.vmap(
            self.__step, in_axes=(0, 0, 0)
        )(step_rng, env_state, action)

        return env_state, (obs, reward, done, info)

    @property
    def action_space(self):
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.action_size,)
        )

    @property
    def observation_space(self):
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.observation_size,)
        )
    

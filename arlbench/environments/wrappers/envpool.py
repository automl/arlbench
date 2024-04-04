import jax
import functools
import jax.numpy as jnp
from arlbench.environments.environment import Environment


class EnvpoolWrapper(Environment):
    def __init__(self, env, n_envs):
        super().__init__(env, n_envs)
        self.handle0_, _, _, self.xla_step_ = self.env.xla()

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, _):
        obs, _ = self.env.reset()
        return self.handle0_, obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state, action, _):
        handle1, (obs, reward, term, trunc, info) = self.xla_step_(env_state, action)
        done = jnp.logical_or(term, trunc)
        return handle1, (obs, reward, done, info)

    @property
    def action_space(self):
        return self.env.action_space
    
    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, _):
        return self.action_space.sample()

    @property
    def observation_space(self):
        return self.env.observation_space
    

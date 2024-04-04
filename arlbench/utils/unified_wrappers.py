import jax
import gymnax
import functools
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple, Any


class Environment(ABC):
    def __init__(self, env, n_envs):
        self.env = env
        self.n_envs = n_envs

    @abstractmethod
    def reset(self, rng) -> Tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod
    def step(self, env_state, action, rng) -> Tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod 
    def action_space(self):
        raise NotImplementedError
    
    @abstractmethod 
    def sample_action(self, rng):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self):
        raise NotImplementedError


class GymnaxWrapper(Environment):
    def __init__(self, env, n_envs, env_params):
        super().__init__(env, n_envs)
        if not isinstance(env, gymnax.environments.environment.Environment):
            raise ValueError(f"Invalid environment class. Expected gymnax.environments.environment.Environment (or subclass) but got '{env.__class__.__name__}'.")

        self.n_envs = n_envs
        self.env_params = env_params

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng):
        reset_rng = jax.random.split(rng, self.n_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_rng, self.env_params
        )
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state, action, rng):
        step_rng = jax.random.split(rng, self.n_envs)
        obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_rng, env_state, action, self.env_params)

        return env_state, (obs, reward, done, info)

    @property
    def action_space(self):
        return self.env.action_space(self.env_params)
    
    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, rng):
        return self.action_space.sample(rng)

    @property
    def observation_space(self):
        return self.env.observation_space(self.env_params)
    

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
    
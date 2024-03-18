from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, Sequence
from flax.training.train_state import TrainState
import functools
import jax
import numpy as np
import gymnax
import jax.numpy as jnp


class Agent(ABC):
    def __init__(self, config, env, env_params) -> None:
        super().__init__()

        self.config = config
        self.env = env
        self.env_params = env_params

    @property
    def action_type(self) -> Tuple[Sequence[int], bool]:
        if isinstance(
            self.env.action_space(self.env_params), gymnax.environments.spaces.Discrete
        ):
            action_size = self.env.action_space(self.env_params).n
            discrete = True
        elif isinstance(self.env.action_space(self.env_params), gymnax.environments.spaces.Box):
            action_size = self.env.action_space(self.env_params).shape[0]
            discrete = False
        else:
            raise NotImplementedError(
                f"Only Discrete and Box action spaces are supported, got {self.env.action_space(self.env_params)}."
            )

        return action_size, discrete
    
    @abstractmethod
    def init(self, rng) -> Tuple[Any, Any, Tuple]:
        pass

    @abstractmethod
    def train(self) -> Tuple[TrainState, Optional[Tuple]]:
        pass

    @abstractmethod 
    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, network_params, obsv, rng = None) -> Any:
        pass

    def _env_episode(self, rng, network_params):
        reset_rng = jax.random.split(rng, 1)
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)
        initial_state = (obsv, env_state, 0.0, False, rng, network_params)

        def cond_fn(state):
            _, _, _, done, _, _ = state
            return jnp.logical_not(done)

        def body_fn(state):
            obsv, env_state, r, done, rng, network_params = state

            # SELECT ACTION
            rng, action_rng = jax.random.split(rng)
            action = self.predict(network_params, obsv, action_rng)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, 1)
            obsv, env_state, reward, done_, _ = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, self.env_params)

            r += jnp.sum(reward)
            done = jnp.any(done_) 

            return obsv, env_state, r, done, rng, network_params

        final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
        _, _, total_reward, _, _, _ = final_state
        return total_reward

    @functools.partial(jax.jit, static_argnums=0)
    def _env_episode_deprecated(self, rng, network_params, _):
        reset_rng = jax.random.split(rng, 1)
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)
        r = 0
        done = False
        while not done:
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            action = self.predict(network_params, obsv, rng)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, 1)
            obsv, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, self.env_params)
            r += reward
        return r

    def eval(self, runner_state, num_eval_episodes) -> float:
        eval_rng = jax.random.split(runner_state.rng, num_eval_episodes)
        rewards = jax.vmap(self._env_episode, in_axes=(0, None))(
            eval_rng, runner_state.train_state.params
        )
        return float(jnp.mean(rewards))


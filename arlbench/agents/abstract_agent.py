from abc import ABC, abstractmethod
from flax.training.train_state import TrainState
from typing import Tuple, Optional, Any
import functools
import jax
import numpy as np


class Agent(ABC):
    def __init__(self, env, env_params) -> None:
        super().__init__()

        self.env = env
        self.env_params = env_params

    @abstractmethod
    def train(self) -> Tuple[TrainState, Optional[Tuple]]:
        pass

    @abstractmethod 
    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, network_params, obsv, rng = None) -> Any:
        pass

    def _env_episode(self, rng, network_params, _):
        reset_rng = jax.random.split(rng, 1)
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)
        r = 0
        done = False
        while not done:
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            action = self.predict(network_params, obsv, rng=rng)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, 1)
            obsv, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, self.env_params)
            r += reward
        return r

    def eval(self, rng, network_params, num_eval_episodes) -> float:
        rewards = jax.vmap(self._env_episode, in_axes=(None, None, 0))(
            rng, network_params, np.arange(num_eval_episodes)
        )
        return float(np.mean(rewards))


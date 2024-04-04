from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, Sequence, Union, Dict
import functools
import jax
import gymnax
import jax.numpy as jnp
from ConfigSpace import Configuration, ConfigurationSpace
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState
import gymnasium
import gym
import numpy as np
from arlbench.utils import Environment


class Algorithm(ABC):
    def __init__(
            self,
            hpo_config: Union[Configuration, Dict], 
            nas_config: Union[Configuration, Dict], 
            env_options: Dict, 
            env: Environment, 
            track_metrics=False,
            track_trajectories=False
        ) -> None:
        super().__init__()

        self.hpo_config = hpo_config
        self.nas_config = nas_config
        self.env_options = env_options
        self.env = env
        self.track_metrics = track_metrics
        self.track_trajectories = track_trajectories

    @property
    def action_type(self) -> Tuple[Sequence[int], bool]:
        if callable(self.env.action_space):
            action_space = self.env.action_space(self.env_params)
        else:
            action_space = self.env.action_space

        if isinstance(
            action_space, gymnax.environments.spaces.Discrete
        ) or isinstance(
            action_space, gym.spaces.Discrete
        ) or isinstance(
            action_space, gymnasium.spaces.Discrete
        ):
            action_size = action_space.n
            discrete = True
        elif isinstance(
            action_space, gymnax.environments.spaces.Box
        ) or isinstance(
            action_space, gym.spaces.Box
        ) or isinstance(
            action_space, gymnasium.spaces.Box
        ):
            action_size = action_space.shape[0]
            discrete = False
        else:
            raise NotImplementedError(
                f"Only Discrete and Box action spaces are supported, got {self.env.action_space}."
            )

        return action_size, discrete
    
    @staticmethod
    @abstractmethod
    def get_hpo_config_space(seed=None) -> ConfigurationSpace:
        pass

    @staticmethod
    @abstractmethod
    def get_default_hpo_config() -> Configuration:
        pass

    @staticmethod
    @abstractmethod
    def get_nas_config_space(seed=None) -> ConfigurationSpace:
        pass

    @staticmethod
    @abstractmethod
    def get_default_nas_config() -> Configuration:
        pass
    
    @abstractmethod
    def init(self, rng) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def train(self, runner_state: Any, buffer_state: Any) -> Tuple[tuple[Any, PrioritisedTrajectoryBufferState], Optional[Tuple]]:
        pass

    @abstractmethod 
    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, network_params, obsv, rng = None) -> Any:
        pass

    @functools.partial(jax.jit, static_argnums=0)
    def _env_episode(self, carry, _):
        rng, network_params = carry

        env_state, obs = self.env.reset(rng) 
        initial_state = (
            env_state,
            obs,
            jnp.full((self.env.n_envs,), 0.), 
            jnp.full((self.env.n_envs,), False),
            rng,
            network_params
        )

        def cond_fn(state):
            _, _, _, done, _, _ = state
            return jnp.logical_not(jnp.all(done))

        def body_fn(state):
            env_state, obs, reward, done, rng, network_params = state

            # SELECT ACTION
            rng, action_rng = jax.random.split(rng)
            action = self.predict(network_params, obs, action_rng)

            # STEP ENV
            rng, step_rng = jax.random.split(rng)
            env_state, (obs, reward_, done_, _) = self.env.step(env_state, action, step_rng)

            # Count rewards only for envs that are not already done
            reward += reward_ * ~done
            
            done = jnp.logical_or(done, done_)

            return env_state, obs, reward, done, rng, network_params

        final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
        _, _, reward, _, _, _ = final_state

        return (rng, network_params), reward  

    def eval(self, runner_state, num_eval_episodes):
        # Number of parallel evaluations, each with n_envs environments
        if num_eval_episodes > self.env.n_envs:
            n_evals = int(jnp.ceil(num_eval_episodes / self.env.n_envs))
        else:
            n_evals = 1

        rewards = []

        carry = runner_state.rng, runner_state.train_state.params

        (_, _), rewards = jax.lax.scan(
            self._env_episode, carry, None, n_evals
        )

        return jnp.concat(rewards)[:num_eval_episodes]
    
    # unify the evaluation methods
    def sac_eval(self, runner_state, num_eval_episodes) -> float:
        eval_rng = jax.random.split(runner_state.rng, num_eval_episodes)
        rewards = jax.vmap(self._env_episode, in_axes=(0, None))(
            eval_rng, runner_state.actor_train_state.params
        )
        return float(jnp.mean(rewards))
    
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, Sequence, Union, Dict
import functools
import jax
import gymnax
import jax.numpy as jnp
from ConfigSpace import Configuration, ConfigurationSpace
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState



class Algorithm(ABC):
    def __init__(
            self,
            hpo_config: Union[Configuration, Dict], 
            nas_config: Union[Configuration, Dict], 
            env_options: Dict, 
            env: Any, 
            env_params: Any,
            track_metrics=False,
            track_trajectories=False
        ) -> None:
        super().__init__()

        self.hpo_config = hpo_config
        self.nas_config = nas_config
        self.env_options = env_options
        self.env = env
        self.env_params = env_params
        self.track_metrics = track_metrics
        self.track_trajectories = track_trajectories

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
    
    def eval(self, runner_state, num_eval_episodes) -> jnp.ndarray:
        eval_rng = jax.random.split(runner_state.rng, num_eval_episodes)
        rewards = jax.vmap(self._env_episode, in_axes=(0, None))(
            eval_rng, runner_state.train_state.params
        )
        return jnp.array(rewards)

    # unify the evaluation methods
    def sac_eval(self, runner_state, num_eval_episodes) -> float:
        eval_rng = jax.random.split(runner_state.rng, num_eval_episodes)
        rewards = jax.vmap(self._env_episode, in_axes=(0, None))(
            eval_rng, runner_state.actor_train_state.params
        )
        return float(jnp.mean(rewards))


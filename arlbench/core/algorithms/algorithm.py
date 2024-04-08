from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import gym
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from flashbax.buffers.prioritised_trajectory_buffer import \
        PrioritisedTrajectoryBufferState

    from arlbench.core.environments import AutoRLEnv
    from arlbench.core.wrappers import AutoRLWrapper


class Algorithm(ABC):
    def __init__(
            self,
            hpo_config: Configuration | dict,
            nas_config: Configuration | dict,
            env_options: dict,
            env: AutoRLEnv | AutoRLWrapper,
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
    def action_type(self) -> tuple[Sequence[int], bool]:
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Discrete | gymnasium.spaces.Discrete | gymnax.environments.spaces.Discrete):
            action_size = action_space.n
            discrete = True
        elif isinstance(action_space, gym.spaces.Box | gymnasium.spaces.Box | gymnax.environments.spaces.Box):
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
    def train(self, runner_state: Any, buffer_state: Any) -> tuple[tuple[Any, PrioritisedTrajectoryBufferState], tuple | None]:
        pass

    @abstractmethod
    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, runner_state, obsv, rng = None) -> Any:
        pass

    @functools.partial(jax.jit, static_argnums=0)
    def _env_episode(self, state, _):
        rng, runner_state = state
        rng, reset_rng = jax.random.split(rng)

        env_state, obs = self.env.reset(reset_rng)
        initial_state = (
            env_state,
            obs,
            jnp.full((self.env.n_envs,), 0.),
            jnp.full((self.env.n_envs,), False),
            rng,
            runner_state
        )

        def cond_fn(state):
            _, _, _, done, _, _ = state
            return jnp.logical_not(jnp.all(done))

        def body_fn(state):
            env_state, obs, reward, done, rng, runner_state = state

            # SELECT ACTION
            rng, action_rng = jax.random.split(rng)
            action = self.predict(runner_state, obs, action_rng)

            # STEP ENV
            rng, step_rng = jax.random.split(rng)
            env_state, (obs, reward_, done_, _) = self.env.step(env_state, action, step_rng)

            # Count rewards only for envs that are not already done
            reward += reward_ * ~done

            done = jnp.logical_or(done, done_)

            return env_state, obs, reward, done, rng, runner_state

        final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
        _, _, reward, _, rng, _ = final_state

        return (rng, runner_state), reward

    def eval(self, runner_state, num_eval_episodes):
        # Number of parallel evaluations, each with n_envs environments
        n_evals = int(np.ceil(num_eval_episodes / self.env.n_envs))
        _, rewards = jax.lax.scan(
            self._env_episode, (runner_state.rng, runner_state), None, n_evals
        )
        return jnp.concat(rewards)[:num_eval_episodes]

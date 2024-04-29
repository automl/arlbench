from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gym
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import chex
    from ConfigSpace import Configuration, ConfigurationSpace

    from arlbench.core.environments import Environment
    from arlbench.core.wrappers import AutoRLWrapper


class Algorithm(ABC):
    """Abstract base class for a reinforcement learning algorithm. Contains basic functionality that is shared among different algorithm implementations."""
    name: str

    def __init__(
            self,
            hpo_config: Configuration,
            nas_config: Configuration,
            env: Environment | AutoRLWrapper,
            eval_env: Environment | AutoRLWrapper | None = None,
            track_trajectories: bool = False,
            track_metrics: bool = False
        ) -> None:
        """Algorithm super-class constructor, is only called by sub-classes.

        Args:
            hpo_config (Configuration): Hyperparameter configuration.
            nas_config (Configuration): Neural architecture configuration.
            env (Environment | AutoRLWrapper): Training environment.
            eval_env (Environment | AutoRLWrapper | None, optional): Evaluation environent (otherwise training environment is used for evaluation). Defaults to None.
            track_trajectories (bool, optional):  Track metrics such as loss and gradients during training. Defaults to False.
            track_metrics (bool, optional): Track trajectories during training. Defaults to False.
        """
        super().__init__()

        self.hpo_config = hpo_config
        self.nas_config = nas_config
        self.env = env
        self.eval_env = env if eval_env is None else eval_env
        self.track_metrics = track_metrics
        self.track_trajectories = track_trajectories

    @property
    def action_type(self) -> tuple[int, bool]:
        """The size and type of actions of the algorithm/environment.

        Returns:
            tuple[int, bool]: Tuple of (action_size, discrete). action_size is the number of possible actions and discrete defines whether the action space is discrete or not.
        """
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
    def get_hpo_config_space(seed: int | None = None) -> ConfigurationSpace:
        """Returns the hyperparameter configuration space of the algorithm.

        Args:
            seed (int | None, optional): Random generator seed that is used to sample configurations. Defaults to None.

        Returns:
            ConfigurationSpace: Hyperparameter configuration space of the algorithm.
        """

    @staticmethod
    @abstractmethod
    def get_default_hpo_config() -> Configuration:
        """Returns the default hyperparameter configuration of the agent.

        Returns:
            Configuration: Default hyperparameter configuration.
        """

    @staticmethod
    @abstractmethod
    def get_nas_config_space(seed: int | None = None) -> ConfigurationSpace:
        """Returns the neural architecture configuration space of the algorithm.

        Args:
            seed (int | None, optional): Random generator seed that is used to sample configurations. Defaults to None.

        Returns:
            ConfigurationSpace: Neural architecture configuration space of the algorithm.
        """

    @staticmethod
    @abstractmethod
    def get_default_nas_config() -> Configuration:
        """Returns the default neural architecture configuration of the agent.

        Returns:
            Configuration: Default neural architecture configuration.
        """

    @staticmethod
    @abstractmethod
    def get_checkpoint_factory(
        runner_state: Any,
        train_result: Any,
    ) -> dict[str, Callable]:
        """Creates a factory dictionary of all posssible checkpointing options for Algorithm.

        Args:
            runner_state (Any): Algorithm runner state.
            train_result (Any): Training result.

        Returns:
            dict[str, Callable]: Dictionary of factory functions.
        """

    @abstractmethod
    def init(self, rng: chex.PRNGKey) -> Any:
        """Initializes the algorithm state. Passed parameters are not initialized and included in the final state.

        Args:
            rng (chex.PRNGKey): Random generator key.

        Returns:
            Any: Algorithm state.
        """

    @abstractmethod
    def train(
        self,
        runner_state: Any,
        buffer_state: Any,
        n_total_timesteps: int = 1000000,
        n_eval_steps: int = 100,
        n_eval_episodes: int = 10,
    ) -> tuple[Any, Any]:
        """Performs one iteration of training.

        Args:
            runner_state (Any): Algorithm runner state.
            buffer_state (Any): Algorithm buffer state.
            n_total_timesteps (int, optional): Total number of training timesteps. Update steps = n_total_timesteps // n_envs. Defaults to 1000000.
            n_eval_steps (int, optional): Number of evaluation steps during training.
            n_eval_episodes (int, optional): Number of evaluation episodes per evaluation during training.

        Returns:
            tuple[Any, Any]: (algorithm_state, training_result).
        """

    @abstractmethod
    @functools.partial(jax.jit, static_argnums=0)
    def predict(
        self,
        runner_state: Any,
        obs: jnp.ndarray,
        rng: chex.PRNGKey | None = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Predict action(s) based on the current observation(s).

        Args:
            runner_state (Any): Algorithm runner state.
            obs (jnp.ndarray): Observation(s).
            rng (chex.PRNGKey | None, optional): Random generator key. Defaults to None.
            deterministic (bool): Return deterministic action. Defaults to True.

        Returns:
            jnp.ndarray: Action(s).
        """

    @functools.partial(jax.jit, static_argnums=0)
    def _env_episode(self, state: tuple[chex.PRNGKey, Any], _: None) -> tuple[tuple[chex.PRNGKey, Any], jnp.ndarray]:
        """Evaluate one episode of evaluation in parallel on n_envs.

        Args:
            state (tuple[chex.PRNGKey, Any]): (rng, runner_state). Current state of the evaluation.
            _ (None): Unused parameter (required for jax.lax.scan).

        Returns:
            tuple[tuple[chex.PRNGKey, Any], jnp.ndarray]: ((rng, runner_state), reward). Current state of the evaluation and cumulative rewards.
        """
        #jax.debug.print("\n\n\n\n\n\nStart Eval Episode")
        rng, runner_state = state
        rng, reset_rng = jax.random.split(rng)

        env_state, obs = self.eval_env.reset(reset_rng)
        #jax.debug.print("obs: {obs}", obs=obs.mean(axis=(-1,-2,-3)))
        initial_state = (
            env_state,
            obs,
            jnp.full((self.eval_env.n_envs,), 0.),
            jnp.full((self.eval_env.n_envs,), False),
            rng,
            runner_state
        )

        def cond_fn(state: tuple) -> jnp.bool:
            """Condition function for JAX while loop. Returns true if not all parallel environments are done.

            Args:
                state (tuple): Current loop state.

            Returns:
                jnp.bool: True if not all environments are done.
            """
            _, _, _, done, _, _ = state
            #jax.debug.print("{done}", done=done)
            #jax.debug.print("{not_done}", not_done=jnp.logical_not(jnp.all(done)))

            return jnp.logical_not(jnp.all(done))

        def body_fn(state: tuple) -> tuple:
            """Body function for JAX while loop. Performs one parallel step in all environments.

            Args:
                state (tuple): Current loop state.

            Returns:
                tuple: Updated loop state.
            """
            env_state, obs, reward, done, rng, runner_state = state

            # Select action
            rng, action_rng = jax.random.split(rng)
            action = self.predict(runner_state, obs, action_rng, deterministic=True)
            #jax.debug.print("obs: {obs}", obs=obs.mean(axis=(-1,-2,-3)))
            #jax.debug.print("action: {action}", action=action)

            # Step
            rng, step_rng = jax.random.split(rng)
            env_state, (obs, reward_, done_, info_) = self.eval_env.step(env_state, action, step_rng)
            #jax.debug.print("{done_}", done_=done_)
            #jax.debug.print("done: {done}", done=done)
            #jax.debug.print("{info}", info=info_["elapsed_step"])
            #jax.debug.print("{info}", info=info_["terminated"])
            #jax.debug.print("lives: {info}", info=info_["lives"])
            #jax.debug.print("info: {info}", info=info_)

            # Count rewards only for envs that are not already done
            reward += reward_ * ~done
            #jax.debug.print("reward: {reward}", reward=reward)
            done = jnp.logical_or(done, done_)

            return env_state, obs, reward, done, rng, runner_state

        final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
        _, _, reward, _, rng, _ = final_state

        return (rng, runner_state), reward

    def eval(self, runner_state: Any, num_eval_episodes: int) -> jnp.ndarray:
        """Evaluate the algorithm.

        Args:
            runner_state (Any): Algorithm runner state.
            num_eval_episodes (int): Number of evaluation episodes.

        Returns:
            jnp.ndarray: Cumulative reward per evaluation episodes. Shape: (n_eval_episodes,).
        """
        # Number of parallel evaluations, each with n_envs environments
        n_evals = int(np.ceil(num_eval_episodes / self.eval_env.n_envs))

        _, rewards = jax.lax.scan(
            self._env_episode, (runner_state.rng, runner_state), None, n_evals
        )
        return jnp.concat(rewards)[:num_eval_episodes]

    def update_hpo_config(self, hpo_config: Configuration):
        """Update the hyperparameter configuration of the algorithm.

        Args:
            hpo_config (Configuration): Hyperparameter configuration.
        """
        self.hpo_config = hpo_config


# Heavily based on the original PQN code here: https://github.com/mttga/purejaxql
"""DQN algorithm."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)
from arlbench.core import running_statistics
from arlbench.core.algorithms.algorithm import Algorithm
from arlbench.core.algorithms.dqn.dqn import (
    DQNMetrics,
    DQNRunnerState,
    DQNState,
    DQNTrainState,
    DQNTrainReturnT,
    DQNTrainingResult,
    Transition
)

from .models import CNNQ, MLPQ

if TYPE_CHECKING:
    import chex
    from flashbax.buffers.prioritised_trajectory_buffer import (
        PrioritisedTrajectoryBufferState,
    )
    from flax.core.frozen_dict import FrozenDict

    from arlbench.core.environments import Environment
    from arlbench.core.running_statistics import RunningStatisticsState
    from arlbench.core.wrappers import Wrapper


class PQN(Algorithm):
    """JAX-based implementation of PQN."""

    name: str = "pqn"

    def __init__(
        self,
        hpo_config: Configuration,
        env: Environment | Wrapper,
        eval_env: Environment | Wrapper | None = None,
        deterministic_eval: bool = True,
        eval_eps: float = 0.05,
        cnn_policy: bool = False,
        nas_config: Configuration | None = None,
        track_trajectories: bool = False,
        track_metrics: bool = False,
    ) -> None:
        """Creates a DQN algorithm instance.

        Args:
            hpo_config (Configuration): Hyperparameter configuration.
            env (Environment | AutoRLWrapper): Training environment.
            eval_env (Environment | AutoRLWrapper | None, optional): Evaluation environment (otherwise training environment is used for evaluation). Defaults to None.
            deterministic_eval (bool, optional): Use deterministic evaluation. Defaults to True.
            eval_eps (float, optional): Epsilon value for non-deterministic evaluation. Defaults to 0.05.
            cnn_policy (bool, optional): Use CNN network architecture. Defaults to False.
            nas_config (Configuration | None, optional): Neural architecture configuration. Defaults to None.
            track_trajectories (bool, optional):  Track metrics such as loss and gradients during training. Defaults to False.
            track_metrics (bool, optional): Track trajectories during training. Defaults to False.
        """
        if nas_config is None:
            nas_config = PQN.get_default_nas_config()

        super().__init__(
            hpo_config,
            nas_config,
            env,
            eval_env=eval_env,
            deterministic_eval=deterministic_eval,
            track_trajectories=track_trajectories,
            track_metrics=track_metrics,
        )

        self.eval_eps = eval_eps

        # For the network, we need the properties of the action space
        action_size, discrete = self.action_type
        network_cls = CNNQ if cnn_policy else MLPQ
        self.network = network_cls(
            action_size,
            discrete=discrete,
            activation=self.nas_config["activation"],
            hidden_size=self.nas_config["hidden_size"],
            normalization=True
        )

    @staticmethod
    def get_hpo_config_space(seed: int | None = None) -> ConfigurationSpace:
        """Returns the hyperparameter optimization (HPO) configuration space for PQN."""
        cs = ConfigurationSpace(
            name="PQNConfigSpace",
            seed=seed,
            space={
                "learning_rate": Float(
                    "learning_rate", (1e-6, 0.1), default=3e-4, log=True
                ),
                "gamma": Float("gamma", (0.8, 1.0), default=0.99),
                "initial_epsilon": Float("initial_epsilon", (0.5, 1.0), default=1.0),
                "target_epsilon": Float("target_epsilon", (0.001, 0.2), default=0.05),
                "exploration_fraction": Float("exploration_fraction", (0.005, 0.5), default=0.1),
                "minibatch_size": Categorical(
                    "minibatch_size", [16, 32, 64, 128, 2048], default=64
                ),
                "n_steps": Categorical(
                    "n_steps", [5, 32, 64, 80, 128, 256, 512], default=128
                ),
                "lambda": Float("gae_lambda", (0.8, 0.9999), default=0.95),
                "normalize_observations": Categorical(
                    "normalize_observations", [True, False], default=False
                ),
            },
        )

        return cs

    @staticmethod
    def get_default_hpo_config() -> Configuration:
        """Returns the default hyperparameter configuration for DQN."""
        return PQN.get_hpo_config_space().get_default_configuration()

    @staticmethod
    def get_nas_config_space(seed=None) -> ConfigurationSpace:
        """Returns the neural architecture search (NAS) configuration space for DQN."""
        return ConfigurationSpace(
            name="PQNNASConfigSpace",
            seed=seed,
            space={
                "activation": Categorical(
                    "activation", ["tanh", "relu"], default="tanh"
                ),
                "hidden_size": Integer("hidden_size", (1, 1024), default=64),
            },
        )

    @staticmethod
    def get_default_nas_config() -> Configuration:
        """Returns the default NAS configuration for DQN."""
        return PQN.get_nas_config_space().get_default_configuration()

    @staticmethod
    def get_checkpoint_factory(
        runner_state: DQNRunnerState,
        train_result: DQNTrainingResult | None,
    ) -> dict[str, Callable]:
        """Creates a factory dictionary of all possible checkpointing options for DQN.

        Args:
            runner_state (DQNRunnerState): Algorithm runner state.
            train_result (DQNTrainingResult | None): Training result.

        Returns:
            dict[str, Callable]: Dictionary of factory functions containing [opt_state, params, target_params, loss, trajectories].
        """
        train_state = runner_state.train_state

        def get_trajectories():
            if train_result is None or train_result.trajectories is None:
                return None

            traj = train_result.trajectories

            trajectories = {}
            trajectories["states"] = jnp.concatenate(traj.obs, axis=0)
            trajectories["action"] = jnp.concatenate(traj.action, axis=0)
            trajectories["reward"] = jnp.concatenate(traj.reward, axis=0)
            trajectories["dones"] = jnp.concatenate(traj.done, axis=0)

            return trajectories

        return {
            "opt_state": lambda: train_state.opt_state,
            "params": lambda: train_state.params,
            "loss": lambda: train_result.metrics.loss
            if train_result and train_result.metrics
            else None,
            "trajectories": get_trajectories,
        }

    def init(
        self,
        rng: chex.PRNGKey,
        network_params: FrozenDict | dict | None = None,
        opt_state: optax.OptState | None = None,
    ) -> DQNState:
        """Initializes DQN state. Passed parameters are not initialized and included in the final state.

        Args:
            rng (chex.PRNGKey): Random generator key.
            buffer_state (PrioritisedTrajectoryBufferState | None, optional): Buffer state. Defaults to None.
            network_params (FrozenDict | dict | None, optional): Networks parameters. Defaults to None.
            target_params (FrozenDict | dict | None, optional): Target network parameters. Defaults to None.
            opt_state (optax.OptState | None, optional): Optimizer state. Defaults to None.

        Returns:
            DQNState: DQN state.
        """
        rng, reset_rng = jax.random.split(rng)
        env_state, obs = self.env.reset(reset_rng)

        # If any of these if not defined, we need a dummy environment transition
        # to initialize them
        if network_params is None:
            dummy_rng = jax.random.PRNGKey(0)
            _action = self.env.sample_actions(dummy_rng)
            _, (_obs, _, _, _) = self.env.step(env_state, _action, dummy_rng)

        rng, init_rng = jax.random.split(rng)
        if network_params is None:
            network_params = self.network.init(init_rng, _obs)

        train_state_kwargs = {
            "apply_fn": self.network.apply,
            "params": network_params,
            "target_params": None,
            "tx": optax.adam(self.hpo_config["learning_rate"]),
            "opt_state": opt_state,
        }
        train_state = DQNTrainState.create_with_opt_state(**train_state_kwargs)

        global_step = 0

        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            normalizer_state=running_statistics.init_state(obs[0]),
            env_state=env_state,
            obs=obs,
            global_step=global_step,
        )

        self.rollout_size = int(self.hpo_config["n_steps"] * self.env.n_envs)
        # Ensure that at least one minibatch is available after each rollout
        if self.hpo_config["minibatch_size"] > self.rollout_size:
            self.minibatch_size = self.rollout_size
        else:
            self.minibatch_size = int(self.hpo_config["minibatch_size"])
        self.n_minibatches = int(self.rollout_size // self.minibatch_size)

        return DQNState(runner_state=runner_state, buffer_state=None)

    @functools.partial(jax.jit, static_argnums=0)
    def predict(
        self,
        runner_state: DQNRunnerState,
        obs: jnp.ndarray,
        rng: chex.PRNGKey,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Predict action(s) based on the current observation(s).

        Args:
            runner_state (DQNRunnerState): Algorithm runner state.
            obs (jnp.ndarray): Observation(s).
            rng (chex.PRNGKey | None, optional): Not used in DQN. Random generator key in other algorithms. Defaults to None.
            deterministic (bool): Return deterministic action. Defaults to True.

        Returns:
            jnp.ndarray: Action(s).
        """

        def random_action(rng: chex.PRNGKey, _) -> jnp.ndarray:
            _rngs = jax.random.split(rng, obs.shape[0])
            return jnp.array(
                [self.env.action_space.sample(_rngs[i]) for i in range(obs.shape[0])]
            )

        def greedy_action(_: chex.PRNGKey, obs: jnp.ndarray) -> jnp.ndarray:
            if self.hpo_config["normalize_observations"]:
                obs = running_statistics.normalize(obs, runner_state.normalizer_state)
            q_values = self.network.apply(runner_state.train_state.params, obs)
            return q_values.argmax(axis=-1)

        def sample_action(rng: chex.PRNGKey, obs: jnp.ndarray) -> jnp.ndarray:
            rnd_action = random_action(rng, obs)
            grd_action = greedy_action(rng, obs)
            return jax.lax.select(
                jax.random.uniform(rng, obs.shape[:1]) < self.eval_eps, rnd_action, grd_action
            )

        return jax.lax.cond(
            deterministic,
            greedy_action,
            sample_action,
            rng,
            obs,
        )

    @functools.partial(jax.jit, static_argnums=(0, 3, 4, 5), donate_argnums=(2,))
    def train(
        self,
        runner_state: DQNRunnerState,
        buffer_state: PrioritisedTrajectoryBufferState,
        n_total_timesteps: int = 1000000,
        n_eval_steps: int = 100,
        n_eval_episodes: int = 10,
    ) -> DQNTrainReturnT:
        """Performs one full training.

        Args:
            runner_state (DQNRunnerState): DQN runner state.
            buffer_state (PrioritisedTrajectoryBufferState): Buffer state.
            n_total_timesteps (int, optional): Total number of training timesteps. Update steps = n_total_timesteps // n_envs. Defaults to 1000000.
            n_eval_steps (int, optional): Number of evaluation steps during training. Defaults to 100.
            n_eval_episodes (int, optional): Number of evaluation episodes per evaluation during training. Defaults to 10.

        Returns:
            DQNTrainReturnT: Tuple of DQN algorithm state and training result.
        """
        n_update_steps = int(
            np.ceil(
                n_total_timesteps
                / self.env.n_envs
                / self.hpo_config["n_steps"]
                / n_eval_steps
            )
        )

        def train_eval_step(
            carry: tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], _: None
        ) -> tuple[
            tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], DQNTrainingResult
        ]:
            """Performs one iteration of training and evaluation.

            Args:
                carry (tuple[DQNRunnerState, PrioritisedTrajectoryBufferState]): DQN runner state and buffer state.
                _ (None): Unused parameter (required for jax.lax.scan).

            Returns:
                tuple[tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], DQNTrainingResult]: Tuple of (DQN runner state, buffer state) and training result.
            """
            runner_state = carry
            runner_state, (metrics, trajectories) = jax.lax.scan(
                self._update_step,
                runner_state,
                jnp.array([n_total_timesteps] * n_update_steps),
                n_update_steps,
            )
            eval_returns = self.eval(runner_state, n_eval_episodes)

            return runner_state, DQNTrainingResult(
                eval_rewards=eval_returns, trajectories=trajectories, metrics=metrics
            )

        runner_state, result = jax.lax.scan(
            train_eval_step,
            runner_state,
            None,
            n_eval_steps,
        )
        return DQNState(runner_state=runner_state, buffer_state=None), result    

    def _update_step(
        self,
        carry: tuple[DQNRunnerState, PrioritisedTrajectoryBufferState],
        n_total_timesteps: int,
    ) -> tuple[
        tuple[DQNRunnerState, PrioritisedTrajectoryBufferState],
        tuple[DQNMetrics | None, Transition | None]
    ]:
        """Performs one iteration of updating including environment steps.

        Args:
            carry (tuple[DQNRunnerState, PrioritisedTrajectoryBufferState]): _description_
            n_total_timesteps (int): Number of environment steps to take.

        Returns:
            tuple[tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], tuple[DQNMetrics | None, Transition | None]]: Updated runner state, recorded transitionsa and metrics if specified.
        """
        runner_state = carry
        (rng, train_state, normalizer_state, env_state, last_obs, global_step) = (
            runner_state
        )

        def collect_rollouts(
            carry: tuple[
                chex.PRNGKey,
                DQNTrainState,
                RunningStatisticsState,
                jnp.ndarray,
                Any,
                int,
            ],
            _: None,
        ) -> tuple[
            tuple[
                chex.PRNGKey,
                DQNTrainState,
                RunningStatisticsState,
                jnp.ndarray,
                Any,
                int,
            ],
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict],
        ]:
            """Takes one environment step (n_envs many steps).

            Args:
                carry (tuple[chex.PRNGKey, DQNTrainState, RunningStatisticsState, jnp.ndarray, Any, int, PrioritisedTrajectoryBufferState]): Carry for jax.lax.scan().
                _ (None): Unused parameter.

            Returns:
                tuple[ tuple[ chex.PRNGKey, DQNTrainState, RunningStatisticsState, jnp.ndarray, Any, int, PrioritisedTrajectoryBufferState, ], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], ]: _description_
            """
            rng, train_state, normalizer_state, obs, env_state, global_step = carry
            rng, rng_action, rng_sampling = jax.random.split(rng, 3)
            if self.hpo_config["normalize_observations"]:
                obs = running_statistics.normalize(obs, runner_state.normalizer_state)
            q_vals = self.network.apply(train_state.params, obs)

            _rngs = jax.random.split(rng_action, self.env.n_envs)
            training_fraction = jnp.min(
                jnp.array([global_step * self.env.n_envs / n_total_timesteps, self.hpo_config["exploration_fraction"]])
            )
            epsilon = self.hpo_config["initial_epsilon"] - training_fraction * (
                (self.hpo_config["initial_epsilon"] - self.hpo_config["target_epsilon"])
                / self.hpo_config["exploration_fraction"]
            )
            
            rand_action = jnp.array(
                    [self.env.action_space.sample(_rngs[i]) for i in range(obs.shape[0])]
                )
            greedy_action = q_vals.argmax(axis=-1)
            action = jax.lax.select(
                jax.random.uniform(rng_sampling, shape=last_obs.shape[:1]) < epsilon,
                rand_action,
                greedy_action,
            )

            env_state, (obsv, reward, done, info) = self.env.step(rng=rng, env_state=env_state, action=action)
            global_step += 1
            return (
                rng,
                train_state,
                normalizer_state,
                obsv,
                env_state,
                global_step,
            ), (
                obsv,
                action,
                reward,
                done,
                info,
                q_vals
            )

        rng, _ = jax.random.split(rng)
        (rng, train_state, normalizer_state, last_obs, env_state, global_step), (obs, action, reward, done, info, q_vals) = jax.lax.scan(
                collect_rollouts,
                (rng, train_state, normalizer_state, last_obs, env_state, global_step),
                None,
                self.hpo_config["n_steps"],
            )
        
        last_q = self.network.apply(train_state.params, last_obs)
        last_q = jnp.max(last_q, axis=-1)

        def _get_target(lambda_returns_and_next_q, r_d_q):
                lambda_returns, next_q = lambda_returns_and_next_q
                reward, done, q_val = r_d_q
                target_bootstrap = (
                    reward + self.hpo_config["gamma"] * (1 - done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + self.hpo_config["gamma"] * self.hpo_config["lambda"] * delta
                )
                lambda_returns = (
                    1 - done
                ) * lambda_returns + done * reward
                next_q = jnp.max(q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

        last_q = last_q * (1 - done[-1])
        lambda_returns = reward[-1] + self.hpo_config["gamma"] * last_q
        r_d_q = [reward, done, q_vals]
        _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], r_d_q),
                reverse=True,
            )
        lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

        if self.hpo_config["normalize_observations"]:
            normalizer_state = running_statistics.update(normalizer_state, obs)

        def _learn_epoch(carry, _):
            train_state, rng = carry

            def _learn_phase(carry, minibatch_and_target):

                train_state, rng = carry
                obs, action, target = minibatch_and_target

                def _loss_fn(params):
                    q_vals = self.network.apply(params, obs)
                    chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)
                    loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                    return loss, chosen_action_qvals - target

                (loss, td_error), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, rng), (loss, td_error, grads)

            def preprocess_transition(x, rng):
                x = x.reshape(-1, *x.shape[2:])
                x = jax.random.permutation(rng, x)
                x = x.reshape(
                        self.n_minibatches, -1, *x.shape[1:]
                    )
                return x

            rng, _rng = jax.random.split(rng)
            obs_batches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), obs
                )
            action_batches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), action
                )
            target_batches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, td_error, grads) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (obs_batches, action_batches, target_batches)
                )
            
            if not self.track_metrics:
                loss = None
                td_error = None
                grads = None

            return (train_state, rng), DQNMetrics(loss=loss, td_error=td_error, grads=grads)
        
        rng, _ = jax.random.split(rng)
        (train_state, rng), metrics = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, self.hpo_config["update_epochs"]
            )

        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            normalizer_state=normalizer_state,
            env_state=env_state,
            obs=last_obs,
            global_step=global_step,
        )
        trajectories = None
        if self.track_trajectories:
            trajectories = Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
        return runner_state, (metrics, trajectories)

# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np
import jax.numpy as jnp
import optax
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from flax.training.train_state import TrainState

from flax.core.frozen_dict import FrozenDict

from arlbench.core.algorithms.algorithm import Algorithm
from arlbench.core import running_statistics
from arlbench.core.running_statistics import RunningStatisticsState
from arlbench.utils import recursive_concat, tuple_concat

from .models import CNNActorCritic, MLPActorCritic

if TYPE_CHECKING:
    import chex

    from arlbench.core.environments import Environment
    from arlbench.core.wrappers import AutoRLWrapper


class PPOTrainState(TrainState):
    """PPO training state."""

    opt_state = None

    @classmethod
    def create_with_opt_state(cls, *, apply_fn, params, tx, opt_state, **kwargs):
        if opt_state is None:
            opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class PPORunnerState(NamedTuple):
    """PPO runner state. Consists of (rng, train_state, env_state, obs, global_step)."""

    rng: chex.PRNGKey
    train_state: PPOTrainState
    normalizer_state: RunningStatisticsState
    env_state: Any
    obs: chex.Array
    global_step: int
    return_buffer_idx: chex.Array | None = None
    return_buffer: chex.Array | None = None
    cur_rewards: chex.Array | None = None


class PPOState(NamedTuple):
    """PPO algorithm state. Consists of (runner_state, buffer_state).

    Note: As PPO does not use a buffer buffer_state is always None and only kept for consistency across algorithms.
    """

    runner_state: PPORunnerState
    buffer_state: None = None


class PPOTrainingResult(NamedTuple):
    """PPO training result. Consists of (eval_rewards, trajectories, metrics)."""

    eval_rewards: jnp.ndarray
    trajectories: Transition | None
    metrics: PPOMetrics | None


class PPOMetrics(NamedTuple):
    """PPO metrics returned by train function. Consists of (loss, grads, advantages)."""

    loss: jnp.ndarray
    grads: jnp.ndarray | dict
    advantages: jnp.ndarray


class Transition(NamedTuple):
    """PPO Transition. Consists of (done, action, value, reward, log_prob, obs, info)."""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


PPOTrainReturnT = tuple[PPOState, PPOTrainingResult]


class PPO(Algorithm):
    """JAX-based implementation of Proximal Policy Optimization (PPO)."""

    name: str = "ppo"

    def __init__(
        self,
        hpo_config: Configuration,
        env: Environment | AutoRLWrapper,
        eval_env: Environment | AutoRLWrapper | None = None,
        cnn_policy: bool = False,
        nas_config: Configuration | None = None,
        track_trajectories: bool = False,
        track_metrics: bool = False,
    ) -> None:
        """Creates a PPO algorithm instance.

        Args:
            hpo_config (Configuration): Hyperparameter configuration.
            env (Environment | AutoRLWrapper): Training environment.
            eval_env (Environment | AutoRLWrapper | None, optional): Evaluation environent (otherwise training environment is used for evaluation). Defaults to None.
            cnn_policy (bool, optional): Use CNN network architecture. Defaults to False.
            nas_config (Configuration | None, optional): Neural architecture configuration. Defaults to None.
            track_trajectories (bool, optional):  Track metrics such as loss and gradients during training. Defaults to False.
            track_metrics (bool, optional): Track trajectories during training. Defaults to False.
        """
        if nas_config is None:
            nas_config = PPO.get_default_nas_config()

        super().__init__(
            hpo_config,
            nas_config,
            env,
            eval_env=eval_env,
            track_metrics=track_metrics,
            track_trajectories=track_trajectories,
        )

        # Update interval = rollout size
        self.rollout_size = int(self.hpo_config["n_steps"] * self.env.n_envs)

        # Ensure that at least one minibatch is available after each rollout
        if self.hpo_config["minibatch_size"] > self.rollout_size:
            warnings.warn(
                f"minibatch_size > update_interval. Setting minibatch size to rollout_size = {self.rollout_size}."
            )
            self.minibatch_size = self.rollout_size
        else:
            self.minibatch_size = int(self.hpo_config["minibatch_size"])

        self.n_minibatches = int(self.rollout_size // self.minibatch_size)

        action_size, discrete = self.action_type
        network_cls = CNNActorCritic if cnn_policy else MLPActorCritic
        self.network = network_cls(
            action_size,
            discrete=discrete,
            activation=self.nas_config["activation"],
            hidden_size=self.nas_config["hidden_size"],
        )

    @staticmethod
    def get_hpo_config_space(seed: int | None = None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="PPOConfigSpace",
            seed=seed,
            space={
                "minibatch_size": Integer("minibatch_size", (4, 4096), default=64),
                "learning_rate": Float("learning_rate", (1e-6, 0.1), default=3e-4, log=True),
                "n_steps": Integer("n_steps", (1, 4096), default=2048),
                "update_epochs": Integer("update_epochs", (1, 32), default=10),
                "gamma": Float("gamma", (0.8, 1.0), default=0.99),
                "gae_lambda": Float("gae_lambda", (0.8, 1.0), default=0.95),
                "clip_eps": Float("clip_eps", (0.0, 0.5), default=0.2),
                "vf_clip_eps": Float("vf_clip_eps", (0.0, 0.5), default=0.2),
                "normalize_advantage": Categorical("normalize_advatange", [True, False], default=True),
                "ent_coef": Float("ent_coef", (0.0, 0.5), default=0.0),
                "vf_coef": Float("vf_coef", (0.0, 1.0), default=0.5),
                "max_grad_norm": Float("max_grad_norm", (0.0, 1.0), default=0.5),
                "normalize_observations": Categorical("normalize_observations", [True, False], default=False),
            },
        )

    @staticmethod
    def get_default_hpo_config() -> Configuration:
        return PPO.get_hpo_config_space().get_default_configuration()

    @staticmethod
    def get_nas_config_space(seed: int | None = None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="PPONASConfigSpace",
            seed=seed,
            space={
                "activation": Categorical(
                    "activation", ["tanh", "relu"], default="tanh"
                ),
                "hidden_size": Integer("hidden_size", (1, 2048), default=64),
            },
        )

    @staticmethod
    def get_default_nas_config() -> Configuration:
        return PPO.get_nas_config_space().get_default_configuration()

    @staticmethod
    def get_checkpoint_factory(
        runner_state: PPORunnerState,
        train_result: PPOTrainingResult | None,
    ) -> dict[str, Callable]:
        """Creates a factory dictionary of all posssible checkpointing options for PPO.

        Args:
            runner_state (PPORunnerState): Algorithm runner state.
            train_result (PPOTrainingResult | None): Training result.

        Returns:
            dict[str, Callable]: Dictionary of factory functions containing [opt_state, params, loss, trajectories].
        """
        train_state = runner_state.train_state

        def get_trajectories() -> dict | None:
            if train_result is None or train_result.trajectories is None:
                return None

            traj = train_result.trajectories

            trajectories = {}
            trajectories["states"] = jnp.concatenate(traj.obs, axis=0)
            trajectories["action"] = jnp.concatenate(traj.action, axis=0)
            trajectories["reward"] = jnp.concatenate(traj.reward, axis=0)
            trajectories["dones"] = jnp.concatenate(traj.done, axis=0)
            trajectories["value"] = jnp.concatenate(traj.value, axis=0)
            trajectories["log_prob"] = jnp.concatenate(traj.log_prob, axis=0)

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
    ) -> PPOState:
        """Initializes PPO state. Passed parameters are not initialized and included in the final state.

        Args:
            rng (chex.PRNGKey): Random generator key.
            network_params (FrozenDict | dict | None, optional): Network parameters. Defaults to None.
            opt_state (optax.OptState | None, optional): Optimizer state. Defaults to None.

        Returns:
            PPOState: PPO state.
        """
        rng, reset_rng = jax.random.split(rng)
        env_state, obs = self.env.reset(reset_rng)

        if network_params is None:
            rng, init_rng = jax.random.split(rng)
            network_params = self.network.init(init_rng, obs)

        tx = optax.chain(
            optax.clip_by_global_norm(self.hpo_config["max_grad_norm"]),
            optax.adam(
                self.hpo_config["learning_rate"],
                eps=1e-5
            ),
        )

        train_state = PPOTrainState.create_with_opt_state(
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
            opt_state=opt_state,
        )

        runner_state = PPORunnerState(
            rng=rng,
            train_state=train_state,
            normalizer_state=running_statistics.init_state(obs[0]),
            env_state=env_state,
            obs=obs,
            global_step=0,
            return_buffer_idx=jnp.array([0]),
            return_buffer=jnp.zeros(100),
            cur_rewards=jnp.zeros(self.env.n_envs),
        )

        return PPOState(runner_state=runner_state, buffer_state=None)

    @functools.partial(jax.jit, static_argnums=0)
    def predict(
        self,
        runner_state: PPORunnerState,
        obs: jnp.ndarray,
        rng: chex.PRNGKey,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Predict action(s) based on the current observation(s).

        Args:
            runner_state (PPORunnerState): Algorithm runner state.
            obs (jnp.ndarray): Observation(s).
            rng (chex.PRNGKey | None, optional): Random generator key. Defaults to None.
            deterministic (bool): Return deterministic action. Defaults to True.

        Returns:
            jnp.ndarray: Action(s).
        """
        if self.hpo_config["normalize_observations"]:
            obs = running_statistics.normalize(obs, runner_state.normalizer_state)
        pi, _ = self.network.apply(runner_state.train_state.params, obs)

        def deterministic_action() -> jnp.ndarray:
            return pi.mode()

        def sampled_action() -> jnp.ndarray:
            return pi.sample(seed=rng)

        return jax.lax.cond(
            deterministic,
            deterministic_action,
            sampled_action,
        )

        # return jnp.clip(action, self.env.action_space.low, self.env.action_space.high)

    @functools.partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def train(
        self,
        runner_state: PPORunnerState,
        _,
        n_total_timesteps: int = 1000000,
        n_eval_steps: int = 100,
        n_eval_episodes: int = 10,
    ) -> PPOTrainReturnT:
        """Performs one iteration of training.

        Args:
            runner_state (PPORunnerState): PPO runner state.
            _ (None): Unused parameter (buffer_state in other algorithms).
            n_total_timesteps (int, optional): Total number of training timesteps. Update steps = n_total_timesteps // n_envs. Defaults to 1000000.
            n_eval_steps (int, optional): Number of evaluation steps during training. Defaults to 100.
            n_eval_episodes (int, optional): Number of evaluation episodes per evaluation during training. Defaults to 10.

        Returns:
            PPOTrainReturnT: Tuple of PPO algorithm state and training result.
        """

        def train_eval_step(
            _runner_state: PPORunnerState, _: None
        ) -> tuple[PPORunnerState, PPOTrainingResult]:
            """Performs one iteration of training and evaluation.

            Args:
                _runner_state (PPORunnerState): PPO runner state.
                _ (None): Unused parameter (required for jax.lax.scan).

            Returns:
                tuple[PPORunnerState, PPOTrainingResult]: Tuple of PPO runner state and training result.
            """
            _runner_state, (metrics, trajectories) = jax.lax.scan(
                self._update_step,
                _runner_state,
                None,
                np.ceil(n_total_timesteps / self.env.n_envs / self.hpo_config["n_steps"] / n_eval_steps),
            )
            eval_returns = self.eval(_runner_state, n_eval_episodes)
            jax.debug.print("{ret}", ret=eval_returns.mean())

            return _runner_state, PPOTrainingResult(
                eval_rewards=eval_returns, trajectories=trajectories, metrics=metrics
            )

        runner_state, train_result = jax.lax.scan(
            train_eval_step,
            runner_state,
            None,
            n_eval_steps,
        )
        return PPOState(runner_state=runner_state), train_result

    @functools.partial(jax.jit, static_argnums=0)
    def _update_step(
        self, runner_state: PPORunnerState, _: None
    ) -> tuple[PPORunnerState, tuple[PPOMetrics | None, Transition | None]]:
        """Performs one PPO step of rollout and update.

        Args:
            runner_state (PPORunnerState): PPO runner state.
            _ (None): Unused parameter (required for jax.lax.scan).

        Returns:
            tuple[PPORunnerState, tuple[PPOMetrics | None, Transition | None]]: Tuple of PPO runner state and tuple of metrics and trajectories (if tracked).
        """
        runner_state, traj_batch = jax.lax.scan(
            self._env_step, runner_state, None, self.hpo_config["n_steps"]
        )
        (rng, train_state, normalizer_state, env_state, last_obs, global_step, return_buffer_idx, return_buffer, cur_rewards) = runner_state
        if self.hpo_config["normalize_observations"]:
            normalizer_state = running_statistics.update(normalizer_state, traj_batch.obs)
            traj_batch = Transition(
                done = traj_batch.done,
                action = traj_batch.action,
                value = traj_batch.value,
                reward = traj_batch.reward,
                log_prob = traj_batch.log_prob,
                obs = running_statistics.normalize(traj_batch.obs, normalizer_state),
                info = traj_batch.info,
            )
            _, last_val = self.network.apply(train_state.params, running_statistics.normalize(last_obs, normalizer_state))
        else:
            _, last_val = self.network.apply(train_state.params, last_obs)


        # Calculate advantage
        advantages, targets = self._calculate_gae(traj_batch, last_val)

        # Update network parameters
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, (loss, grads) = jax.lax.scan(
            self._update_epoch, update_state, None, self.hpo_config["update_epochs"]
        )
        train_state = update_state[0]
        rng = update_state[-1]

        runner_state = PPORunnerState(
            rng=rng,
            train_state=train_state,
            normalizer_state=normalizer_state,
            env_state=env_state,
            obs=last_obs,
            global_step=global_step,
            return_buffer_idx=return_buffer_idx,
            return_buffer=runner_state.return_buffer,
            cur_rewards=runner_state.cur_rewards
        )
        metrics, trajectories = None, None
        if self.track_metrics:
            metrics = PPOMetrics(loss=loss, grads=grads, advantages=advantages)
        if self.track_trajectories:
            trajectories = traj_batch
        return runner_state, (metrics, trajectories)

    @functools.partial(jax.jit, static_argnums=0)
    def _env_step(
        self, runner_state: PPORunnerState, _: None
    ) -> tuple[PPORunnerState, Transition]:
        """Perform one environment step (n_envs step in case of parallel environments).

        Args:
            runner_state (PPORunnerState): PPO runner state.
            _ (None): Unused parameter (required for jax.lax.scan).

        Returns:
            tuple[PPORunnerState, Transition]: Tuple of PPO runner state and batch of transitions.
        """
        (rng, train_state, normalizer_state, env_state, last_obs, global_step, return_buffer_idx, return_buffer, cur_rewards) = runner_state

        # Select action(s)
        rng, _rng = jax.random.split(rng)
        if self.hpo_config["normalize_observations"]:
            pi, value = self.network.apply(train_state.params, running_statistics.normalize(last_obs, normalizer_state))
        else:
            pi, value = self.network.apply(train_state.params, last_obs)

        action, log_prob = pi.sample_and_log_prob(seed=_rng)

        clipped_action = action
        if not self.action_type[1]:  # continuous action space
            clipped_action = jnp.clip(
                action, self.env.action_space.low, self.env.action_space.high
            )

        # Perform env step
        rng, _rng = jax.random.split(rng)
        env_state, (obsv, reward, done, info) = self.env.step(
            env_state, clipped_action, _rng
        )
        global_step += 1

        transition = Transition(done, action, value, reward, log_prob, last_obs, info)
        cur_rewards += reward
        print_reward = jnp.array([False])
        for i in range(self.env.n_envs):
            def rew_update(i, return_buffer, return_buffer_idx):
                return_buffer = return_buffer.at[return_buffer_idx%100].set(cur_rewards[i])
                return_buffer_idx += 1
                return return_buffer, return_buffer_idx, return_buffer_idx%100==0
            return_buffer, return_buffer_idx, cur_print_rew = jax.lax.cond(
                done[i],
                lambda rew, idx: rew_update(i, rew, idx),
                lambda rew, idx: (rew, idx, jnp.array([False])),
                return_buffer, return_buffer_idx
            )
            print_reward = jnp.logical_or(print_reward, cur_print_rew)
        cur_rewards *= (1 - done)

        def print_return(return_buffer):
            #jax.debug.print("Current Return: {rew}", rew=return_buffer.mean())
            return return_buffer
        jax.lax.cond(
            print_reward[0],
            print_return,
            lambda x: x,
            return_buffer,
        )

        runner_state = PPORunnerState(
            train_state=train_state,
            normalizer_state=normalizer_state,
            env_state=env_state,
            obs=obsv,
            rng=rng,
            global_step=global_step,
            return_buffer_idx=return_buffer_idx,
            return_buffer=return_buffer,
            cur_rewards=cur_rewards,
        )
        return runner_state, transition

    @functools.partial(jax.jit, static_argnums=0)
    def _calculate_gae(
        self, transition_batch: Transition, value: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generalized advantage estimation.

        Args:
            transition_batch (Transition): Batch of transitions (rollout).
            value (jnp.ndarray): Previous value estimation for each transition.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: (advantages, targets). Tuple of advantage and estimated return for each transition of the batch.
        """
        _, advantages = jax.lax.scan(
            self._get_advantages,
            (jnp.zeros_like(value), value),
            transition_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + transition_batch.value

    @functools.partial(jax.jit, static_argnums=0)
    def _get_advantages(
        self,
        gae_and_next_value: tuple[jnp.ndarray, jnp.ndarray],
        transitions: Transition,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Calculate advantages for a transition batch compatible with jax.lax.scan.

        Args:
            gae_and_next_value (tuple[jnp.ndarray, jnp.ndarray]): Current loop state including gae and value estimation.
            transitions (Transition): Transition batch.

        Returns:
            tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]: ((gae, value), gae). Tuple of loop variable (gae, value) and advantages estimation.
        """
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transitions.done,
            transitions.value,
            transitions.reward,
        )
        delta = reward + self.hpo_config["gamma"] * next_value * (1 - done) - value
        gae = (
            delta
            + self.hpo_config["gamma"]
            * self.hpo_config["gae_lambda"]
            * (1 - done)
            * gae
        )
        return (gae, value), gae

    @functools.partial(jax.jit, static_argnums=0)
    def _update_epoch(
        self,
        update_state: tuple[
            PPOTrainState, Transition, jnp.ndarray, jnp.ndarray, chex.PRNGKey
        ],
        _: None,
    ) -> tuple[
        tuple[PPOTrainState, Transition, jnp.ndarray, jnp.ndarray, chex.PRNGKey],
        tuple[tuple | None, FrozenDict | None],
    ]:
        """One epoch of network updates using minibatches of the current transition batch.

        Args:
            update_state (tuple[PPOTrainState, Transition, jnp.ndarray, jnp.ndarray, chex.PRNGKey]): (train_state, transition_batch, advantages, targets, rng) Current update state.
            _ (None): Unused parameter (required for jax.lax.scan).

        Returns:
            tuple[tuple[PPOTrainState, Transition, jnp.ndarray, jnp.ndarray, chex.PRNGKey], tuple[tuple | None, tuple | None]]: Tuple of (train_state, transition_batch, advantages, targets, rng) and (loss, grads) if tracked.
        """
        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)

        batch_size = self.rollout_size
        trimmed_batch_size = self.n_minibatches * self.minibatch_size

        permutation = jax.random.permutation(_rng, batch_size)
        batch = (traj_batch, advantages, targets)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
        )

        trimmed_batch = jax.tree_util.tree_map(
            lambda x: x[:trimmed_batch_size], shuffled_batch
        )

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [self.n_minibatches, -1, *list(x.shape[1:])]),
            trimmed_batch,
        )

        train_state, (loss, grads) = jax.lax.scan(
            self._update_minibatch, train_state, minibatches
        )

        if trimmed_batch_size < batch_size:
            remaining_batch = jax.tree_util.tree_map(
                lambda x: x[trimmed_batch_size:], shuffled_batch
            )
            remaining_minibatch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [1, -1, *list(x.shape[1:])]),
                remaining_batch,
            )
            train_state, (remaining_loss, remaining_grads) = jax.lax.scan(
                self._update_minibatch, train_state, remaining_minibatch
            )
            if self.track_metrics:
                loss = jax.tree_util.tree_map(
                    lambda x, y: jnp.concatenate((x, y), axis=0), loss, remaining_loss
                )
                grads = jax.tree_util.tree_map(
                    lambda x, y: jnp.concatenate((x, y), axis=0), grads, remaining_grads
                )

        update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, (loss, grads) if self.track_metrics else (None, None)

    @functools.partial(jax.jit, static_argnums=0)
    def _update_minibatch(
        self,
        train_state: PPOTrainState,
        batch_info: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[PPOTrainState, tuple[tuple | None, FrozenDict | None]]:
        """Update network parameters using one minibatch.

        Args:
            train_state (PPOTrainState): PPO training state.
            batch_info (tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): Minibatch of transitions, advantages and targets.

        Returns:
            tuple[PPOTrainState, tuple[tuple | None, tuple | None]]: Tuple of PPO train state and (loss, grads) if tracked.
        """
        traj_batch, advantages, targets = batch_info

        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

        train_state = train_state.apply_gradients(grads=grads)

        out = (total_loss, grads) if self.track_metrics else (None, None)
        return train_state, out

    @functools.partial(jax.jit, static_argnums=0)
    def _loss_fn(
        self,
        params: FrozenDict | dict,
        traj_batch: Transition,
        gae: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Calculate loss given the current batch of transitions.

        Args:
            params (FrozenDict | dict): Network parameters.
            traj_batch (Transition): Batch of transitions
            gae (jnp.ndarray): Advantages.
            targets (jnp.ndarray): Targets.

        Returns:
            tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]: Tuple of (total_loss, (value_loss, actor_loss, entropy)).
        """
        # Rerun network
        pi, value = self.network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # Calculate value loss
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-self.hpo_config["vf_clip_eps"], self.hpo_config["vf_clip_eps"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # Calculate actor loss
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        if self.hpo_config["normalize_advantage"]:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - self.hpo_config["clip_eps"],
                1.0 + self.hpo_config["clip_eps"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.hpo_config["vf_coef"] * value_loss
            - self.hpo_config["ent_coef"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)

# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from ConfigSpace import (Categorical, Configuration, ConfigurationSpace, Float,
                         Integer)
from flax.training.train_state import TrainState

from arlbench.core.algorithms.algorithm import Algorithm
from arlbench.utils import flatten_dict

from .models import CNNActorCritic, MLPActorCritic

if TYPE_CHECKING:
    import chex

    from arlbench.core.environments import Environment
    from arlbench.core.wrappers import AutoRLWrapper


class PPOTrainState(TrainState):
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
    rng: chex.PRNGKey
    train_state: PPOTrainState
    env_state: Any
    obs: chex.Array
    global_step: int

class PPOState(NamedTuple):
    runner_state: PPORunnerState
    buffer_state: None = None

class PPOTrainingResult(NamedTuple):
    eval_rewards: jnp.ndarray
    trajectories: Transition | None
    metrics: PPOMetrics | None

class PPOMetrics(NamedTuple):
    loss: Any
    grads: Any
    advantages: Any

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

PPOTrainReturnT = tuple[PPOState, PPOTrainingResult]

class PPO(Algorithm):
    name = "ppo"

    def __init__(
        self,
        hpo_config: Configuration,
        env: Environment | AutoRLWrapper,
        cnn_policy: bool = False,
        nas_config: Configuration | None = None,
        track_trajectories=False,
        track_metrics=False
    ) -> None:
        if nas_config is None:
            nas_config = PPO.get_default_nas_config()

        super().__init__(
            hpo_config,
            nas_config,
            env,
            track_metrics=track_metrics,
            track_trajectories=track_trajectories
        )

        # compute actual update interval based on n_envs * env_steps
        self.update_interval = int(self.hpo_config["n_steps"] * self.env.n_envs)

        # ensure that at least one minibatch is available after each rollout
        if self.hpo_config["minibatch_size"] > self.update_interval:
            # todo: add a warning here
            self.minibatch_size = self.update_interval
        else:
            self.minibatch_size = int(self.hpo_config["minibatch_size"])

        self.n_minibatches = int(self.update_interval // self.minibatch_size)

        action_size, discrete = self.action_type
        if cnn_policy:
            self.network = CNNActorCritic(
                action_size,
                discrete=discrete,
                activation=self.nas_config["activation"],
                hidden_size=self.nas_config["hidden_size"],
            )
        else:
            self.network = MLPActorCritic(
                action_size,
                discrete=discrete,
                activation=self.nas_config["activation"],
                hidden_size=self.nas_config["hidden_size"],
            )

    @staticmethod
    def get_hpo_config_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="PPOConfigSpace",
            seed=seed,
            space={
                "minibatch_size": Integer("minibatch_size", (4, 1024), default=256),
                "lr": Float("lr", (1e-5, 0.1), default=2.5e-4),
                "n_steps": Integer("n_steps", (1, 1000), default=100),
                "update_epochs": Integer("update_epochs", (1, int(1e5)), default=10),
                "activation": Categorical("activation", ["tanh", "relu"], default="tanh"),
                "gamma": Float("gamma", (0., 1.), default=0.99),
                "gae_lambda": Float("gae_lambda", (0., 1.), default=0.95),
                "clip_eps": Float("clip_eps", (0., 1.), default=0.2),
                "ent_coef": Float("ent_coef", (0., 1.), default=0.01),
                "vf_coef": Float("vf_coef", (0., 1.), default=0.5),
                "max_grad_norm": Float("max_grad_norm", (0., 10.), default=5)
            },
        )

    @staticmethod
    def get_default_hpo_config() -> Configuration:
        return PPO.get_hpo_config_space().get_default_configuration()

    @staticmethod
    def get_nas_config_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="PPONASConfigSpace",
            seed=seed,
            space={
                "activation": Categorical("activation", ["tanh", "relu"], default="tanh"),
                "hidden_size": Integer("hidden_size", (1, 1024), default=64),
            },
        )

    @staticmethod
    def get_default_nas_config() -> Configuration:
        return PPO.get_nas_config_space().get_default_configuration()

    @staticmethod
    def get_checkpoint_factory(
        runner_state: PPORunnerState,
        train_result: PPOTrainingResult,
    ) -> dict[str, Callable]:
        train_state = runner_state.train_state

        def get_trajectories():
            traj = train_result.trajectories
            assert traj is not None

            trajectories = {}
            trajectories["states"] = jnp.concatenate(traj.obs, axis=0)
            trajectories["action"] = jnp.concatenate(traj.action, axis=0)
            trajectories["reward"] = jnp.concatenate(traj.reward, axis=0)
            trajectories["dones"] = jnp.concatenate(traj.done, axis=0)
            trajectories["value"] = jnp.concatenate(traj.value, axis=0)
            trajectories["log_prob"] = jnp.concatenate(traj.log_prob, axis=0)

            return trajectories

        return {
            "opt_state": lambda : train_state.opt_state,
            "params": lambda : train_state.params,
            "loss": lambda : train_result.metrics.loss if train_result.metrics else None,
            "trajectories": get_trajectories
        }

    def init(self, rng, network_params=None, opt_state=None) -> PPOState:
        rng, reset_rng = jax.random.split(rng)
        env_state, obs = self.env.reset(reset_rng)

        if network_params is None:
            rng, init_rng = jax.random.split(rng)
            network_params = self.network.init(init_rng, obs)

        tx = optax.chain(
                optax.clip_by_global_norm(self.hpo_config["max_grad_norm"]),
                optax.adam(self.hpo_config["lr"], eps=1e-5),
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
            env_state=env_state,
            obs=obs,
            global_step=0
        )

        return PPOState(runner_state=runner_state, buffer_state=None)

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, runner_state, obs, rng) -> int:
        pi, _ = self.network.apply(runner_state.train_state.params, obs)
        return pi.sample(seed=rng)

    @functools.partial(jax.jit, static_argnums=(0,3,4,5))
    def train(
        self,
        runner_state: PPORunnerState,
        _,  # dummy for bufer state
        n_total_timesteps: int = 1000000,
        n_eval_steps:  int= 100,
        n_eval_episodes: int = 10,
    ) -> PPOTrainReturnT:
        def train_eval_step(_runner_state, _):
            _runner_state, (metrics, trajectories) = jax.lax.scan(
                self._update_step,
                _runner_state,
                None,
                n_total_timesteps // self.update_interval // n_eval_steps
            )
            eval_returns = self.eval(_runner_state, n_eval_episodes)

            return _runner_state, (metrics, trajectories, eval_returns)

        runner_state, (metrics, trajectories, eval_returns) = jax.lax.scan(
            train_eval_step,
            runner_state,
            None,
            n_eval_steps,
        )
        return PPOState(
            runner_state=runner_state
        ), PPOTrainingResult(
            eval_rewards=eval_returns,
            metrics=metrics,
            trajectories=trajectories
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _update_step(
        self,
        runner_state: PPORunnerState,
        _
    ) -> tuple[PPORunnerState, tuple[PPOMetrics | None, Transition | None]]:
        runner_state, traj_batch = jax.lax.scan(
            self._env_step, runner_state, None, self.hpo_config["n_steps"]
        )

        # CALCULATE ADVANTAGE
        (
            rng,
            train_state,
            env_state,
            last_obs,
            global_step
        ) = runner_state
        _, last_val = self.network.apply(train_state.params, last_obs)

        advantages, targets = self._calculate_gae(traj_batch, last_val)

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, (
            loss,
            grads,
            minibatches
        ) = jax.lax.scan(self._update_epoch, update_state, None, self.hpo_config["update_epochs"])
        train_state = update_state[0]
        rng = update_state[-1]

        runner_state = PPORunnerState(
            rng=rng,
            train_state=train_state,
            env_state=env_state,
            obs=last_obs,
            global_step=global_step
        )
        metrics, tracjectories = None, None
        if self.track_metrics:
            metrics = PPOMetrics(
                loss=loss,
                grads=grads,
                advantages=advantages
            )
        if self.track_trajectories:
            tracjectories = traj_batch
        return runner_state, (metrics, tracjectories)

    @functools.partial(jax.jit, static_argnums=0)
    def _env_step(self, runner_state, _):
        (
            rng,
            train_state,
            env_state,
            last_obs,
            global_step
        ) = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = self.network.apply(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        env_state, (obsv, reward, done, info) = self.env.step(env_state, action, _rng)
        global_step += self.env.n_envs

        transition = Transition(
            done, action, value, reward, log_prob, last_obs, info
        )
        runner_state = PPORunnerState(
            train_state=train_state,
            env_state=env_state,
            obs=obsv,
            rng=rng,
            global_step=global_step
        )
        return runner_state, transition

    @functools.partial(jax.jit, static_argnums=0)
    def _calculate_gae(self, traj_batch, last_val):
        _, advantages = jax.lax.scan(
            self._get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    @functools.partial(jax.jit, static_argnums=0)
    def _get_advantages(self, gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + self.hpo_config["gamma"] * next_value * (1 - done) - value
        gae = (
            delta
            + self.hpo_config["gamma"] * self.hpo_config["gae_lambda"] * (1 - done) * gae
        )
        return (gae, value), gae

    @functools.partial(jax.jit, static_argnums=0)
    def _update_epoch(self, update_state, _):
        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)

        batch_size = self.update_interval
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
            lambda x: jnp.reshape(
                x, [self.n_minibatches, -1, *list(x.shape[1:])]
            ),
            trimmed_batch,
        )

        train_state, (total_loss, grads) = jax.lax.scan(
            self._update_minbatch, train_state, minibatches
        )

        if trimmed_batch_size < batch_size:
            remaining_batch = jax.tree_util.tree_map(
                lambda x: x[trimmed_batch_size:], shuffled_batch
            )
            remaining_minibatch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [1, -1, *list(x.shape[1:])]
                ),
                remaining_batch,
            )
            train_state, (remaining_total_loss, remaining_grads) = jax.lax.scan(
                self._update_minbatch, train_state, remaining_minibatch
            )
            if self.track_metrics:
                total_loss = (*total_loss, *remaining_total_loss)
                grads = (*grads, *remaining_grads)

        update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, (
            total_loss,
            grads,
            minibatches
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _update_minbatch(self, train_state, batch_info):
        traj_batch, advantages, targets = batch_info

        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params, traj_batch, advantages, targets
        )

        train_state = train_state.apply_gradients(grads=grads)

        # TODO find a better way of doing this
        grads = flatten_dict(grads)

        out = (total_loss, grads) if self.track_metrics else (None, None)
        return train_state, out

    @functools.partial(jax.jit, static_argnums=0)
    def _loss_fn(self, params, traj_batch, gae, targets):
         # RERUN NETWORK
        pi, value = self.network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-self.hpo_config["clip_eps"], self.hpo_config["clip_eps"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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


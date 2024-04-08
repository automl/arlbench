# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple

import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
import optax
from ConfigSpace import (Categorical, Configuration, ConfigurationSpace, Float,
                         Integer)
from flax.training.train_state import TrainState

from arlbench.core.environments import AutoRLEnv

from .algorithm import Algorithm
from .common import TimeStep
from .models import ActorCritic

if TYPE_CHECKING:
    import chex

    from arlbench.core.environments import AutoRLEnv
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class PPO(Algorithm):
    def __init__(
        self,
        hpo_config: Configuration | dict,
        env_options: dict,
        env: AutoRLEnv | AutoRLWrapper,
        nas_config: Configuration | dict | None = None,
        track_trajectories=False,
        track_metrics=False
    ) -> None:
        if nas_config is None:
            nas_config = PPO.get_default_nas_config()

        super().__init__(
            hpo_config,
            nas_config,
            env_options,
            env,
            track_metrics=track_metrics,
            track_trajectories=track_trajectories
        )
        self.n_minibatches = self.env.n_envs * env_options["n_env_steps"] // self.hpo_config["minibatch_size"]

        action_size, discrete = self.action_type
        self.network = ActorCritic(
            action_size,
            discrete=discrete,
            activation=self.nas_config["activation"],
            hidden_size=self.nas_config["hidden_size"],
        )

        self.buffer = fbx.make_flat_buffer(
            max_length=self.hpo_config["buffer_size"],
            min_length=self.hpo_config["buffer_batch_size"],
            sample_batch_size=self.hpo_config["buffer_batch_size"],
            add_sequences=False,
            add_batch_size=self.env.n_envs
        )

        self.n_total_updates = (
            env_options["n_total_timesteps"]
            // env_options["n_env_steps"]
            // self.env.n_envs
        )
        update_interval = jnp.ceil(self.n_total_updates / env_options["n_env_steps"])
        if update_interval < 1:
            update_interval = 1
            print(
                "WARNING: The number of iterations selected in combination with your timestep, n_envs and n_env_steps settings results in 0 steps per iteration. Rounded up to 1, this means more total steps will be executed."
            )

    @staticmethod
    def get_hpo_config_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="PPOConfigSpace",
            seed=seed,
            space={
                "buffer_size": Integer("buffer_size", (1, int(1e7)), default=int(1e6)),
                "buffer_batch_size": Integer("buffer_batch_size", (1, 1024), default=64),
                "minibatch_size": Integer("minibatch_size", (4, 1024), default=256),
                "lr": Float("lr", (1e-5, 0.1), default=2.5e-4),
                "update_epochs": Integer("update_epochs", (1, int(1e5)), default=10),
                "activation": Categorical("activation", ["tanh", "relu"], default="tanh"),
                "hidden_size": Integer("hidden_size", (1, 1024), default=64),
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

    def init(self, rng, buffer_state=None, network_params=None, opt_state=None):
        rng, _rng = jax.random.split(rng)
        env_state, obs = self.env.reset(_rng)

        if buffer_state is None or network_params is None:
            dummy_rng = jax.random.PRNGKey(0)
            _action = self.env.sample_actions(dummy_rng)
            _, (_obs, _reward, _done, _) = self.env.step(env_state, _action, dummy_rng)

        if buffer_state is None:
            _timestep = TimeStep(last_obs=_obs[0], obs=_obs[0], action=_action[0], reward=_reward[0], done=_done[0])
            buffer_state = self.buffer.init(_timestep)

        _, _rng = jax.random.split(rng)
        if network_params is None:
            network_params = self.network.init(_rng, _obs)

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

        rng, _rng = jax.random.split(rng)

        runner_state = PPORunnerState(
            rng=_rng,
            train_state=train_state,
            env_state=env_state,
            obs=obs,
        )

        return runner_state, buffer_state

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, runner_state, obs, rng) -> int:
        pi, _ = self.network.apply(runner_state.train_state.params, obs)
        return pi.sample(seed=rng)

    @functools.partial(jax.jit, static_argnums=0, donate_argnums=(2,))
    def train(
        self,
        runner_state,
        buffer_state
    ) -> tuple[tuple[PPORunnerState, Any], tuple | None]:
        (runner_state, buffer_state), out = jax.lax.scan(
            self._update_step, (runner_state, buffer_state), None, self.n_total_updates
        )
        return (runner_state, buffer_state), out

    @functools.partial(jax.jit, static_argnums=0)
    def _update_step(
        self,
        carry,
        _
    ):
        runner_state, buffer_state = carry
        (runner_state, buffer_state), traj_batch = jax.lax.scan(
            self._env_step, (runner_state, buffer_state), None, self.env_options["n_env_steps"]
        )

        # CALCULATE ADVANTAGE
        (
            rng,
            train_state,
            env_state,
            last_obs
        ) = runner_state
        _, last_val = self.network.apply(train_state.params, last_obs)

        advantages, targets = self._calculate_gae(traj_batch, last_val)

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, (
            loss_info,
            grads,
            minibatches,
            param_hist,
        ) = jax.lax.scan(self._update_epoch, update_state, None, self.hpo_config["update_epochs"])
        train_state = update_state[0]
        rng = update_state[-1]

        runner_state = PPORunnerState(
            rng=rng,
            train_state=train_state,
            env_state=env_state,
            obs=last_obs
        )
        if self.track_trajectories:
            out = (
                loss_info,
                grads,
                traj_batch,
                {
                    "advantages": advantages,
                    "param_history": param_hist["params"],
                    "minibatches": minibatches,
                },
            )
        elif self.track_metrics:
            out = (
                loss_info,
                grads,
                {"advantages": advantages, "param_history": param_hist["params"]},
            )
        else:
            out = None
        return (runner_state, buffer_state), out

    @functools.partial(jax.jit, static_argnums=0)
    def _env_step(self, carry, _):
        runner_state, buffer_state = carry
        (
            rng,
            train_state,
            env_state,
            last_obs
        ) = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = self.network.apply(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        env_state, (obsv, reward, done, info) = self.env.step(env_state, action, _rng)

        timestep = TimeStep(last_obs=last_obs, obs=obsv, action=action, reward=reward, done=done)
        buffer_state = self.buffer.add(buffer_state, timestep)

        transition = Transition(
            done, action, value, reward, log_prob, last_obs, info
        )
        runner_state = PPORunnerState(
            train_state=train_state,
            env_state=env_state,
            obs=obsv,
            rng=rng
        )
        return (runner_state, buffer_state), transition

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

        trimmed_batch_size = int(self.n_minibatches * self.hpo_config["minibatch_size"])
        batch_size = self.env_options["n_env_steps"] * self.env.n_envs

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
        update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, (
            total_loss,
            grads,
            minibatches,
            train_state.params.unfreeze().copy() if isinstance(train_state.params, flax.core.FrozenDict) else train_state.params.copy(),
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _update_minbatch(self, train_state, batch_info):
        traj_batch, advantages, targets = batch_info

        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params, traj_batch, advantages, targets
        )
        train_state = train_state.apply_gradients(grads=grads)
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


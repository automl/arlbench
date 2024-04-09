# The DQN Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple

import flashbax as fbx
import jax
import jax.lax
import jax.numpy as jnp
import optax
from ConfigSpace import (Categorical, Configuration, ConfigurationSpace, Float,
                         Integer)
from flax.training.train_state import TrainState

from .models import MLPQ, CNNQ
from ..algorithm import Algorithm
from ..buffers import uniform_sample
from ..common import TimeStep

if TYPE_CHECKING:
    import chex


class DQNTrainState(TrainState):
    target_params: None | chex.Array | dict = None
    opt_state = None

    @classmethod
    def create_with_opt_state(cls, *, apply_fn, params, target_params, tx, opt_state, **kwargs):
        if opt_state is None:
            opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class DQNRunnerState(NamedTuple):
    rng: chex.PRNGKey
    train_state: DQNTrainState
    env_state: Any
    obs: chex.Array
    global_step: int

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    q_pred: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class DQN(Algorithm):
    def __init__(
        self,
        hpo_config: Configuration | dict,
        options: dict,
        env: Any,
        cnn_policy: bool = False,
        nas_config: Configuration | dict | None = None,
        track_trajectories=False,
        track_metrics=False
    ) -> None:
        if nas_config is None:
            nas_config = DQN.get_default_nas_config()

        super().__init__(
            hpo_config,
            nas_config,
            options,
            env,
            track_trajectories=track_trajectories,
            track_metrics=track_metrics
        )

        action_size, discrete = self.action_type
        if cnn_policy:            
            self.network = CNNQ(
                action_size,
                discrete=discrete,
                activation=self.nas_config["activation"],
                hidden_size=self.nas_config["hidden_size"],
            )
        else:
            self.network = MLPQ(
                action_size,
                discrete=discrete,
                activation=self.nas_config["activation"],
                hidden_size=self.nas_config["hidden_size"],
            )

        priority_exponent = self.hpo_config.get("buffer_beta", 1.0)
        self.buffer = fbx.make_prioritised_flat_buffer(
            max_length=self.hpo_config["buffer_size"],
            min_length=self.hpo_config["buffer_batch_size"],
            sample_batch_size=self.hpo_config["buffer_batch_size"],
            add_sequences=False,
            add_batch_size=self.env.n_envs,
            priority_exponent=priority_exponent
        )
        if self.hpo_config["buffer_prio_sampling"] is True:  # todo: shouldn't this be the other way around?
            sample_fn = functools.partial(
                uniform_sample,
                batch_size=self.hpo_config["buffer_batch_size"],
                sequence_length=2,
                period=1
            )
            self.buffer = self.buffer.replace(sample=sample_fn)

    @staticmethod
    def get_hpo_config_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="DQNConfigSpace",
            seed=seed,
            space={
                "buffer_size": Integer("buffer_size", (1, int(1e7)), default=int(1e6)),
                "buffer_batch_size": Integer("buffer_batch_size", (1, 1024), default=64),
                "buffer_prio_sampling": Categorical("buffer_prio_sampling", [True, False], default=False),
                "buffer_alpha": Float("buffer_alpha", (0., 1.), default=0.9),
                "buffer_beta": Float("buffer_beta", (0., 1.), default=0.9),
                "buffer_epsilon": Float("buffer_epsilon", (0., 1e-3), default=1e-5),
                "lr": Float("lr", (1e-5, 0.1), default=2.5e-4),
                "update_epochs": Integer("update_epochs", (1, int(1e5)), default=10),
                "activation": Categorical("activation", ["tanh", "relu"], default="tanh"),
                "hidden_size": Integer("hidden_size", (1, 1024), default=64),
                "gamma": Float("gamma", (0., 1.), default=0.99),
                "tau": Float("tau", (0., 1.), default=1.0),
                "epsilon": Float("epsilon", (0., 1.), default=0.1),
                "use_target_network": Categorical("use_target_network", [True, False], default=True),
                "train_frequency": Integer("train_frequency", (1, int(1e5)), default=4),
                "learning_starts": Integer("learning_starts", (1024, int(1e5)), default=10000),
                "target_network_update_freq": Integer("target_network_update_freq", (1, int(1e5)), default=100)
            },
        )

        # only use PER parameters if PER is enabled
        # however, we still need the hyperparameters to add samples, even though we don't sampling based on priorities
        # cs.add_conditions([
        #     EqualsCondition(cs["buffer_alpha"], cs["buffer_prio_sampling"], True),
        #     EqualsCondition(cs["buffer_beta"], cs["buffer_prio_sampling"], True),
        #     EqualsCondition(cs["buffer_epsilon"], cs["buffer_prio_sampling"], True)
        # ])


    @staticmethod
    def get_default_hpo_config() -> Configuration:
        return DQN.get_hpo_config_space().get_default_configuration()

    @staticmethod
    def get_nas_config_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="DQNNASConfigSpace",
            seed=seed,
            space={
                "activation": Categorical("activation", ["tanh", "relu"], default="tanh"),
                "hidden_size": Integer("hidden_size", (1, 1024), default=64),
            },
        )


    @staticmethod
    def get_default_nas_config() -> Configuration:
        return DQN.get_nas_config_space().get_default_configuration()

    def init(self, rng, buffer_state=None, network_params=None, target_params=None, opt_state=None) -> tuple[DQNRunnerState, Any]:
        rng, _rng = jax.random.split(rng)

        env_state, obs = self.env.reset(rng)

        if buffer_state is None or network_params is None or target_params is None:
            dummy_rng = jax.random.PRNGKey(0)
            _action = self.env.sample_actions(dummy_rng)
            _, (_obs, _reward, _done, _) = self.env.step(env_state, _action, dummy_rng)

        if buffer_state is None:
            _timestep = TimeStep(last_obs=_obs[0], obs=_obs[0], action=_action[0], reward=_reward[0], done=_done[0])
            buffer_state = self.buffer.init(_timestep)

        _, _rng = jax.random.split(rng)
        if network_params is None:
            network_params = self.network.init(_rng, _obs)
        if target_params is None:
            target_params = self.network.init(_rng, _obs)

        train_state_kwargs = {
            "apply_fn": self.network.apply,
            "params": network_params,
            "target_params": target_params,
            "tx": optax.adam(self.hpo_config["lr"], eps=1e-5),
            "opt_state": opt_state,
        }
        train_state = DQNTrainState.create_with_opt_state(**train_state_kwargs)

        rng, _rng = jax.random.split(rng)
        global_step = 0

        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            env_state=env_state,
            obs=obs,
            global_step=global_step
        )

        return runner_state, buffer_state

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, runner_state, obs, _) -> int:
        q_values = self.network.apply(runner_state.train_state.params, obs)
        return q_values.argmax(axis=-1)

    @functools.partial(jax.jit, static_argnums=0, donate_argnums=(2,))
    def train(
        self,
        runner_state,
        buffer_state
    )-> tuple[tuple[DQNRunnerState, Any], tuple | None]:
        (runner_state, buffer_state), out = jax.lax.scan(
            self._update_step, (runner_state, buffer_state), None, (self.env_options["n_total_timesteps"]//self.hpo_config["train_frequency"])//self.env.n_envs
        )
        return (runner_state, buffer_state), out

    def update(
        self,
        train_state,
        observations,
        actions,
        next_observations,
        rewards,
        dones
    ):
        if self.hpo_config["use_target_network"]:
            q_next_target = self.network.apply(
                train_state.target_params, next_observations
            )  # (batch_size, num_actions)
        else:
            q_next_target = self.network.apply(
                train_state.params, next_observations
            )  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * self.hpo_config["gamma"] * q_next_target

        def mse_loss(params):
            q_pred = self.network.apply(
                params, observations
            )  # (batch_size, num_actions)
            q_pred = q_pred[
                jnp.arange(q_pred.shape[0]), actions.squeeze().astype(int)
            ]  # (batch_size,)
            td_error = q_pred - next_q_value
            return (td_error ** 2).mean(), td_error

        (loss_value, td_error), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss_value, td_error, grads, train_state.opt_state

    def _update_step(
        self,
        carry,
        _
    ):
        runner_state, buffer_state = carry
        (
            rng,
            train_state,
            env_state,
            last_obs,
            global_step
        ) = runner_state
        rng, _rng = jax.random.split(rng)

        def random_action():
            return self.env.sample_actions(rng)

        def greedy_action():
            q_values = self.network.apply(train_state.params, last_obs)
            return q_values.argmax(axis=-1)

        def take_step(carry, _):
            obsv, env_state, global_step, buffer_state = carry
            action = jax.lax.cond(
                jax.random.uniform(rng) < self.hpo_config["epsilon"],
                random_action,
                greedy_action,
            )

            env_state, (obsv, reward, done, info) = self.env.step(env_state, action, _rng)

            timestep = TimeStep(last_obs=last_obs, obs=obsv, action=action, reward=reward, done=done)
            buffer_state = self.buffer.add(buffer_state, timestep)

            # global_step += 1
            global_step += self.env.n_envs
            return (obsv, env_state, global_step, buffer_state), (
                obsv,
                action,
                reward,
                done,
                info,
            )

        def do_update(train_state, buffer_state):
            batch = self.buffer.sample(buffer_state, rng)
            train_state, loss, td_error, grads, opt_state = self.update(
                train_state,
                batch.experience.first.last_obs,
                batch.experience.first.action,
                batch.experience.first.obs,
                batch.experience.first.reward,
                batch.experience.first.done,
            )
            new_prios = jnp.power(
                jnp.abs(td_error) + self.hpo_config["buffer_epsilon"], self.hpo_config["buffer_alpha"]
            )
            buffer_state = self.buffer.set_priorities(buffer_state, batch.indices, new_prios)
            return train_state, loss, td_error, grads, opt_state, buffer_state

        def dont_update(train_state, buffer_state):
            return (
                train_state,
                ((jnp.array([0]) - jnp.array([0])) ** 2).mean(),
                jnp.ones(self.hpo_config["buffer_batch_size"]),
                train_state.params,
                train_state.opt_state,
                buffer_state,
            )

        def target_update():
            return train_state.replace(
                target_params=optax.incremental_update(
                    train_state.params, train_state.target_params, self.hpo_config["tau"]
                )
            )

        def dont_target_update():
            return train_state

        (last_obs, env_state, global_step, buffer_state), (
            observations,
            action,
            reward,
            done,
            info,
        ) = jax.lax.scan(
            take_step,
            (last_obs, env_state, global_step, buffer_state),
            None,
            self.hpo_config["train_frequency"],
        )

        train_state, loss, td_error, grads, opt_state, buffer_state = jax.lax.cond(
            (global_step > self.hpo_config["learning_starts"])
            & (global_step % self.hpo_config["train_frequency"] == 0),
            do_update,
            dont_update,
            train_state,
            buffer_state,
        )
        train_state = jax.lax.cond(
            (global_step > self.hpo_config["learning_starts"])
            & (global_step % self.hpo_config["target_network_update_freq"] == 0),
            target_update,
            dont_target_update,
        )
        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            env_state=env_state,
            obs=last_obs,
            global_step=global_step
        )
        if self.track_trajectories:
            metric = (
                loss,
                grads,
                Transition(
                    obs=observations,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                    q_pred=[jnp.zeros_like(td_error)]  # todo: why are we logging q_pred here?
                ),
                {"td_error": [td_error]},
            )
        elif self.track_metrics:
            metric = (
                loss,
                grads,
                {"q_pred": [jnp.zeros_like(td_error)], "td_error": [td_error]},  # todo: why are we logging q_pred here?
            )
        else:
            metric = None
        return (runner_state, buffer_state), metric
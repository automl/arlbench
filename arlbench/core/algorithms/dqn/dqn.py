# The DQN Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import flashbax as fbx
import jax
import jax.lax
import jax.numpy as jnp
import optax
from ConfigSpace import (Categorical, Configuration, ConfigurationSpace, Float,
                         Integer)
from flax.training.train_state import TrainState

from arlbench.core.algorithms.algorithm import Algorithm
from arlbench.core.algorithms.buffers import uniform_sample
from arlbench.core.algorithms.common import TimeStep

from .models import CNNQ, MLPQ

if TYPE_CHECKING:
    import chex
    from flashbax.buffers.prioritised_trajectory_buffer import \
        PrioritisedTrajectoryBufferState
    from flax.core.frozen_dict import FrozenDict

    from arlbench.core.environments import Environment
    from arlbench.core.wrappers import AutoRLWrapper



class DQNTrainState(TrainState):
    """DQN training state."""
    target_params: None | chex.Array | dict | FrozenDict = None
    opt_state: optax.OptState

    @classmethod
    def create_with_opt_state(cls, *, apply_fn: Callable, params: FrozenDict[str, Any], target_params: Any, tx, opt_state: optax.OptState, **kwargs):
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
    """DQN runner state. Consists of (rng, train_state, env_state, obs, global_step)."""
    rng: chex.PRNGKey
    train_state: DQNTrainState
    env_state: Any
    obs: chex.Array
    global_step: int

class DQNState(NamedTuple):
    """DQN algorithm state. Consists of (runner_state, buffer_state)."""
    runner_state: DQNRunnerState
    buffer_state: PrioritisedTrajectoryBufferState


class DQNTrainingResult(NamedTuple):
    """DQN training result. Consists of (eval_rewards, trajectories, metrics)."""
    eval_rewards: jnp.ndarray
    trajectories: Transition | None
    metrics: DQNMetrics | None

class DQNMetrics(NamedTuple):
    """DQN metrics returned by train function. Consists of (loss, grads, advantages)."""
    loss: jnp.ndarray
    grads: jnp.ndarray | tuple
    td_error: jnp.ndarray

class Transition(NamedTuple):
    """DQN Transition. Consists of (done, action, reward, obs, info)."""
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: dict


DQNTrainReturnT = tuple[DQNState, DQNTrainingResult]


class DQN(Algorithm):
    """JAX-based implementation of Deep Q-Network (DQN)."""
    name: str = "dqn"

    def __init__(
        self,
        hpo_config: Configuration,
        env: Environment | AutoRLWrapper,
        eval_env: Environment | AutoRLWrapper | None = None,
        cnn_policy: bool = False,
        nas_config: Configuration | None = None,
        track_trajectories: bool = False,
        track_metrics: bool = False
    ) -> None:
        """Creates a DQN algorithm instance.

        Args:
            hpo_config (Configuration): Hyperparameter configuration of the algorithm which can be optimized using hyperparameter optimization (HPO).
            env (Environment | AutoRLWrapper): eural architecture of the algorithm components which can be optimized using neural architecture search (NAS).
            eval_env (Environment | AutoRLWrapper | None, optional): _description_. Defaults to None.
            cnn_policy (bool, optional): _description_. Defaults to False.
            nas_config (Configuration | None, optional): _description_. Defaults to None.
            track_trajectories (bool, optional): _description_. Defaults to False.
            track_metrics (bool, optional): _description_. Defaults to False.
        """
        if nas_config is None:
            nas_config = DQN.get_default_nas_config()

        super().__init__(
            hpo_config,
            nas_config,
            env,
            eval_env=eval_env,
            track_trajectories=track_trajectories,
            track_metrics=track_metrics
        )

        action_size, discrete = self.action_type
        network_cls = CNNQ if cnn_policy else MLPQ
        self.network = network_cls(
            action_size,
            discrete=discrete,
            activation=self.nas_config["activation"],
            hidden_size=self.nas_config["hidden_size"],
        )

        self.buffer = fbx.make_prioritised_flat_buffer(
            max_length=self.hpo_config["buffer_size"],
            min_length=self.hpo_config["buffer_batch_size"],
            sample_batch_size=self.hpo_config["buffer_batch_size"],
            add_sequences=False,
            add_batch_size=self.env.n_envs,
            priority_exponent=self.hpo_config["buffer_beta"]
        )
        if self.hpo_config["buffer_prio_sampling"] is False:
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
                "gamma": Float("gamma", (0., 1.), default=0.99),
                "tau": Float("tau", (0., 1.), default=1.0),
                "epsilon": Float("epsilon", (0., 1.), default=0.1),
                "use_target_network": Categorical("use_target_network", [True, False], default=True),
                "train_frequency": Integer("train_frequency", (1, int(1e5)), default=4),
                "gradient steps": Integer("gradient_steps", (1, int(1e5)), default=1),
                "learning_starts": Integer("learning_starts", (10, int(1e5)), default=1000),
                "target_network_update_freq": Integer("target_network_update_freq", (1, int(1e5)), default=10)
            },
        )


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

    @staticmethod
    def get_checkpoint_factory(
        runner_state: DQNRunnerState,
        train_result: DQNTrainingResult,
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

            return trajectories

        return {
            "opt_state": lambda: train_state.opt_state,
            "params": lambda: train_state.params,
            "target_params": lambda: train_state.target_params,
            "loss": lambda: train_result.metrics.loss if train_result.metrics else None,
            "trajectories": get_trajectories
        }

    def init(
            self,
            rng: chex.PRNGKey,
            buffer_state: PrioritisedTrajectoryBufferState | None = None,
            network_params: FrozenDict | dict | None = None,
            target_params: FrozenDict | dict | None = None,
            opt_state: optax.OptState | None = None
        ) -> DQNState:
        rng, reset_rng = jax.random.split(rng)

        env_state, obs = self.env.reset(reset_rng)

        if buffer_state is None or network_params is None or target_params is None:
            dummy_rng = jax.random.PRNGKey(0)
            _action = self.env.sample_actions(dummy_rng)
            _, (_obs, _reward, _done, _) = self.env.step(env_state, _action, dummy_rng)

        if buffer_state is None:
            _timestep = TimeStep(last_obs=_obs[0], obs=_obs[0], action=_action[0], reward=_reward[0], done=_done[0])
            buffer_state = self.buffer.init(_timestep)


        if network_params is None:
            rng, init_rng = jax.random.split(rng)
            network_params = self.network.init(init_rng, _obs)
        if target_params is None:
            rng, target_init_rng = jax.random.split(rng)
            target_params = self.network.init(target_init_rng, _obs)

        train_state_kwargs = {
            "apply_fn": self.network.apply,
            "params": network_params,
            "target_params": target_params,
            "tx": optax.adam(self.hpo_config["lr"], eps=1e-5),
            "opt_state": opt_state,
        }
        train_state = DQNTrainState.create_with_opt_state(**train_state_kwargs)

        global_step = 0

        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            env_state=env_state,
            obs=obs,
            global_step=global_step
        )

        assert buffer_state is not None

        return DQNState(
            runner_state=runner_state,
            buffer_state=buffer_state
        )

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, runner_state: DQNRunnerState, obs: jnp.ndarray, rng: chex.PRNGKey, deterministic: bool = True) -> jnp.ndarray:
        q_values = self.network.apply(runner_state.train_state.params, obs)
        return q_values.argmax(axis=-1)

    @functools.partial(jax.jit, static_argnums=(0,3,4,5), donate_argnums=(2,))
    def train(
        self,
        runner_state: DQNRunnerState,
        buffer_state: PrioritisedTrajectoryBufferState,
        n_total_timesteps: int = 1000000,
        n_eval_steps: int = 100,
        n_eval_episodes: int = 10,
    )-> DQNTrainReturnT:
        def train_eval_step(
            carry: tuple[DQNRunnerState, PrioritisedTrajectoryBufferState],
            _: None
        ) -> tuple[tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], DQNTrainingResult]:
            runner_state, buffer_state = carry
            (runner_state, buffer_state), (metrics, trajectories) = jax.lax.scan(
                self._update_step,
                (runner_state, buffer_state),
                None,
                n_total_timesteps//self.env.n_envs//self.hpo_config["train_frequency"]//n_eval_steps
            )
            eval_returns = self.eval(runner_state, n_eval_episodes)
            jax.debug.print("{eval_returns}", eval_returns=eval_returns)

            return (runner_state, buffer_state), DQNTrainingResult(eval_rewards=eval_returns, trajectories=trajectories, metrics=metrics)

        (runner_state, buffer_state), result = jax.lax.scan(
            train_eval_step,
            (runner_state, buffer_state),
            None,
            n_eval_steps,
        )
        return DQNState(
            runner_state=runner_state,
            buffer_state=buffer_state
        ), result

    def update(
        self,
        train_state: DQNTrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray
    ) -> tuple[DQNTrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        def mse_loss(params: FrozenDict | dict) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        return train_state, loss_value, td_error, grads

    def _update_step(
        self,
        carry: tuple[DQNRunnerState, PrioritisedTrajectoryBufferState],
        _
    ) -> tuple[tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], tuple[DQNMetrics | None, Transition | None]]:
        runner_state, buffer_state = carry
        (
            rng,
            train_state,
            env_state,
            last_obs,
            global_step
        ) = runner_state

        def random_action(rng: chex.PRNGKey, _) -> jnp.ndarray:
            return self.env.sample_actions(rng)

        def greedy_action(_: chex.PRNGKey, obs: jnp.ndarray) -> jnp.ndarray:
            q_values = self.network.apply(train_state.params, obs)
            return q_values.argmax(axis=-1)

        def take_step(
                carry: tuple[jnp.ndarray, jnp.ndarray, Any, int, PrioritisedTrajectoryBufferState],
                _: None
            ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, Any, int, PrioritisedTrajectoryBufferState], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]]:

            rng, last_obs, env_state, global_step, buffer_state = carry
            rng, sample_rng, action_rng = jax.random.split(rng, 3)
            action = jax.lax.cond(
                jax.random.uniform(sample_rng) < self.hpo_config["epsilon"],
                random_action,
                greedy_action,
                action_rng,
                last_obs
            )

            rng, step_rng = jax.random.split(rng)
            env_state, (obsv, reward, done, info) = self.env.step(env_state, action, step_rng)

            timestep = TimeStep(last_obs=last_obs, obs=obsv, action=action, reward=reward, done=done)
            buffer_state = self.buffer.add(buffer_state, timestep)

            global_step += 1
            return (rng, obsv, env_state, global_step, buffer_state), (
                obsv,
                action,
                reward,
                done,
                info,
            )

        def do_update(
                rng: chex.PRNGKey,
                train_state: DQNTrainState,
                buffer_state: PrioritisedTrajectoryBufferState
            ) -> tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState, DQNMetrics]:
            def gradient_step(
                carry: tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState],
                _: None
            ) -> tuple[tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState], DQNMetrics]:
                rng, train_state, buffer_state = carry
                rng, batch_sample_rng = jax.random.split(rng)
                batch = self.buffer.sample(buffer_state, batch_sample_rng)
                train_state, loss, td_error, grads = self.update(
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

                if not self.track_metrics:
                    loss = None
                    td_error = None
                    grads = None

                return (
                    rng, train_state, buffer_state,
                ), DQNMetrics(loss=loss, td_error=td_error, grads=grads)
            carry, metrics = jax.lax.scan(
                gradient_step,
                (rng, train_state, buffer_state),
                None,
                self.hpo_config["gradient_steps"],
            )
            rng, train_state, buffer_state = carry
            return rng, train_state, buffer_state, metrics

        def dont_update(
                rng: chex.PRNGKey,
                train_state: DQNTrainState,
                buffer_state: PrioritisedTrajectoryBufferState
            ) -> tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState, DQNMetrics]:
            if self.track_metrics:
                loss = jnp.array([((jnp.array([0]) - jnp.array([0])) ** 2).mean()] * self.hpo_config["gradient_steps"])
                td_error = jnp.array(
                    [[[1] * self.hpo_config["buffer_batch_size"]] * self.hpo_config["gradient_steps"]]
                ).mean(axis=0)
                grads = jax.tree_map(lambda x: jnp.stack([x] * self.hpo_config["gradient_steps"]), train_state.params)
            else:
                loss = None
                td_error = None
                grads = None
            return rng, train_state, buffer_state, DQNMetrics(loss=loss, td_error=td_error, grads=grads)

        def target_update(train_state) -> DQNTrainState:
            return train_state.replace(
                target_params=optax.incremental_update(
                    train_state.params, train_state.target_params, self.hpo_config["tau"]
                )
            )

        def dont_target_update(train_state) -> DQNTrainState:
            return train_state

        (rng, last_obs, env_state, global_step, buffer_state), (
            observations,
            action,
            reward,
            done,
            info,
        ) = jax.lax.scan(
            take_step,
            (rng, last_obs, env_state, global_step, buffer_state),
            None,
            self.hpo_config["train_frequency"]
        )

        rng, train_state, buffer_state, metrics = jax.lax.cond(
            (global_step > self.hpo_config["learning_starts"]),
            #& (global_step % self.hpo_config["train_frequency"] == 0),  # todo: is this needed?
            do_update,
            dont_update,
            rng,
            train_state,
            buffer_state,
        )
        train_state = jax.lax.cond(  # todo: move this into the env_step loop?!
            (global_step > self.hpo_config["learning_starts"])
            & (global_step % self.hpo_config["target_network_update_freq"] == 0),
            target_update,
            dont_target_update,
            train_state,
        )
        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            env_state=env_state,
            obs=last_obs,
            global_step=global_step
        )
        tracjectories = None
        if self.track_trajectories:
            tracjectories = Transition(
                obs=observations,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
        return (runner_state, buffer_state), (metrics, tracjectories)

# The SAC Code is heavily based on stable-baselines JAX
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import flashbax as fbx
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from flax.training.train_state import TrainState

from arlbench.core.algorithms.algorithm import Algorithm
from arlbench.core.algorithms.buffers import uniform_sample
from arlbench.core.algorithms.common import TimeStep

from .models import (
    AlphaCoef,
    SACCNNActor,
    SACCNNCritic,
    SACMLPActor,
    SACMLPCritic,
    SACVectorCritic,
)

if TYPE_CHECKING:
    import chex
    from flashbax.buffers.prioritised_trajectory_buffer import (
        PrioritisedTrajectoryBufferState,
    )
    from flax.core.frozen_dict import FrozenDict

    from arlbench.core.environments import Environment
    from arlbench.core.wrappers import AutoRLWrapper

# todo: separate learning rate for critic and actor??


class SACTrainState(TrainState):
    """SAC training state."""

    target_params: None | chex.Array | dict = None
    network_state = None

    @classmethod
    def create_with_opt_state(
        cls, *, apply_fn, params, target_params, tx, opt_state, **kwargs
    ):
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


class SACRunnerState(NamedTuple):
    """SAC runner state. Consists of (rng, actor_train_state, critic_train_state, alpha_train_state, env_state, obs, global_step)."""

    rng: chex.PRNGKey
    actor_train_state: SACTrainState
    critic_train_state: SACTrainState
    alpha_train_state: SACTrainState
    env_state: Any
    obs: chex.Array
    global_step: int


class SACState(NamedTuple):
    """SAC algorithm state. Consists of (runner_state, buffer_state)."""

    runner_state: SACRunnerState
    buffer_state: PrioritisedTrajectoryBufferState


class SACTrainingResult(NamedTuple):
    """SAC training result. Consists of (eval_rewards, trajectories, metrics)."""

    eval_rewards: jnp.ndarray
    trajectories: Transition | None
    metrics: SACMetrics | None


class SACMetrics(NamedTuple):
    """SAC metrics returned by train function. Consists of (actor_loss, critic_loss, alpha_loss, td_error, actor_grads, critic_grads)."""

    actor_loss: jnp.ndarray
    critic_loss: jnp.ndarray
    alpha_loss: jnp.ndarray
    td_error: jnp.ndarray
    actor_grads: FrozenDict
    critic_grads: FrozenDict


class Transition(NamedTuple):
    """SAC Transition. Consists of (done, action, value, reward, obs, info)."""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


SACTrainReturnT = tuple[SACState, SACTrainingResult]


class SAC(Algorithm):
    """JAX-based implementation of Soft-Actor-Critic (SAC)."""

    name: str = "sac"

    def __init__(
        self,
        hpo_config: Configuration,
        env: Environment | AutoRLWrapper,
        eval_env: Environment | AutoRLWrapper | None = None,
        cnn_policy: bool = False,
        nas_config: Configuration | None = None,
        track_metrics: bool = False,
        track_trajectories: bool = False,
    ) -> None:
        """Creates a SAC algorithm instance.

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
            nas_config = SAC.get_default_nas_config()
        super().__init__(
            hpo_config,
            nas_config,
            env,
            eval_env=eval_env,
            track_trajectories=track_trajectories,
            track_metrics=track_metrics,
        )

        action_size, discrete = self.action_type
        if discrete:
            raise ValueError("SAC does not support discrete action spaces.")

        actor_cls = SACCNNActor if cnn_policy else SACMLPActor
        self.actor_network = actor_cls(
            action_size,
            activation=nas_config["activation"],
            hidden_size=nas_config["hidden_size"],
        )
        self.critic_network = SACVectorCritic(
            critic=SACCNNCritic if cnn_policy else SACMLPCritic,
            action_dim=action_size,
            activation=nas_config["activation"],
            hidden_size=nas_config["hidden_size"],
            n_critics=2,
        )
        alpha_init = float(self.hpo_config["alpha"])
        assert alpha_init > 0.0, "The initial value of alpha must be greater than 0"
        self.alpha = AlphaCoef(alpha_init=alpha_init)

        self.buffer = fbx.make_prioritised_flat_buffer(
            max_length=self.hpo_config["buffer_size"],
            min_length=self.hpo_config["buffer_batch_size"],
            sample_batch_size=self.hpo_config["buffer_batch_size"],
            add_sequences=False,
            add_batch_size=self.env.n_envs,
            priority_exponent=self.hpo_config["buffer_beta"],
        )
        if self.hpo_config["buffer_prio_sampling"] is False:
            sample_fn = functools.partial(
                uniform_sample,
                batch_size=self.hpo_config["buffer_batch_size"],
                sequence_length=2,
                period=1,
            )
            self.buffer = self.buffer.replace(sample=sample_fn)

        # target for automatic entropy tuning
        self.target_entropy = -jnp.prod(jnp.array(self.env.action_space.shape)).astype(
            jnp.float32
        )

    @staticmethod
    def get_hpo_config_space(seed: int | None = None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="SACConfigSpace",
            seed=seed,
            space={
                "buffer_size": Integer("buffer_size", (1, int(1e7)), default=int(1e6)),
                "buffer_batch_size": Integer(
                    "buffer_batch_size", (1, 1024), default=256
                ),
                "buffer_prio_sampling": Categorical(
                    "buffer_prio_sampling", [True, False], default=False
                ),
                "buffer_alpha": Float("buffer_alpha", (0.0, 1.0), default=0.9),
                "buffer_beta": Float("buffer_beta", (0.0, 1.0), default=0.9),
                "buffer_epsilon": Float("buffer_epsilon", (0.0, 1e-3), default=1e-5),
                "lr": Float("lr", (1e-5, 0.1), default=3e-4),
                "gradient steps": Integer("gradient steps", (1, int(1e5)), default=1),
                "gamma": Float("gamma", (0.0, 1.0), default=0.99),
                "tau": Float("tau", (0.0, 1.0), default=0.005),
                "use_target_network": Categorical(
                    "use_target_network", [True, False], default=True
                ),
                "train_frequency": Integer("train_frequency", (1, int(1e5)), default=1),
                "learning_starts": Integer(
                    "learning_starts", (1, int(1e5)), default=100
                ),
                "target_network_update_freq": Integer(
                    "target_network_update_freq", (1, int(1e5)), default=1
                ),
                "alpha_auto": Categorical("alpha_auto", [True, False], default=True),
                "alpha": Float("alpha", (0.0, 1.0), default=1.0),
            },
        )

    @staticmethod
    def get_default_hpo_config() -> Configuration:
        return SAC.get_hpo_config_space().get_default_configuration()

    @staticmethod
    def get_nas_config_space(seed: int | None = None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="SACNASConfigSpace",
            seed=seed,
            space={
                "activation": Categorical(
                    "activation", ["tanh", "relu"], default="tanh"
                ),
                "hidden_size": Integer("hidden_size", (1, 1024), default=256),
            },
        )

    @staticmethod
    def get_default_nas_config() -> Configuration:
        return SAC.get_nas_config_space().get_default_configuration()

    @staticmethod
    def get_checkpoint_factory(
        runner_state: SACRunnerState,
        train_result: SACTrainingResult | None,
    ) -> dict[str, Callable]:
        """Creates a factory dictionary of all posssible checkpointing options for SAC.

        Args:
            runner_state (SACRunnerState): Algorithm runner state.
            train_result (SACTrainingResult | None): Training result.

        Returns:
            dict[str, Callable]: Dictionary of factory functions containing:
             - actor_opt_state
             - critic_opt_state
             - alpha_opt_state
             - actor_network_params
             - critic_network_params
             - critic_target_params
             - alpha_network_params
             - actor_loss
             - critic_loss
             - alpha_loss
             - trajectories
        """
        actor_train_state = runner_state.actor_train_state
        critic_train_state = runner_state.critic_train_state
        alpha_train_state = runner_state.alpha_train_state

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
            "actor_opt_state": lambda: actor_train_state.opt_state,
            "critic_opt_state": lambda: critic_train_state.opt_state,
            "alpha_opt_state": lambda: alpha_train_state.opt_state,
            "actor_network_params": lambda: actor_train_state.params,
            "critic_network_params": lambda: critic_train_state.params,
            "critic_target_params": lambda: critic_train_state.target_params,
            "alpha_network_params": lambda: alpha_train_state.params,
            "actor_loss": lambda: train_result.metrics.actor_loss
            if train_result and train_result.metrics
            else None,
            "critic_loss": lambda: train_result.metrics.critic_loss
            if train_result and train_result.metrics
            else None,
            "alpha_loss": lambda: train_result.metrics.alpha_loss
            if train_result and train_result.metrics
            else None,
            "trajectories": get_trajectories,
        }

    def init(
        self,
        rng: chex.PRNGKey,
        buffer_state: PrioritisedTrajectoryBufferState | None = None,
        actor_network_params: FrozenDict | dict | None = None,
        critic_network_params: FrozenDict | dict | None = None,
        critic_target_params: FrozenDict | dict | None = None,
        alpha_network_params: FrozenDict | dict | None = None,
        actor_opt_state: optax.OptState | None = None,
        critic_opt_state: optax.OptState | None = None,
        alpha_opt_state: optax.OptState | None = None,
    ) -> SACState:
        """Initializes SAC state. Passed parameters are not initialized and included in the final state.

        Args:
            actor_network_params (FrozenDict | dict | None, optional): Actor network parameters. Defaults to None.
            critic_network_params (FrozenDict | dict | None, optional): Critic network parameters. Defaults to None.
            critic_target_params (FrozenDict | dict | None, optional): Critic target network parameters. Defaults to None.
            alpha_network_params (FrozenDict | dict | None, optional): Alpha network parameters. Defaults to None.
            actor_opt_state (optax.OptState | None, optional): Actor optimizer state. Defaults to None.
            critic_opt_state (optax.OptState | None, optional): Critic optimizer state. Defaults to None.
            alpha_opt_state (optax.OptState | None, optional): Alpha optimizer state. Defaults to None.

        Returns:
            SACState: SAC state.
        """
        rng, env_rng = jax.random.split(rng)
        env_state, obs = self.env.reset(env_rng)

        if (
            buffer_state is None
            or actor_network_params is None
            or critic_network_params is None
        ):
            dummy_rng = jax.random.PRNGKey(0)
            _action = self.env.sample_actions(dummy_rng)
            _, (_obs, _reward, _done, _) = self.env.step(env_state, _action, dummy_rng)

        if buffer_state is None:
            _timestep = TimeStep(
                last_obs=_obs[0],
                obs=_obs[0],
                action=_action[0],
                reward=_reward[0],
                done=_done[0],
            )
            buffer_state = self.buffer.init(_timestep)

        if actor_network_params is None:
            rng, actor_rng = jax.random.split(rng)
            actor_network_params = self.actor_network.init(actor_rng, _obs)
        if critic_network_params is None:
            rng, critic_rng = jax.random.split(rng)
            critic_network_params = self.critic_network.init(critic_rng, _obs, _action)
        if critic_target_params is None:
            critic_target_params = critic_network_params

        actor_train_state = SACTrainState.create_with_opt_state(
            apply_fn=self.actor_network.apply,
            params=actor_network_params,
            target_params=None,
            tx=optax.adam(
                self.hpo_config["lr"], eps=1e-5
            ),  # todo: change to actor specific lr
            opt_state=actor_opt_state,
        )
        critic_train_state = SACTrainState.create_with_opt_state(
            apply_fn=self.critic_network.apply,
            params=critic_network_params,
            target_params=critic_target_params,
            tx=optax.adam(
                self.hpo_config["lr"], eps=1e-5
            ),  # todo: change to critic specific lr
            opt_state=critic_opt_state,
        )
        if alpha_network_params is None:
            rng, init_rng = jax.random.split(rng)
            alpha_network_params = self.alpha.init(init_rng)
        alpha_train_state = SACTrainState.create_with_opt_state(
            apply_fn=self.alpha.apply,
            params=alpha_network_params,
            target_params=None,
            tx=optax.adam(
                self.hpo_config["lr"], eps=1e-5
            ),  # todo: how to set lr, check with stable-baselines
            opt_state=alpha_opt_state,
        )

        global_step = 0

        runner_state = SACRunnerState(
            rng=rng,
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            alpha_train_state=alpha_train_state,
            env_state=env_state,
            obs=obs,
            global_step=global_step,
        )

        assert buffer_state is not None

        return SACState(runner_state=runner_state, buffer_state=buffer_state)

    @functools.partial(jax.jit, static_argnums=0)
    def predict(
        self,
        runner_state: SACRunnerState,
        obs: jnp.ndarray,
        rng: chex.PRNGKey,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Predict action(s) based on the current observation(s).

        Args:
            runner_state (SACRunnerState): Algorithm runner state.
            obs (jnp.ndarray): Observation(s).
            rng (chex.PRNGKey | None, optional): Not used in DQN. Random generator key in other algorithms. Defaults to None.
            deterministic (bool): Not used in DQN. Return deterministic action in other algorithms. Defaults to True.

        Returns:
            jnp.ndarray: Action(s).
        """
        pi = self.actor_network.apply(runner_state.actor_train_state.params, obs)

        def deterministic_action():
            return pi.mode()

        def sampled_action():
            return pi.sample(seed=rng)

        action = jax.lax.cond(
            deterministic,
            deterministic_action,
            sampled_action,
        )

        # todo: we need to check that the action spaces are finite
        low, high = self.env.action_space.low, self.env.action_space.high
        # check if low or high are none, nan or inf and set to 1
        if low is None or np.isnan(low).any() or np.isinf(low).any():
            low = -jnp.ones_like(action)
        if high is None or np.isnan(high).any() or np.isinf(high).any():
            high = jnp.ones_like(action)
        return low + (action + 1.0) * 0.5 * (high - low)

    @functools.partial(jax.jit, static_argnums=(0, 3, 4, 5), donate_argnums=(2,))
    def train(
        self,
        runner_state: SACRunnerState,
        buffer_state: PrioritisedTrajectoryBufferState,
        n_total_timesteps: int = 1000000,
        n_eval_steps: int = 100,
        n_eval_episodes: int = 10,
    ) -> SACTrainReturnT:
        """Performs one iteration of training.

        Args:
            runner_state (SACTrainReturnT): SAC runner state.
            _ (None): Unused parameter (buffer_state in other algorithms).
            n_total_timesteps (int, optional): Total number of training timesteps. Update steps = n_total_timesteps // n_envs. Defaults to 1000000.
            n_eval_steps (int, optional): Number of evaluation steps during training.
            n_eval_episodes (int, optional): Number of evaluation episodes per evaluation during training.

        Returns:
            SACTrainReturnT: Tuple of PPO algorithm state and training result.
        """

        def train_eval_step(
            carry: tuple[SACRunnerState, PrioritisedTrajectoryBufferState], _: None
        ) -> tuple[
            tuple[SACRunnerState, PrioritisedTrajectoryBufferState], SACTrainingResult
        ]:
            """_summary_

            Args:
                carry (tuple[SACRunnerState, PrioritisedTrajectoryBufferState]): _description_
                _ (None): _description_

            Returns:
                tuple[ tuple[SACRunnerState, PrioritisedTrajectoryBufferState], SACTrainingResult ]: _description_
            """
            runner_state, buffer_state = carry
            (runner_state, buffer_state), (metrics, trajectories) = jax.lax.scan(
                self._update_step,
                (runner_state, buffer_state),
                None,
                np.ceil(n_total_timesteps / self.env.n_envs / self.hpo_config["train_frequency"] / n_eval_steps),
            )
            eval_returns = self.eval(runner_state, n_eval_episodes)

            return (runner_state, buffer_state), SACTrainingResult(
                metrics=metrics, trajectories=trajectories, eval_rewards=eval_returns
            )

        (runner_state, buffer_state), result = jax.lax.scan(
            train_eval_step,
            (runner_state, buffer_state),
            None,
            n_eval_steps,
        )
        return SACState(runner_state=runner_state, buffer_state=buffer_state), result

    def update_critic(
        self,
        actor_train_state: SACTrainState,
        critic_train_state: SACTrainState,
        alpha_train_state: SACTrainState,
        batch: Transition,
        rng: chex.PRNGKey,
    ) -> tuple[SACTrainState, jnp.ndarray, jnp.ndarray, FrozenDict, chex.PRNGKey]:
        """_summary_

        Args:
            actor_train_state (SACTrainState): _description_
            critic_train_state (SACTrainState): _description_
            alpha_train_state (SACTrainState): _description_
            batch (Transition): _description_
            rng (chex.PRNGKey): _description_

        Returns:
            tuple[SACTrainState, jnp.ndarray, jnp.ndarray, FrozenDict, chex.PRNGKey]: _description_
        """
        rng, action_rng = jax.random.split(rng, 2)
        pi = self.actor_network.apply(actor_train_state.params, batch.obs)
        next_state_actions, next_log_prob = pi.sample_and_log_prob(seed=action_rng)

        alpha_value = self.alpha.apply(alpha_train_state.params)

        qf_next_target = self.critic_network.apply(
            critic_train_state.target_params, batch.obs, next_state_actions
        )

        qf_next_target = jnp.min(qf_next_target, axis=0)
        qf_next_target = qf_next_target - alpha_value * next_log_prob
        target_q_value = (
            batch.reward + (1 - batch.done) * self.hpo_config["gamma"] * qf_next_target
        )

        def mse_loss(params: FrozenDict):
            """_summary_

            Args:
                params (FrozenDict): _description_

            Returns:
                _type_: _description_
            """
            q_pred = self.critic_network.apply(params, batch.last_obs, batch.action)
            td_error = target_q_value - q_pred
            return 0.5 * (td_error**2).mean(axis=1).sum(), jnp.abs(td_error)

        (loss_value, td_error), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            critic_train_state.params
        )
        critic_train_state = critic_train_state.apply_gradients(grads=grads)
        return critic_train_state, loss_value, td_error, grads, rng

    def update_actor(
        self,
        actor_train_state: SACTrainState,
        critic_train_state: SACTrainState,
        alpha_train_state: SACTrainState,
        batch: Transition,
        rng: chex.PRNGKey,
    ) -> tuple[SACTrainState, jnp.ndarray, jnp.ndarray, FrozenDict, chex.PRNGKey]:
        """_summary_

        Args:
            actor_train_state (SACTrainState): _description_
            critic_train_state (SACTrainState): _description_
            alpha_train_state (SACTrainState): _description_
            batch (Transition): _description_
            rng (chex.PRNGKey): _description_

        Returns:
            tuple[SACTrainState, jnp.ndarray, jnp.ndarray, FrozenDict, chex.PRNGKey]: _description_
        """
        rng, action_rng = jax.random.split(rng, 2)

        def actor_loss(
            actor_params: FrozenDict,
            critic_params: FrozenDict,
            alpha_params: FrozenDict,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """_summary_

            Args:
                actor_params (FrozenDict): _description_
                critic_params (FrozenDict): _description_
                alpha_params (FrozenDict): _description_

            Returns:
                tuple[jnp.ndarray, jnp.ndarray]: _description_
            """
            pi = self.actor_network.apply(actor_params, batch.last_obs)
            actor_actions, log_prob = pi.sample_and_log_prob(seed=action_rng)

            qf_pi = self.critic_network.apply(
                critic_params, batch.last_obs, actor_actions
            )
            min_qf_pi = jnp.min(qf_pi, axis=0)

            alpha_value = self.alpha.apply(alpha_params)
            actor_loss = (alpha_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_train_state.params,
            critic_train_state.params,
            alpha_train_state.params,
        )
        actor_train_state = actor_train_state.apply_gradients(grads=grads)

        return actor_train_state, loss_value, entropy, grads, rng

    def update_alpha(
        self, alpha_train_state: SACTrainState, entropy: jnp.ndarray
    ) -> tuple[SACTrainState, jnp.ndarray]:
        """_summary_

        Args:
            alpha_train_state (SACTrainState): _description_
            entropy (jnp.ndarray): _description_

        Returns:
            tuple[SACTrainState, jnp.ndarray]: _description_
        """

        def get_alpha_loss(params: FrozenDict) -> jnp.ndarray:
            """_summary_

            Args:
                params (FrozenDict): _description_

            Returns:
                jnp.ndarray: _description_
            """
            alpha_value = self.alpha.apply(params)
            return alpha_value * (entropy - self.target_entropy).mean()  # type: ignore[union-attr]

        alpha_loss, grads = jax.value_and_grad(get_alpha_loss)(alpha_train_state.params)
        alpha_train_state = alpha_train_state.apply_gradients(grads=grads)

        return alpha_train_state, alpha_loss

    def _update_step(
        self, carry: tuple[SACRunnerState, PrioritisedTrajectoryBufferState], _: None
    ) -> tuple[
        tuple[SACRunnerState, PrioritisedTrajectoryBufferState],
        tuple[SACMetrics | None, Transition | None],
    ]:
        """_summary_

        Args:
            carry (tuple[SACRunnerState, PrioritisedTrajectoryBufferState]): _description_
            _ (None): _description_

        Returns:
            tuple[ tuple[SACRunnerState, PrioritisedTrajectoryBufferState], tuple[SACMetrics | None, Transition | None], ]: _description_
        """

        def do_update(
            rng: chex.PRNGKey,
            actor_train_state: SACTrainState,
            critic_train_state: SACTrainState,
            alpha_train_state: SACTrainState,
            buffer_state: PrioritisedTrajectoryBufferState,
        ) -> tuple[
            chex.PRNGKey,
            SACTrainState,
            SACTrainState,
            SACTrainState,
            PrioritisedTrajectoryBufferState,
            SACMetrics,
        ]:
            """_summary_

            Args:
                rng (chex.PRNGKey): _description_
                actor_train_state (SACTrainState): _description_
                critic_train_state (SACTrainState): _description_
                alpha_train_state (SACTrainState): _description_
                buffer_state (PrioritisedTrajectoryBufferState): _description_

            Returns:
                tuple[ chex.PRNGKey, SACTrainState, SACTrainState, SACTrainState, PrioritisedTrajectoryBufferState, SACMetrics, ]: _description_
            """

            def gradient_step(
                carry: tuple[
                    chex.PRNGKey,
                    SACTrainState,
                    SACTrainState,
                    SACTrainState,
                    PrioritisedTrajectoryBufferState,
                ],
                _: None,
            ) -> tuple[
                tuple[
                    chex.PRNGKey,
                    SACTrainState,
                    SACTrainState,
                    SACTrainState,
                    PrioritisedTrajectoryBufferState,
                ],
                SACMetrics,
            ]:
                """_summary_

                Args:
                    carry (tuple[ chex.PRNGKey, SACTrainState, SACTrainState, SACTrainState, PrioritisedTrajectoryBufferState, ]): _description_
                    _ (None): _description_

                Returns:
                    tuple[ tuple[ chex.PRNGKey, SACTrainState, SACTrainState, SACTrainState, PrioritisedTrajectoryBufferState, ], SACMetrics, ]: _description_
                """
                (
                    rng,
                    actor_train_state,
                    critic_train_state,
                    alpha_train_state,
                    buffer_state,
                ) = carry
                rng, batch_sample_rng = jax.random.split(rng)
                batch = self.buffer.sample(buffer_state, batch_sample_rng)
                critic_train_state, critic_loss, td_error, critic_grads, rng = (
                    self.update_critic(
                        actor_train_state,
                        critic_train_state,
                        alpha_train_state,
                        batch.experience.first,
                        rng,
                    )
                )
                actor_train_state, actor_loss, entropy, actor_grads, rng = (
                    self.update_actor(
                        actor_train_state,
                        critic_train_state,
                        alpha_train_state,
                        batch.experience.first,
                        rng,
                    )
                )
                alpha_train_state, alpha_loss = self.update_alpha(
                    alpha_train_state, entropy
                )
                new_prios = jnp.power(
                    td_error.mean(axis=0) + self.hpo_config["buffer_epsilon"],
                    self.hpo_config["buffer_alpha"],
                )
                buffer_state = self.buffer.set_priorities(
                    buffer_state, batch.indices, new_prios
                )
                metrics = SACMetrics(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    alpha_loss=alpha_loss,
                    td_error=td_error.mean(axis=0),
                    actor_grads=actor_grads,
                    critic_grads=critic_grads,
                )
                return (
                    rng,
                    actor_train_state,
                    critic_train_state,
                    alpha_train_state,
                    buffer_state,
                ), metrics

            carry, metrics = jax.lax.scan(
                gradient_step,
                (
                    rng,
                    actor_train_state,
                    critic_train_state,
                    alpha_train_state,
                    buffer_state,
                ),
                None,
                self.hpo_config["gradient steps"],
            )
            (
                rng,
                actor_train_state,
                critic_train_state,
                alpha_train_state,
                buffer_state,
            ) = carry
            return (
                rng,
                actor_train_state,
                critic_train_state,
                alpha_train_state,
                buffer_state,
                metrics,
            )

        def dont_update(
            rng: chex.PRNGKey,
            actor_train_state: SACTrainState,
            critic_train_state: SACTrainState,
            alpha_train_state: SACTrainState,
            buffer_state: PrioritisedTrajectoryBufferState,
        ) -> tuple[
            chex.PRNGKey,
            SACTrainState,
            SACTrainState,
            SACTrainState,
            PrioritisedTrajectoryBufferState,
            SACMetrics,
        ]:
            """_summary_

            Args:
                rng (chex.PRNGKey): _description_
                actor_train_state (SACTrainState): _description_
                critic_train_state (SACTrainState): _description_
                alpha_train_state (SACTrainState): _description_
                buffer_state (PrioritisedTrajectoryBufferState): _description_

            Returns:
                tuple[ chex.PRNGKey, SACTrainState, SACTrainState, SACTrainState, PrioritisedTrajectoryBufferState, SACMetrics, ]: _description_
            """
            single_loss = jnp.array(
                [((jnp.array([0]) - jnp.array([0])) ** 2).mean()]
                * self.hpo_config["gradient steps"]
            )
            td_error = jnp.array(
                [
                    [[0] * self.hpo_config["buffer_batch_size"]]
                    * self.hpo_config["gradient steps"]
                ]
            ).mean(axis=0)
            actor_grads = jax.tree_map(
                lambda x: jnp.stack([x] * self.hpo_config["gradient steps"]),
                actor_train_state.params,
            )
            critic_grads = jax.tree_map(
                lambda x: jnp.stack([x] * self.hpo_config["gradient steps"]),
                critic_train_state.params,
            )
            metrics = SACMetrics(
                actor_loss=single_loss,
                critic_loss=single_loss,
                alpha_loss=single_loss,
                td_error=td_error,
                actor_grads=actor_grads,
                critic_grads=critic_grads,
            )
            return (
                rng,
                actor_train_state,
                critic_train_state,
                alpha_train_state,
                buffer_state,
                metrics,
            )

        runner_state, buffer_state = carry
        (
            (runner_state, buffer_state),
            (
                done,
                action,
                value,
                reward,
                last_obs,
                info,
            ),
        ) = jax.lax.scan(
            self._env_step,
            (runner_state, buffer_state),
            None,
            self.hpo_config["train_frequency"],
        )
        (
            rng,
            actor_train_state,
            critic_train_state,
            alpha_train_state,
            _,
            _,
            global_step,
        ) = runner_state
        rng, _rng = jax.random.split(rng)

        (
            rng,
            actor_train_state,
            critic_train_state,
            alpha_train_state,
            buffer_state,
            step_metrics,
        ) = jax.lax.cond(
            (global_step > self.hpo_config["learning_starts"])
            & (global_step % self.hpo_config["train_frequency"] == 0),
            do_update,
            dont_update,
            rng,
            actor_train_state,
            critic_train_state,
            alpha_train_state,
            buffer_state,
        )
        runner_state = SACRunnerState(
            rng=rng,
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            alpha_train_state=alpha_train_state,
            env_state=runner_state.env_state,
            obs=runner_state.obs,
            global_step=runner_state.global_step,
        )
        actor_loss, critic_loss, alpha_loss, td_error, actor_grads, critic_grads = (
            step_metrics
        )
        metrics, tracjectories = None, None
        if self.track_metrics:
            metrics = SACMetrics(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                alpha_loss=alpha_loss,
                actor_grads=actor_grads,
                critic_grads=critic_grads,
                td_error=td_error,
            )
        if self.track_trajectories:
            tracjectories = Transition(
                obs=last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                info=info,
            )
        return (runner_state, buffer_state), (metrics, tracjectories)

    @functools.partial(jax.jit, static_argnums=0)
    def _env_step(
        self, carry: tuple[SACRunnerState, PrioritisedTrajectoryBufferState], _: None
    ) -> tuple[tuple[SACRunnerState, PrioritisedTrajectoryBufferState], Transition]:
        """_summary_

        Args:
            carry (tuple[SACRunnerState, PrioritisedTrajectoryBufferState]): _description_
            _ (None): _description_

        Returns:
            tuple[tuple[SACRunnerState, PrioritisedTrajectoryBufferState], Transition]: _description_
        """
        runner_state, buffer_state = carry
        (
            rng,
            actor_train_state,
            critic_train_state,
            alpha_train_state,
            env_state,
            last_obs,
            global_step,
        ) = runner_state

        # Select action(s)
        rng, _rng = jax.random.split(rng)
        pi = self.actor_network.apply(actor_train_state.params, last_obs)

        buffer_action = pi.sample(seed=_rng)
        low, high = self.env.action_space.low, self.env.action_space.high
        if low is None or np.isnan(low).any() or np.isinf(low).any():
            low = -jnp.ones_like(buffer_action)
        if high is None or np.isnan(high).any() or np.isinf(high).any():
            high = jnp.ones_like(buffer_action)
        action = low + (buffer_action + 1.0) * 0.5 * (high - low)

        # Perform environment step
        rng, _rng = jax.random.split(rng)
        env_state, (obsv, reward, done, info) = self.env.step(env_state, action, _rng)

        timestep = TimeStep(
            last_obs=last_obs, obs=obsv, action=buffer_action, reward=reward, done=done
        )
        buffer_state = self.buffer.add(buffer_state, timestep)

        global_step += 1

        def target_update(train_state) -> SACTrainState:
            """_summary_

            Args:
                train_state (_type_): _description_

            Returns:
                SACTrainState: _description_
            """
            return train_state.replace(
                target_params=optax.incremental_update(
                    train_state.params,
                    train_state.target_params,
                    self.hpo_config["tau"],
                )
            )

        def dont_target_update(train_state) -> SACTrainState:
            """_summary_

            Args:
                train_state (_type_): _description_

            Returns:
                SACTrainState: _description_
            """
            return train_state

        critic_train_state = jax.lax.cond(  # todo: move this into the env_step loop?!
            (global_step > np.ceil(self.hpo_config["learning_starts"] / self.env.n_envs))
            & (global_step % np.ceil(self.hpo_config["target_network_update_freq"] / self.env.n_envs) == 0),
            target_update,
            dont_target_update,
            critic_train_state,
        )

        value = jnp.zeros_like(reward)
        transition = Transition(done, action, value, reward, last_obs, info)
        runner_state = SACRunnerState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            alpha_train_state=alpha_train_state,
            env_state=env_state,
            obs=obsv,
            rng=rng,
            global_step=global_step,
        )
        return (runner_state, buffer_state), transition

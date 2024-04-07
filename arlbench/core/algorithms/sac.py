# The SAC Code is heavily based on stable-baselines JAX
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple

import flashbax as fbx
import jax
import jax.lax
import jax.numpy as jnp
import optax
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from flax.training.train_state import TrainState

from .algorithm import Algorithm
from .common import TimeStep
from .models import AlphaCoef, SACActor, SACVectorCritic

if TYPE_CHECKING:
    import chex

# todo: separate learning rate for critic and actor??

class SACRunnerState(NamedTuple):
    rng: chex.PRNGKey
    actor_train_state: Any
    critic_train_state: Any
    alpha_train_state: Any
    env_state: Any
    obs: chex.Array
    global_step: int


class SACTrainState(TrainState):
    target_params: None | chex.Array | dict = None
    network_state = None

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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class SAC(Algorithm):
    def __init__(
            self,
            hpo_config: Configuration | dict,
            options: dict,
            env: Any,
            nas_config: Configuration | dict = None,
            track_metrics=False,
            track_trajectories=False,
    ) -> None:
        if nas_config is None:
            nas_config = SAC.get_default_nas_config()
        super().__init__(
            hpo_config,
            nas_config,
            options,
            env,
            track_trajectories=track_trajectories,
            track_metrics=track_metrics
        )

        action_size, discrete = self.action_type
        self.actor_network = SACActor(
            action_size,
            activation=nas_config["activation"],
            hidden_size=nas_config["hidden_size"],
        )
        self.critic_network = SACVectorCritic(
            action_size,
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
            add_batch_size=self.env_options["n_envs"],
            priority_exponent=self.hpo_config["buffer_beta"],
        )

    @staticmethod
    def get_hpo_config_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="SACConfigSpace",
            seed=seed,
            space={
                "buffer_size": Integer("buffer_size", (1, int(1e7)), default=int(1e6)),
                "buffer_batch_size": Integer("buffer_batch_size", (1, 1024), default=256),
                "buffer_alpha": Float("buffer_alpha", (0., 1.), default=0.9),
                "buffer_beta": Float("buffer_beta", (0., 1.), default=0.9),
                "buffer_epsilon": Float("buffer_epsilon", (0., 1e-3), default=1e-5),
                "lr": Float("lr", (1e-5, 0.1), default=3e-4),
                "gradient steps": Integer("gradient steps", (1, int(1e5)), default=1),
                "policy_delay": Integer("policy_delay", (1, int(1e5)), default=1),
                "gamma": Float("gamma", (0., 1.), default=0.99),
                "tau": Float("tau", (0., 1.), default=0.005),
                "use_target_network": Categorical("use_target_network", [True, False], default=True),
                "train_frequency": Integer("train_frequency", (1, int(1e5)), default=1),
                "learning_starts": Integer("learning_starts", (1, int(1e5)), default=100),
                "target_network_update_freq": Integer("target_network_update_freq", (1, int(1e5)), default=1),
                "alpha_auto": Categorical("alpha_auto", [True, False], default=True),
                "alpha": Float("alpha", (0., 1.), default=1.0),
            },
        )

    @staticmethod
    def get_default_hpo_config() -> Configuration:
        return SAC.get_hpo_config_space().get_default_configuration()

    @staticmethod
    def get_nas_config_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="SACNASConfigSpace",
            seed=seed,
            space={
                "activation": Categorical("activation", ["tanh", "relu"], default="tanh"),
                "hidden_size": Integer("hidden_size", (1, 1024), default=256),
            },
        )

    @staticmethod
    def get_default_nas_config() -> Configuration:
        return SAC.get_nas_config_space().get_default_configuration()

    def init(
            self, rng, actor_network_params=None, critic_network_params=None, critic_target_params=None
    ) -> tuple[SACRunnerState, Any]:
        rng, env_rng = jax.random.split(rng)
        env_state, obs = self.env.reset(env_rng)


        dummy_rng = jax.random.PRNGKey(0)
        _action = self.env.sample_actions(dummy_rng)
        _, (_obs, _reward, _done, _) = self.env.step(env_state, _action, dummy_rng)

        _timestep = TimeStep(last_obs=_obs[0], obs=_obs[0], action=_action[0], reward=_reward[0], done=_done[0])
        buffer_state = self.buffer.init(_timestep)

        _, _rng = jax.random.split(rng)
        if actor_network_params is None:
            actor_network_params = self.actor_network.init(_rng, _obs)
        if critic_network_params is None:
            critic_network_params = self.critic_network.init(_rng, _obs, _action)
        if critic_target_params is None:
            critic_target_params = critic_network_params

        actor_train_state = SACTrainState.create_with_opt_state(
            apply_fn=self.actor_network.apply,
            params=actor_network_params,
            target_params=None,
            tx=optax.adam(self.hpo_config["lr"], eps=1e-5),  # todo: change to actor specific lr
            opt_state=None,
        )
        critic_train_state = SACTrainState.create_with_opt_state(
            apply_fn=self.critic_network.apply,
            params=critic_network_params,
            target_params=critic_target_params,
            tx=optax.adam(self.hpo_config["lr"], eps=1e-5),  # todo: change to critic specific lr
            opt_state=None,
        )
        _, _rng = jax.random.split(rng)
        alpha_train_state = SACTrainState.create_with_opt_state(
            apply_fn=self.alpha.apply,
            params=self.alpha.init(_rng),
            target_params=None,
            tx=optax.adam(self.hpo_config["lr"], eps=1e-5),  # todo: how to set lr, check with stable-baselines
            opt_state=None,
        )
        # target for automatic entropy tuning
        self.target_entropy = -jnp.prod(jnp.array(self.env.action_space.shape)).astype(jnp.float32)

        rng, _rng = jax.random.split(rng)
        global_step = 0

        runner_state = SACRunnerState(
            rng=rng,
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            alpha_train_state=alpha_train_state,
            env_state=env_state,
            obs=obs,
            global_step=global_step
        )

        return runner_state, buffer_state


    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, runner_state, obs, rng) -> int:
        pi = self.actor_network.apply(runner_state.actor_train_state.params, obs)
        action = pi.mode()
        # todo: we need to check that the action spaces are finite
        low, high = self.env.action_space.low, self.env.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)


    @functools.partial(jax.jit, static_argnums=0, donate_argnums=(2,))
    def train(
            self,
            runner_state,
            buffer_state
    )-> tuple[tuple[SACRunnerState, Any], tuple | None]:
        def train_eval_step(carry, _):
            runner_state, buffer_state = carry
            (runner_state, buffer_state), out = jax.lax.scan(
                self._update_step,
                (runner_state, buffer_state),
                None,
                ((self.env_options["n_total_timesteps"]//self.hpo_config["train_frequency"])//self.env_options["n_envs"])//self.env_options["n_eval_steps"],
            )
            reward = self.eval(runner_state, self.env_options["n_eval_episodes"])
            jax.debug.print("Reward: {reward}", reward=reward.mean())

            return (runner_state, buffer_state), (reward.mean(), out)

        (runner_state, buffer_state), (reward, out) = jax.lax.scan(
            train_eval_step,
            (runner_state, buffer_state),
            None,
            self.env_options["n_eval_steps"],
        )
        return (runner_state, buffer_state), (reward, out)

    def update_critic(self, actor_train_state, critic_train_state, alpha_train_state, batch, rng):
        rng, action_rng = jax.random.split(rng, 2)
        pi = self.actor_network.apply(actor_train_state.params, batch.obs)
        next_state_actions, next_log_prob = pi.sample_and_log_prob(seed=action_rng)

        alpha_value = self.alpha.apply(alpha_train_state.params)

        qf_next_target = self.critic_network.apply(
            critic_train_state.target_params, batch.obs, next_state_actions
        )

        qf_next_target = jnp.min(qf_next_target, axis=0)
        qf_next_target = qf_next_target - alpha_value * next_log_prob
        target_q_value = batch.reward + (1 - batch.done) * self.hpo_config["gamma"] * qf_next_target

        def mse_loss(params):
            q_pred = self.critic_network.apply(params, batch.last_obs, batch.action)
            td_error = target_q_value - q_pred
            return 0.5 * (td_error ** 2).mean(axis=1).sum(), jnp.abs(td_error)

        (loss_value, td_error), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            critic_train_state.params
        )
        critic_train_state = critic_train_state.apply_gradients(grads=grads)
        return critic_train_state, loss_value, td_error, grads, rng,

    def update_actor(self, actor_train_state, critic_train_state, alpha_train_state, batch, rng):
        rng, action_rng = jax.random.split(rng, 2)

        def actor_loss(actor_params, critic_params, alpha_params):
            pi = self.actor_network.apply(actor_params, batch.last_obs)
            actor_actions, log_prob = pi.sample_and_log_prob(seed=action_rng)

            qf_pi = self.critic_network.apply(critic_params,batch.last_obs, actor_actions)
            min_qf_pi = jnp.min(qf_pi, axis=0)

            alpha_value = self.alpha.apply(alpha_params)
            actor_loss = (alpha_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_train_state.params, critic_train_state.params, alpha_train_state.params
        )
        actor_train_state = actor_train_state.apply_gradients(grads=grads)

        return actor_train_state, loss_value, entropy, grads, rng

    def update_alpha(self, alpha_train_state, entropy):
        def alpha_loss(params):
            alpha_value = self.alpha.apply(params)
            return alpha_value * (entropy - self.target_entropy).mean()  # type: ignore[union-attr]

        alpha_loss, grads = jax.value_and_grad(alpha_loss)(alpha_train_state.params)
        alpha_train_state = alpha_train_state.apply_gradients(grads=grads)

        return alpha_train_state, alpha_loss

    def _update_step(
            self,
            carry,
            _
    ):
        def do_update(rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state):
            def gradient_step(carry, _):
                rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state = carry
                rng, batch_sample_rng = jax.random.split(rng)
                batch = self.buffer.sample(buffer_state, batch_sample_rng)
                critic_train_state, critic_loss, td_error, critic_grads, rng = self.update_critic(
                    actor_train_state,
                    critic_train_state,
                    alpha_train_state,
                    batch.experience.first,
                    rng,
                )
                # todo: consider policy_delay here!?
                actor_train_state, actor_loss, entropy, actor_grads, rng = self.update_actor(
                    actor_train_state,
                    critic_train_state,
                    alpha_train_state,
                    batch.experience.first,
                    rng,
                )
                alpha_train_state, alpha_loss = self.update_alpha(alpha_train_state, entropy)
                new_prios = jnp.power(
                    td_error.mean(axis=0) + self.hpo_config["buffer_epsilon"], self.hpo_config["buffer_alpha"]
                )
                buffer_state = self.buffer.set_priorities(buffer_state, batch.indices, new_prios)
                return (
                    rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state
                ), (
                    critic_loss, actor_loss, alpha_loss, td_error.mean(axis=0), actor_grads, critic_grads,
                )

            carry, metrics = jax.lax.scan(
                gradient_step,
                (rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state),
                None,
                self.hpo_config["gradient steps"],
            )
            rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state = carry
            return rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state, metrics

        def dont_update(rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state):
            single_loss = jnp.array([((jnp.array([0]) - jnp.array([0])) ** 2).mean()] * self.hpo_config["gradient steps"])
            td_error = jnp.array(
                [[[0] * self.hpo_config["buffer_batch_size"]] * self.hpo_config["gradient steps"]]
            ).mean(axis=0)
            actor_grads = jax.tree_map(lambda x: jnp.stack([x] * self.hpo_config["gradient steps"]), actor_train_state.params)
            critic_grads = jax.tree_map(lambda x: jnp.stack([x] * self.hpo_config["gradient steps"]), critic_train_state.params)
            metrics = (single_loss, single_loss, single_loss, td_error, actor_grads, critic_grads)
            return rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state, metrics

        # soft update
        def target_update():
            return critic_train_state.replace(
                target_params=optax.incremental_update(
                    critic_train_state.params, critic_train_state.target_params, self.hpo_config["tau"]
                )
            )
        def dont_target_update():
            return critic_train_state

        runner_state, buffer_state = carry
        (runner_state, buffer_state), (
            done,
            action,
            value,
            reward,
            last_obs,
            info,
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
            env_state,
            last_obs,
            global_step
        ) = runner_state
        rng, _rng = jax.random.split(rng)

        rng, actor_train_state, critic_train_state, alpha_train_state, buffer_state, metrics = jax.lax.cond(
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
        critic_train_state = jax.lax.cond(
            (global_step > self.hpo_config["learning_starts"])
            & (global_step % self.hpo_config["target_network_update_freq"] == 0),
            target_update,
            dont_target_update,
            )
        runner_state = SACRunnerState(
            rng=rng,
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            alpha_train_state=alpha_train_state,
            env_state=runner_state.env_state,
            obs=runner_state.obs,
            global_step=runner_state.global_step
        )
        actor_loss, critic_loss, alpha_loss, td_error, actor_grads, critic_grads = metrics
        if self.track_trajectories:
            metric = (
                (actor_loss, critic_loss, alpha_loss),
                (actor_grads, critic_grads),
                Transition(
                    obs=last_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                ),
                {"td_error": [td_error]},
            )
        elif self.track_metrics:
            metric = (
                (actor_loss, critic_loss, alpha_loss),
                (actor_grads, critic_grads),
                {"td_error": [td_error]},
            )
        else:
            metric = None
        return (runner_state, buffer_state), metric

    @functools.partial(jax.jit, static_argnums=0)
    def _env_step(self, carry, _):
        runner_state, buffer_state = carry
        (
            rng,
            actor_train_state,
            critic_train_state,
            alpha_train_state,
            env_state,
            last_obs,
            global_step
        ) = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi = self.actor_network.apply(actor_train_state.params, last_obs)
        buffer_action = pi.sample(seed=_rng)
        low, high = self.env.action_space.low, self.env.action_space.high
        action = low + (buffer_action + 1.0) * 0.5 * (high - low)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        env_state, (obsv, reward, done, info) = self.env.step(env_state, action, _rng)

        timestep = TimeStep(last_obs=last_obs, obs=obsv, action=buffer_action, reward=reward, done=done)
        buffer_state = self.buffer.add(buffer_state, timestep)

        value = jnp.zeros_like(reward)
        transition = Transition(
            done, action, value, reward, last_obs, info
        )
        runner_state = SACRunnerState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            alpha_train_state=alpha_train_state,
            env_state=env_state,
            obs=obsv,
            rng=rng,
            global_step=global_step + self.env_options["n_envs"]
        )
        return (runner_state, buffer_state), transition

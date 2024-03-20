# The DQN Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import chex
import optax
from .common import TimeStep
from flax.training.train_state import TrainState
from typing import NamedTuple, Union
from typing import Any, Dict, Optional
import chex
import jax.lax
import flashbax as fbx
from .agent import Agent
import functools
from .models import Q
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical


class DQNRunnerState(NamedTuple):
    rng: chex.PRNGKey
    train_state: Any
    env_state: Any
    obs: chex.Array
    global_step: int


class DQNTrainState(TrainState):
    target_params: Union[None, chex.Array, dict] = None
    opt_state = None

    @classmethod
    def create_with_opt_state(cls, *, apply_fn, params, target_params, tx, opt_state, **kwargs):
        if opt_state is None:
            opt_state = tx.init(params)
        obj = cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
        return obj


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    q_pred: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class DQN(Agent):
    def __init__(
        self,
        config: Union[Configuration, Dict],
        options: Dict,
        env: Any,
        env_params: Any,
        track_trajectories=False,
        track_metrics=False
    ) -> None:
        super().__init__(
            config,
            options,
            env,
            env_params,
            track_trajectories=track_trajectories,
            track_metrics=track_metrics
        )

        action_size, discrete = self.action_type
        self.network = Q(
            action_size,
            discrete=discrete,
            activation=self.config["activation"],
            hidden_size=self.config["hidden_size"],
        )

        self.buffer = fbx.make_prioritised_flat_buffer(
            max_length=self.config["buffer_size"],
            min_length=self.config["buffer_batch_size"],
            sample_batch_size=self.config["buffer_batch_size"],
            add_sequences=False,
            add_batch_size=self.env_options["n_envs"],
            priority_exponent=self.config["buffer_beta"]    
        )

    @staticmethod
    def get_configuration_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="PPOConfigSpace",
            seed=seed,
            space={
                "buffer_size": Integer("buffer_size", (1, int(1e10)), default=int(1e6)),
                "buffer_batch_size": Integer("buffer_batch_size", (1, 1024), default=64),
                "buffer_alpha": Float("buffer_alpha", (0., 1.), default=0.9),
                "buffer_beta": Float("buffer_beta", (0., 1.), default=0.9),
                "buffer_epsilon": Float("buffer_epsilon", (0., 1.), default=0.9),
                "lr": Float("lr", (1e-5, 0.1), default=2.5e-4),
                "update_epochs": Integer("update_epochs", (1, int(1e5)), default=10),
                # 0 = tanh, 1 = relu, see agents.models.ACTIVATIONS
                "activation": Categorical("activation", [0, 1], default=0),
                "hidden_size": Integer("hidden_size", (1, 1024), default=64),
                "gamma": Float("gamma", (0., 1.), default=0.99),
                "tau": Float("tau", (0., 1.), default=1.0),
                "epsilon": Float("epsilon", (0., 1.), default=0.1),
                "use_target_network": Categorical("use_target_network", [True, False], default=True),
                "train_frequency": Integer("train_frequency", (1, int(1e5)), default=4),
                "learning_starts": Integer("learning_starts", (1, int(1e5)), default=10000),
                "target_network_update_freq": Integer("target_network_update_freq", (1, int(1e5)), default=100)
            },
        )

    @staticmethod
    def get_default_configuration() -> Configuration:
        return DQN.get_configuration_space().get_default_configuration()

    def init(self, rng, network_params=None, target_params=None) -> tuple[DQNRunnerState, Any]:
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.env_options["n_envs"])

        last_obsv, last_env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_rng, self.env_params
        )
        
        dummy_rng = jax.random.PRNGKey(0) 
        _action = self.env.action_space().sample(dummy_rng)
        _, _env_state = self.env.reset(rng, self.env_params)
        _obs, _, _reward, _done, _ = self.env.step(rng, _env_state, _action, self.env_params)

        _timestep = TimeStep(last_obs=_obs, obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = self.buffer.init(_timestep)

        _, _rng = jax.random.split(rng)
        if network_params is None:
            network_params = self.network.init(_rng, _obs)
        if target_params is None:
            target_params = self.network.init(_rng, _obs)
        opt_state = None

        train_state_kwargs = {
            "apply_fn": self.network.apply,
            "params": network_params,
            "target_params": target_params,
            "tx": optax.adam(self.config["lr"], eps=1e-5),
            "opt_state": opt_state,
        }
        train_state = DQNTrainState.create_with_opt_state(**train_state_kwargs)

        rng, _rng = jax.random.split(rng)
        global_step = 0

        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            env_state=last_env_state,
            obs=last_obsv,
            global_step=global_step
        )    

        return runner_state, buffer_state

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, network_params, obsv, _) -> int:
        q_values = self.network.apply(network_params, obsv)
        return q_values.argmax(axis=-1)

    @functools.partial(jax.jit, static_argnums=0, donate_argnums=(2,))
    def train(
        self,
        runner_state,
        buffer_state
    )-> tuple[tuple[DQNRunnerState, Any], Optional[tuple]]:
        (runner_state, buffer_state), out = jax.lax.scan(
            self._update_step, (runner_state, buffer_state), None, (self.env_options["n_total_timesteps"]//self.config["train_frequency"])//self.env_options["n_envs"]
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
        if self.config["use_target_network"]:
            q_next_target = self.network.apply(
                train_state.target_params, next_observations
            )  # (batch_size, num_actions)
        else:
            q_next_target = self.network.apply(
                train_state.params, next_observations
            )  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * self.config["gamma"] * q_next_target

        def mse_loss(params):
            q_pred = self.network.apply(
                params, observations
            )  # (batch_size, num_actions)
            q_pred = q_pred[
                jnp.arange(q_pred.shape[0]), actions.squeeze().astype(int)
            ]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss_value, q_pred, grads, train_state.opt_state

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
            return jnp.array(
                [
                    self.env.action_space(self.env_params).sample(rng)
                    for _ in range(self.env_options["n_envs"])
                ]
            )

        def greedy_action():
            q_values = self.network.apply(train_state.params, last_obs)
            action = q_values.argmax(axis=-1)
            return action

        def take_step(carry, _):
            obsv, env_state, global_step, buffer_state = carry
            action = jax.lax.cond(
                jax.random.uniform(rng) < self.config["epsilon"],
                random_action,
                greedy_action,
            )

            rng_step = jax.random.split(_rng, self.env_options["n_envs"])
            obsv, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, self.env_params)

            def no_target_td(train_state):
                return self.network.apply(train_state.params, obsv).argmax(axis=-1)

            def target_td(train_state):
                return self.network.apply(train_state.target_params, obsv).argmax(
                    axis=-1
                )

            q_next_target = jax.lax.cond(
                self.config["use_target_network"], target_td, no_target_td, train_state
            )

            # td_error = (
            #     reward
            #     + (1 - done) * self.config["gamma"] * jnp.expand_dims(q_next_target, -1)    # why expand dims?
            #     - self.network.apply(train_state.params, last_obs).take(action)
            # )
            td_error = (
                reward
                + (1 - done) * self.config["gamma"] * q_next_target
                - self.network.apply(train_state.params, last_obs).take(action)
            )
            transition_weight = jnp.power(
                jnp.abs(td_error) + self.config["buffer_epsilon"], self.config["buffer_alpha"]
            )
            timestep = TimeStep(last_obs=last_obs, obs=obsv, action=action, reward=reward, done=done)
            buffer_state = self.buffer.add(buffer_state, timestep)

            # compute indices of newly added buffer elements
            added_indices = jnp.arange(
                0,
                len(obsv)
            ) + buffer_state.current_index
            buffer_state = self.buffer.set_priorities(buffer_state, added_indices, transition_weight)
            
            # global_step += 1
            global_step += self.env_options["n_envs"]
            return (obsv, env_state, global_step, buffer_state), (
                obsv,
                action,
                reward,
                done,
                info,
                td_error,
            )

        def do_update(train_state, buffer_state):
            # batch = buffer.sample_fn(buffer_state, rng, config["batch_size"])
            batch = self.buffer.sample(buffer_state, rng)
            batch = self.buffer.sample(buffer_state, rng).experience.first
            train_state, loss, q_pred, grads, opt_state = self.update(
                train_state,
                batch.last_obs,
                batch.action,
                batch.obs,
                batch.reward,
                batch.done,
            )
            return train_state, loss, q_pred, grads, opt_state

        def dont_update(train_state, _):
            return (
                train_state,
                ((jnp.array([0]) - jnp.array([0])) ** 2).mean(),
                jnp.ones(self.config["buffer_batch_size"]),
                train_state.params,
                train_state.opt_state
            )

        def target_update():
            return train_state.replace(
                target_params=optax.incremental_update(
                    train_state.params, train_state.target_params, self.config["tau"]
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
            td_error,
        ) = jax.lax.scan(
            take_step,
            (last_obs, env_state, global_step, buffer_state),
            None,
            self.config["train_frequency"],
        )

        train_state, loss, q_pred, grads, opt_state = jax.lax.cond(
            (global_step > self.config["learning_starts"])
            & (global_step % self.config["train_frequency"] == 0),
            do_update,
            dont_update,
            train_state,
            buffer_state,
        )
        train_state = jax.lax.cond(
            (global_step > self.config["learning_starts"])
            & (global_step % self.config["target_network_update_freq"] == 0),
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
                    q_pred=[q_pred],
                ),
                {"td_error": [td_error]},
            )
        elif self.track_metrics:
            metric = (
                loss,
                grads,
                {"q_pred": [q_pred], "td_error": [td_error]},
            )
        else:
            metric = None
        return (runner_state, buffer_state), metric
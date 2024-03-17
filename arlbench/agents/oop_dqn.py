# The DQN Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import chex
import optax
from .common import ExtendedTrainState
from typing import NamedTuple
import dejax.utils as utils
from typing import Callable, Any, Tuple
import gymnax
import chex
import jax.lax
import flashbax as fbx
from .abstract_agent import Agent
import functools


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
        config,
        env,
        env_params,
        network,
        num_updates,
    ) -> None:
        super().__init__(env, env_params)

        self.config = config
        self.network = network
        self.num_updates = num_updates

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, network_params, obsv, _) -> int:
        q_values = self.network.apply(network_params, obsv)
        return q_values.argmax(axis=-1)

    @functools.partial(jax.jit, static_argnums=0)
    def train(
        self,
        rng,
        network_params,
        target_params,
        opt_state,
        obsv,
        env_state,
        buffer_state,
        global_step
    ):
        train_state_kwargs = {
            "apply_fn": self.network.apply,
            "params": network_params,
            "tx": optax.adam(self.config["lr"], eps=1e-5),
            "opt_state": opt_state,
        }
        train_state_kwargs["target_params"] = target_params
        train_state = ExtendedTrainState.create_with_opt_state(**train_state_kwargs)
        buffer = fbx.make_prioritised_flat_buffer(
            max_length=int(self.config["buffer_size"]),
            min_length=self.config["batch_size"],
            sample_batch_size=self.config["batch_size"],
            priority_exponent=self.config["beta"]
        )

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, buffer_state, global_step)
        runner_state, out = jax.lax.scan(
            self._update_step, runner_state, None, (self.config["total_timesteps"]//self.config["train_frequency"])//self.config["num_envs"]
        )
        return runner_state, out
    
    @functools.partial(jax.jit, static_argnums=0)
    def update(
            self, train_state, observations, actions, next_observations, rewards, dones
        ):
            if self.config["target"]:
                q_next_target = self.network.apply(
                    train_state.target_params, next_observations
                )  # (batch_size, num_actions)
            else:
                q_next_target = self.network.apply(
                    train_state.params, next_observations
                )  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            next_q_value = rewards + (1 - dones) * self.config["gamma"] * q_next_target

            (loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(
                train_state.params, observations, actions, next_q_value
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss_value, q_pred, grads, train_state.opt_state
    
    @functools.partial(jax.jit, static_argnums=0)
    def mse_loss(self, params, observations, actions, next_q_value):
        q_pred = self.network.apply(
            params, observations
        )  # (batch_size, num_actions)
        q_pred = q_pred[
            jnp.arange(q_pred.shape[0]), actions.squeeze().astype(int)
        ]  # (batch_size,)
        return ((q_pred - next_q_value) ** 2).mean(), q_pred
    
    @functools.partial(jax.jit, static_argnums=0)
    def _update_step(self, runner_state, _):
        (
            train_state,
            env_state,
            last_obs,
            rng,
            buffer_state,
            global_step,
        ) = runner_state

        def do_update(train_state, buffer_state):
            # batch = buffer.sample_fn(buffer_state, rng, config["batch_size"])
            batch = buffer.sample(buffer_state, rng)
            train_state, loss, q_pred, grads, opt_state = self.update(
                train_state,
                batch[0],
                batch[2],
                batch[1],
                batch[3],
                batch[4],
            )
            return train_state, loss, q_pred, grads, opt_state

        def dont_update(train_state, _):
            return (
                train_state,
                ((jnp.array([0]) - jnp.array([0])) ** 2).mean(),
                jnp.ones(config["batch_size"]),
                train_state.params,
                train_state.opt_state,
                None
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
            self.take_step,
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
        runner_state = (
            train_state,
            env_state,
            last_obs,
            rng,
            buffer_state,
            global_step,
        )
        if self.config["track_traj"]:
            metric = (
                loss,
                grads,
                opt_state,
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
        elif self.config["track_metrics"]:
            metric = (
                loss,
                grads,
                opt_state,
                {"q_pred": [q_pred], "td_error": [td_error]},
            )
        else:
            metric = None
        return runner_state, metric
    
    @functools.partial(jax.jit, static_argnums=0)
    def take_step(self, carry, _, rng, train_state, last_obs):
        def random_action():
            return jnp.array(
                [
                    self.env.action_space(self.env_params).sample(rng)
                    for _ in range(self.config["num_envs"])
                ]
            )

        def greedy_action():
            q_values = self.network.apply(train_state.params, last_obs)
            action = q_values.argmax(axis=-1)
            return action
        
        rng, _rng = jax.random.split(rng)
    
        obsv, env_state, global_step, buffer_state = carry
        action = jax.lax.cond(
            jax.random.uniform(rng) < self.config["epsilon"],
            random_action,
            greedy_action,
        )

        rng_step = jax.random.split(_rng, self.config["num_envs"])
        obsv, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, self.env_params)
        action = jnp.expand_dims(action, -1)
        done = jnp.expand_dims(done, -1)
        reward = jnp.expand_dims(reward, -1)

        def no_target_td(train_state):
            return self.network.apply(train_state.params, obsv).argmax(axis=-1)

        def target_td(train_state):
            return self.network.apply(train_state.target_params, obsv).argmax(
                axis=-1
            )

        q_next_target = jax.lax.cond(
            self.config["target"], target_td, no_target_td, train_state
        )

        td_error = (
            reward
            + (1 - done) * self.config["gamma"] * jnp.expand_dims(q_next_target, -1)
            - self.network.apply(train_state.params, last_obs).take(action)
        )
        transition_weight = jnp.power(
            jnp.abs(td_error) + self.config["buffer_epsilon"], self.config["alpha"]
        )

        # buffer_state = buffer.add_batch_fn(
        #     buffer_state,
        #     ((last_obs, obsv, action, reward, done), transition_weight),
        # )
        buffer_state = buffer.add(
            buffer_state,
            (last_obs, obsv, action, reward, done),
        )
        # buffer_state = buffer.set_priorities(buffer_state, ...)
        
        global_step += 1
        return (obsv, env_state, global_step, buffer_state), (
            obsv,
            action,
            reward,
            done,
            info,
            td_error,
        )
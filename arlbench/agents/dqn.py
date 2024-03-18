# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import chex
import optax
from .common import TimeStep
from typing import NamedTuple, Union
from flax.training.train_state import TrainState
import dejax.utils as utils
from typing import Callable, Any, Tuple
import gymnax
import chex
import jax.lax
import flashbax as fbx


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


def make_train_dqn(config, env, network, _):
    def train(
        rng,
        env_params,
        network_params,
        target_params,
        opt_state,
        obsv,
        env_state,
        buffer,
        buffer_state,
        global_step
    ):
        train_state_kwargs = {
            "apply_fn": network.apply,
            "params": network_params,
            "target_params": target_params,
            "tx": optax.adam(config["lr"], eps=1e-5),
            "opt_state": opt_state,
        }
        train_state = DQNTrainState.create_with_opt_state(**train_state_kwargs)

        # TRAIN LOOP
        def update(
            train_state, observations, actions, next_observations, rewards, dones
        ):
            if config["target"]:
                q_next_target = network.apply(
                    train_state.target_params, next_observations
                )  # (batch_size, num_actions)
            else:
                q_next_target = network.apply(
                    train_state.params, next_observations
                )  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            next_q_value = rewards + (1 - dones) * config["gamma"] * q_next_target

            def mse_loss(params):
                q_pred = network.apply(
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

        def _update_step(runner_state, unused):
            (
                train_state,
                env_state,
                last_obs,
                rng,
                buffer_state,
                global_step,
            ) = runner_state
            rng, _rng = jax.random.split(rng)

            def random_action():
                return jnp.array(
                    [
                        env.action_space(env_params).sample(rng)
                        for _ in range(config["num_envs"])
                    ]
                )

            def greedy_action():
                q_values = network.apply(train_state.params, last_obs)
                action = q_values.argmax(axis=-1)
                return action

            def take_step(carry, _):
                obsv, env_state, global_step, buffer_state = carry
                action = jax.lax.cond(
                    jax.random.uniform(rng) < config["epsilon"],
                    random_action,
                    greedy_action,
                )

                rng_step = jax.random.split(_rng, config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                def no_target_td(train_state):
                    return network.apply(train_state.params, obsv).argmax(axis=-1)

                def target_td(train_state):
                    return network.apply(train_state.target_params, obsv).argmax(
                        axis=-1
                    )

                q_next_target = jax.lax.cond(
                    config["target"], target_td, no_target_td, train_state
                )

                td_error = (
                    reward
                    + (1 - done) * config["gamma"] * jnp.expand_dims(q_next_target, -1)
                    - network.apply(train_state.params, last_obs).take(action)
                )
                transition_weight = jnp.power(
                    jnp.abs(td_error) + config["buffer_epsilon"], config["alpha"]
                )
                timestep = TimeStep(last_obs=last_obs, obs=obsv, action=action, reward=reward, done=done)
                
                buffer_state = buffer.add(buffer_state, timestep)
                # buffer_state = buffer.set_priorities(buffer_state, ...)
                
                # global_step += 1
                global_step += config["num_envs"]
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
                batch = buffer.sample(buffer_state, rng).experience.first
                train_state, loss, q_pred, grads, opt_state = update(
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
                    jnp.ones(config["batch_size"]),
                    train_state.params,
                    train_state.opt_state
                )

            def target_update():
                return train_state.replace(
                    target_params=optax.incremental_update(
                        train_state.params, train_state.target_params, config["tau"]
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
                config["train_frequency"],
            )

            train_state, loss, q_pred, grads, opt_state = jax.lax.cond(
                (global_step > config["learning_starts"])
                & (global_step % config["train_frequency"] == 0),
                do_update,
                dont_update,
                train_state,
                buffer_state,
            )
            train_state = jax.lax.cond(
                (global_step > config["learning_starts"])
                & (global_step % config["target_network_update_freq"] == 0),
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
            if config["track_traj"]:
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
            elif config["track_metrics"]:
                metric = (
                    loss,
                    grads,
                    opt_state,
                    {"q_pred": [q_pred], "td_error": [td_error]},
                )
            else:
                metric = None
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, buffer_state, global_step)
        runner_state, out = jax.lax.scan(
            _update_step, runner_state, None, (config["total_timesteps"]//config["train_frequency"])//config["num_envs"]
        )
        return runner_state, out

    return train

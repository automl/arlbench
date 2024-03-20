# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Union, Any, Dict, Optional
import chex
from .common import TimeStep
from flax.training.train_state import TrainState
import flashbax as fbx
import flax
import functools
from .agent import Agent
from .models import ActorCritic
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical


class PPORunnerState(NamedTuple):
    rng: chex.PRNGKey
    train_state: Any
    env_state: Any
    obs: chex.Array

class PPOTrainState(TrainState):
    target_params: Union[None, chex.Array, dict] = None
    opt_state = None

    @classmethod
    def create_with_opt_state(cls, *, apply_fn, params, tx, opt_state, **kwargs):
        if opt_state is None:
            opt_state = tx.init(params)
        obj = cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
        return obj


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class PPO(Agent):
    def __init__(
        self,
        config: Union[Configuration, Dict],
        env_options: Dict,
        env: Any,
        env_params: Any,
        track_metrics=False,
        track_trajectories=False
    ) -> None:
        super().__init__(
            config,
            env_options,
            env,
            env_params,
            track_metrics=track_metrics,
            track_trajectories=track_trajectories
        )
        # self.minibatch_size = (
        #     env_options["n_envs"] * env_options["n_env_steps"] // config["n_minibatches"]
        # )
        # TODO validate if it's okay to define minibatch_size rather than n_minibatches
        self.n_minibatches = env_options["n_envs"] * env_options["n_env_steps"] // config["minibatch_size"]

        action_size, discrete = self.action_type
        self.network = ActorCritic(
            action_size,
            discrete=discrete,
            activation=config["activation"],
            hidden_size=config["hidden_size"],
        )

        self.buffer = fbx.make_prioritised_flat_buffer(
            max_length=config["buffer_size"],
            min_length=config["buffer_batch_size"],
            sample_batch_size=config["buffer_batch_size"],
            add_sequences=False,
            add_batch_size=env_options["n_envs"],
            priority_exponent=config["buffer_beta"]    
        )

        self.n_total_updates = (
            env_options["n_total_timesteps"]
            // env_options["n_env_steps"]
            // env_options["n_envs"]
        )
        update_interval = jnp.ceil(self.n_total_updates / env_options["n_env_steps"])
        if update_interval < 1:
            update_interval = 1
            print(
                "WARNING: The number of iterations selected in combination with your timestep, n_envs and n_env_steps settings results in 0 steps per iteration. Rounded up to 1, this means more total steps will be executed."
            )

    @staticmethod
    def get_configuration_space(seed=None) -> ConfigurationSpace:
        return ConfigurationSpace(
            name="PPOConfigSpace",
            seed=seed,
            space={
                "buffer_size": Integer("buffer_size", (1, int(1e7)), default=int(1e6)),
                "buffer_batch_size": Integer("buffer_batch_size", (1, 1024), default=64),
                "buffer_beta": Float("buffer_beta", (0., 1.), default=0.9),
                "minibatch_size": Integer("minibatch_size", (4, 1024), default=256),
                "lr": Float("lr", (1e-5, 0.1), default=2.5e-4),
                "update_epochs": Integer("update_epochs", (1, int(1e5)), default=10),
                # 0 = tanh, 1 = relu, see agents.models.ACTIVATIONS
                "activation": Categorical("activation", [0, 1], default=0),        
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
    def get_default_configuration() -> Configuration:
        return PPO.get_configuration_space().get_default_configuration()

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, network_params=None, opt_state=None):
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

        tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(self.config["lr"], eps=1e-5),
            )
        if opt_state is None:
            opt_state = tx.init(network_params)
        
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
            env_state=last_env_state,
            obs=last_obsv,
        )

        return runner_state, buffer_state

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, network_params, obsv, rng) -> int:
        pi, _ = self.network.apply(network_params, obsv)
        return pi.sample(seed=rng)

    @functools.partial(jax.jit, static_argnums=0, donate_argnums=(2,))
    def train(
        self,
        runner_state,
        buffer_state
    ) -> tuple[tuple[PPORunnerState, Any], Optional[tuple]]:
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
        ) = jax.lax.scan(self._update_epoch, update_state, None, self.config["update_epochs"])
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
        rng_step = jax.random.split(_rng, self.env_options["n_envs"])
        obsv, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, self.env_params)

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
        delta = reward + self.config["gamma"] * next_value * (1 - done) - value
        gae = (
            delta
            + self.config["gamma"] * self.config["gae_lambda"] * (1 - done) * gae
        )
        return (gae, value), gae

    @functools.partial(jax.jit, static_argnums=0)
    def _update_epoch(self, update_state, _):
        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)

        # TODO verify this approach of discarding data points
        trimmed_batch_size = int(self.n_minibatches * self.config["minibatch_size"])
        batch_size = self.env_options["n_env_steps"] * self.env_options["n_envs"]

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
                x, [self.n_minibatches, -1] + list(x.shape[1:])
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
        if self.track_metrics:
            out = (total_loss, grads)
        else:
            out = (None, None)
        return train_state, out

    @functools.partial(jax.jit, static_argnums=0)
    def _loss_fn(self, params, traj_batch, gae, targets):
         # RERUN NETWORK
        pi, value = self.network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-self.config["clip_eps"], self.config["clip_eps"])
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
                1.0 - self.config["clip_eps"],
                1.0 + self.config["clip_eps"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.config["vf_coef"] * value_loss
            - self.config["ent_coef"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)


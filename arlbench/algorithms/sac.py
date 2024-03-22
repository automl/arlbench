import jax
import jax.numpy as jnp
import chex
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple
import dejax.utils as utils
from typing import Callable, Any, Tuple, Optional
import gymnax
import chex
import jax.lax
import flashbax as fbx
import functools
from arlbench.algorithms import Algorithm
from .common import TimeStep


# TODO implement
class SACTrainState(TrainState):
    actor_apply_fn: Callable
    critic_apply_fn: Callable
    actor_params: chex.ArrayTree
    critic_params: chex.ArrayTree
    target_critic_params: chex.ArrayTree
    actor_tx: Any
    critic_tx: Any
    actor_opt_state: Optional[chex.ArrayTree] = None
    critic_opt_state: Optional[chex.ArrayTree] = None

    @classmethod
    def create_with_opt_state(
        cls,
        *,
        actor_apply_fn,
        critic_apply_fn,
        actor_params, 
        critic_params, 
        actor_tx, 
        critic_tx,
        actor_opt_state, 
        critic_opt_state, 
        **kwargs
    ):
        if actor_opt_state is None:
            actor_opt_state = actor_tx.init(actor_params)
        if critic_opt_state is None:
            critic_opt_state = critic_tx.init(critic_params)

        target_critic_params = critic_params

        obj = cls(
            step=0,
            actor_apply_fn=actor_apply_fn,
            critic_apply_fn=critic_apply_fn,
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
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


class SAC(Algorithm):
    def __init__(
        self,
        config,
        env,
        env_params
    ) -> None:
        super().__init__(config, env, env_params)

        config["minibatch_size"] = (
            config["num_envs"] * config["num_steps"] // config["num_minibatches"]
        )
        self.config = config

        action_size, discrete = self.action_type
        self.network = ActorCritic(
            action_size,
            discrete=discrete,
        )

        self.buffer = fbx.make_prioritised_flat_buffer(
            max_length=config["buffer_size"],
            min_length=config["batch_size"],
            sample_batch_size=config["batch_size"],
            add_sequences=False,
            add_batch_size=config["num_envs"],
            priority_exponent=config["beta"]    
        )

        self.total_updates = (
            config["total_timesteps"]
            // config["num_steps"]
            // config["num_envs"]
        )
        update_interval = np.ceil(self.total_updates / config["num_steps"])
        if update_interval < 1:
            update_interval = 1
            print(
                "WARNING: The number of iterations selected in combination with your timestep, num_env and num_step settings results in 0 steps per iteration. Rounded up to 1, this means more total steps will be executed."
            )

    def init(self, rng, network_params=None):
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.config["num_envs"])

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
        opt_state = None    

        return (
            last_obsv,
            last_env_state
        ), (
            network_params,
            buffer_state,
            opt_state
        )

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, network_params, obsv, rng) -> int:
        pi, _ = self.network.apply(network_params, obsv)
        return pi.sample(seed=rng)

    @functools.partial(jax.jit, static_argnums=0)
    def train(
        self,
        rng,
        obsv,
        env_state,
        agent_state
    ):
        (
            network_params,
            buffer_state,
            opt_state
        ) = agent_state
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
        runner_state = (train_state, env_state, obsv, _rng, buffer_state)
        runner_state, out = jax.lax.scan(
            self._update_step, runner_state, None, self.total_updates
        )
        return runner_state, out
    
    @functools.partial(jax.jit, static_argnums=0)
    def _update_step(self, runner_state, unused):
        runner_state, traj_batch = jax.lax.scan(
            self._env_step, runner_state, None, self.config["num_steps"]
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng, buffer_state = runner_state
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

        runner_state = (train_state, env_state, last_obs, rng, buffer_state)
        if self.config["track_traj"]:
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
        elif self.config["track_metrics"]:
            out = (
                loss_info,
                grads,
                {"advantages": advantages, "param_history": param_hist["params"]},
            )
        else:
            out = None
        return runner_state, out
    
    @functools.partial(jax.jit, static_argnums=0)
    def _env_step(self, runner_state, unused):
        train_state, env_state, last_obs, rng, buffer_state = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = self.network.apply(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, self.config["num_envs"])
        obsv, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, self.env_params)

         # TODO make this running, apparently there is a problem with the shape of transitions
        # https://github.com/instadeepai/flashbax?tab=readme-ov-file#quickstart-
        # buffer_state = buffer.add(
        #     buffer_state,
        #     (
        #         last_obs,
        #         obsv,
        #         jnp.expand_dims(action, -1),
        #         jnp.expand_dims(reward, -1),
        #         jnp.expand_dims(done, -1),   
        #     ),
        # )

        transition = Transition(
            done, action, value, reward, log_prob, last_obs, info
        )
        runner_state = (train_state, env_state, obsv, rng, buffer_state)
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
        delta = reward + self.config["gamma"] * next_value * (1 - done) - value
        gae = (
            delta
            + self.config["gamma"] * self.config["gae_lambda"] * (1 - done) * gae
        )
        return (gae, value), gae

    @functools.partial(jax.jit, static_argnums=0)
    def _update_epoch(self, update_state, unused):
        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = int(self.config["minibatch_size"] * self.config["num_minibatches"])
        assert (
            batch_size == self.config["num_steps"] * self.config["num_envs"]
        ), "batch size must be equal to number of steps * number of envs"
        permutation = jax.random.permutation(_rng, batch_size)
        batch = (traj_batch, advantages, targets)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
        )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x, [self.config["num_minibatches"], -1] + list(x.shape[1:])
            ),
            shuffled_batch,
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
        if self.config["track_metrics"]:
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

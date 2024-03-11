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


Item = chex.ArrayTree
ItemBatch = chex.ArrayTree
IntScalar = chex.Array
ItemUpdateFn = Callable[[Item], Item]
BoolScalar = chex.Array


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    q_pred: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class SACExtendedTrainState(TrainState):
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
    

def make_train_sac(config, env, actor_network, critic_network):
    def train(
            rng,
            env_params,
            actor_params,
            critic_params,
            target_critic_params,
            actor_opt_state,
            critic_opt_state,
            obsv,
            env_state,
            buffer_state,
            global_step
    ):  
        train_state_kwargs = {
            "actor_apply_fn": actor_network.apply,
            "critic_apply_fn": critic_network.apply,
            "actor_params": actor_params,
            "critic_params": critic_params,
            "target_critic_params": target_critic_params,
            "actor_tx": optax.adam(config["actor_lr"], eps=1e-5),
            "critic_tx": optax.adam(config["critic_lr"], eps=1e-5),
            "actor_opt_state": actor_opt_state,
            "critic_opt_state": critic_opt_state,
        }

        train_state = SACExtendedTrainState.create_with_opt_state(**train_state_kwargs)
        buffer = fbx.make_prioritised_flat_buffer(
            max_length=int(config["buffer_size"]),
            min_length=int(config["batch_size"]),
            sample_batch_size=int(config["batch_size"]),
            priority_exponent=float(config["beta"])
        )
        global_step = global_step
        
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # Collect transitions and store in buffer
            
            # Sample mini-batches from replay buffer
            
            # Compute targets for Q-value updates
            
            # Update critic networks
            
            # Update actor network
            
            # Adjust entropy temperature (if implementing automatic temperature adjustment)
            
            # Update target networks
            
            # Optionally track metrics and return updated state and metrics
            return runner_state, metrics
        
        # Main training loop using jax.lax.scan or equivalent
        rng, _rng = jax.random.split(rng)
        runner_state = (actor_params, critic_params, target_critic_params, actor_opt_state, critic_opt_state, obsv, env_state, buffer_state)
        runner_state, out = jax.lax.scan(_update_step, runner_state, None, num_updates)
        
        return runner_state, out

    return train
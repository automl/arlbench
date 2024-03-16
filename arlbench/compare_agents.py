import jax
import jax.numpy as jnp
import numpy as np
import random
import flax
from flax import struct
from flax.training import orbax_utils
from flax.core.frozen_dict import FrozenDict
import gymnax
import orbax
import chex
from typing import Tuple, Union, Any, Dict
from agents import (
    make_train_ppo,
    ActorCritic,
    JAXPPO
)

from utils import (
    make_eval,
    make_env,
)

from gymnax.environments.environment import Environment
import flashbax as fbx
import time


# ------------------------------------------------------------------------
# 1) Environment initialization
# ------------------------------------------------------------------------
# AGENT = "Nested"
AGENT = "OOP"

config = {
    "buffer_size": 1000000,
    "batch_size": 64,
    "beta": 0.9,
    "minibatch_size": 64,
    "max_grad_norm": 5
}
instance = {
    "env_framework": "gymnax",
    "env_name": "Pendulum-v1",
    "total_timesteps": 1e6,
    "num_steps": 200,
    "num_envs": 2,
    "buffer_size": 1000000,
    "batch_size": 64,
    "beta": 0.9,
    "lr": 2.5e-4,
    "update_epochs": 4,
    "num_minibatches": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "anneal_lr": True,
    "track_metrics": False,
    "track_traj": False,
    "num_eval_episodes": 10
}

env, env_params = make_env(instance)

# TODO how to access state.rng here?
# for now, use initial rng based on config seed
rng = jax.random.PRNGKey(42)
rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng)

last_obsv, last_env_state = jax.vmap(env.reset, in_axes=(0, None))(
    reset_rng, env_params
)
global_step = 0

if isinstance(
    env.action_space(env_params), gymnax.environments.spaces.Discrete
):
    action_size = env.action_space(env_params).n
    action_buffer_size = 1
    discrete = True
elif isinstance(env.action_space(env_params), gymnax.environments.spaces.Box):
    action_size = env.action_space(env_params).shape[0]
    if len(env.action_space(env_params).shape) > 1:
        action_buffer_size = [
            env.action_space(env_params).shape[0],
            env.action_space(env_params).shape[1],
        ]
    elif env.name == "BraxToGymnaxWrapper":
        action_buffer_size = [action_size, 1]
    else:
        action_buffer_size = action_size

    discrete = False
else:
    raise NotImplementedError(
        f"Only Discrete and Box action spaces are supported, got {env.action_space(env_params)}."
    )

# ------------------------------------------------------------------------
# 2) Network and buffer initialization
# ------------------------------------------------------------------------
network = ActorCritic(
    action_size,
    discrete=discrete,
)
buffer = fbx.make_prioritised_flat_buffer(
    max_length=int(config["buffer_size"]),
    min_length=config["batch_size"],
    sample_batch_size=config["batch_size"],
    priority_exponent=config["beta"]
)
init_x = jnp.zeros(env.observation_space(env_params).shape)
buffer_state = buffer.init(
    (
        jnp.zeros(init_x.shape),
        jnp.zeros(init_x.shape),
        jnp.zeros(action_buffer_size),
        jnp.zeros(1),
        jnp.zeros(1),
    )
)

_, _rng = jax.random.split(rng)
network_params = network.init(_rng, init_x)
target_params = network.init(_rng, init_x)
opt_state = None
eval_func = make_eval(instance, network)
total_updates = (instance["total_timesteps"] // instance["num_steps"] // instance["num_envs"])
update_interval = np.ceil(total_updates / 1000000)
if update_interval < 1:
    update_interval = 1
    print(
        "WARNING: The number of iterations selected in combination with your timestep, num_env and num_step settings results in 0 steps per iteration. Rounded up to 1, this means more total steps will be executed."
    )

# ------------------------------------------------------------------------
# 3) Training
# ------------------------------------------------------------------------

train_func = jax.jit(
    make_train_ppo(
        instance, env, network, total_updates
    )
)

train_args = (
    rng,
    env_params,
    network_params,
    opt_state,
    last_obsv,
    last_env_state,
    buffer_state,
)

print(f"### {AGENT} ###")
start = time.time()


if AGENT == "Nested":
    runner_state, metrics = train_func(*train_args)
    network_params = runner_state[0].params
    last_obsv = runner_state[2]
    last_env_state = runner_state[1]
    buffer_state = runner_state[4]
    opt_info = runner_state[0].opt_state

    reward = eval_func(rng, network_params)

elif AGENT == "OOP":
    agent = JAXPPO(instance, env, env_params, network, total_updates)

    start = time.time()
    runner_state, metrics = agent.train(rng, network_params, opt_state, last_obsv, last_env_state, buffer_state)
    network_params = runner_state[0].params
    last_obsv = runner_state[2]
    last_env_state = runner_state[1]
    buffer_state = runner_state[4]
    opt_info = runner_state[0].opt_state

    reward = eval_func(rng, network_params)

train_time = time.time() - start

print(f"{train_time:.2f}s")
print(reward)

import jax
import numpy as np
import gymnax
import flashbax as fbx
import time

from agents import (
    make_train_ppo,
    make_train_dqn,
    Q,
    ActorCritic,
    PPO,
    DQN,
    TimeStep
)
from utils import (
    make_eval,
    make_env,
)


ALGORITHMS = {
    "ppo": (make_train_ppo, ActorCritic),
    "dqn": (make_train_dqn, Q),
    "oop-ppo": PPO,
    "oop-dqn": DQN
}

PPO_CONFIG = {
    "env_framework": "gymnax",
    "env_name": "CartPole-v1",
    "total_timesteps": 1e6,
    "num_steps": 500,
    "num_envs": 2,
    "buffer_size": 1000000,
    "batch_size": 64,
    "beta": 0.9,
    "lr": 2.5e-4,
    "update_epochs": 4,
    "activation": "tanh",
    "hidden_size": 64,
    "num_minibatches": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 5,
    "anneal_lr": True,
    "track_metrics": False,
    "track_traj": False,
    "num_eval_episodes": 10
}

DQN_CONFIG = {
    "env_framework": "gymnax",
    "env_name": "CartPole-v1",
    "total_timesteps": 1e6,
    "train_frequency": 10,
    "target_network_update_freq": 200,
    "learning_starts": 1000,
    "epsilon": 0.1,
    "tau": 1.0,
    "buffer_epsilon": 0.5,
    "alpha": 0.9,
    "beta": 0.9,
    "target": True,
    "num_steps": 500,
    "num_envs": 10,
    "buffer_size": 1000000,
    "batch_size": 128,
    "activation": "tanh",
    "hidden_size": 64,
    "beta": 0.9,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "anneal_lr": True,
    "track_metrics": False,
    "track_traj": False,
    "num_eval_episodes": 10
}

def prepare_training(algorithm, instance):
    env, env_params = make_env(instance)

    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, instance["num_envs"])

    last_obsv, last_env_state = jax.vmap(env.reset, in_axes=(0, None))(
        reset_rng, env_params
    )

    if isinstance(
        env.action_space(env_params), gymnax.environments.spaces.Discrete
    ):
        action_size = env.action_space(env_params).n
        discrete = True
    elif isinstance(env.action_space(env_params), gymnax.environments.spaces.Box):
        action_size = env.action_space(env_params).shape[0]
        discrete = False
    else:
        raise NotImplementedError(
            f"Only Discrete and Box action spaces are supported, got {env.action_space(env_params)}."
        )


    network = ALGORITHMS[algorithm][1](
        action_size,
        discrete=discrete,
    )
    buffer = fbx.make_prioritised_flat_buffer(
        max_length=instance["buffer_size"],
        min_length=instance["batch_size"],
        sample_batch_size=instance["batch_size"],
        add_sequences=False,
        add_batch_size=instance["num_envs"],
        priority_exponent=instance["beta"]    
    )
    dummy_rng = jax.random.PRNGKey(0) 
    _action = env.action_space().sample(dummy_rng)
    _, _env_state = env.reset(rng, env_params)
    _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)

    _timestep = TimeStep(last_obs=_obs, obs=_obs, action=_action, reward=_reward, done=_done)
    buffer_state = buffer.init(_timestep)

    _, _rng = jax.random.split(rng)
    network_params = network.init(_rng, _obs)
    if "dqn" in algorithm:
        target_params = network.init(_rng, _obs)
    else:
        target_params = None
    opt_state = None

    if "ppo" in algorithm:
        total_updates = (
            instance["total_timesteps"]
            // instance["num_steps"]
            // instance["num_envs"]
        )
        update_interval = np.ceil(total_updates / instance["num_steps"])
        if update_interval < 1:
            update_interval = 1
            print(
                "WARNING: The number of iterations selected in combination with your timestep, num_env and num_step settings results in 0 steps per iteration. Rounded up to 1, this means more total steps will be executed."
            )
    else:
        total_updates = None

    return (
        rng,
        env, 
        env_params, 
        network, 
        network_params, 
        target_params,
        opt_state, 
        last_obsv, 
        last_env_state, 
        buffer,
        buffer_state, 
        total_updates
    )

def test_nested_ppo():
    (
        rng,
        env, 
        env_params, 
        network, 
        network_params, 
        target_params,
        opt_state, 
        last_obsv, 
        last_env_state, 
        buffer,
        buffer_state, 
        total_updates
    ) = prepare_training("ppo", PPO_CONFIG)

    train_func = jax.jit(
        make_train_ppo(
            PPO_CONFIG, env, network, total_updates
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

    start = time.time()
    runner_state, _ = train_func(*train_args)
    network_params = runner_state[0].params
    training_time = time.time() - start

    eval_func = make_eval(PPO_CONFIG, network)
    return (eval_func(rng, network_params), training_time)

def test_oop_ppo():
    env, env_params = make_env(PPO_CONFIG)
    rng = jax.random.PRNGKey(42)

    agent = PPO(PPO_CONFIG, env, env_params)
    runner_state = agent.init(rng)
    
    start = time.time()
    runner_state, _ = agent.train(runner_state)
    training_time = time.time() - start
    return (agent.eval(runner_state, PPO_CONFIG["num_eval_episodes"]), training_time)


def test_nested_dqn():
    (
        rng,
        env, 
        env_params, 
        network, 
        network_params, 
        target_params,
        opt_state, 
        last_obsv, 
        last_env_state, 
        buffer,
        buffer_state, 
        total_updates
    ) = prepare_training("dqn", DQN_CONFIG)

    train_func = make_train_dqn(
            DQN_CONFIG, env, network, total_updates
        )
    
    global_step = 0

    train_args = (
        rng,
        env_params,
        network_params,
        target_params,
        opt_state,
        last_obsv,
        last_env_state,
        buffer,
        buffer_state,
        global_step,
    )

    start = time.time()
    runner_state, _ = train_func(*train_args)
    network_params = runner_state[0].params
    global_step = runner_state[5]
    training_time = time.time() - start
    print(global_step)

    eval_func = make_eval(DQN_CONFIG, network)
    return (eval_func(rng, network_params), training_time)

def test_oop_dqn():
    env, env_params = make_env(DQN_CONFIG)
    rng = jax.random.PRNGKey(42)

    agent = DQN(DQN_CONFIG, env, env_params)
    runner_state = agent.init(rng)
    
    start = time.time()
    runner_state, _ = agent.train(runner_state)
    training_time = time.time() - start
    return (agent.eval(runner_state, DQN_CONFIG["num_eval_episodes"]), training_time)


print("# OOP PPO #")
print(test_oop_ppo())

# print("# Nested PPO #")
# print(test_nested_ppo())

print("# OOP DQN #")
print(test_oop_dqn())

# print("# Nested DQN #")
# print(test_nested_dqn())
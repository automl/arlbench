import jax
import numpy as np
import gymnax
import flashbax as fbx
import time

from arlbench.agents import (
    make_train_ppo,
    make_train_dqn,
    Q,
    ActorCritic,
    PPO,
    DQN,
    TimeStep
)
from arlbench.utils import (
    make_env,
)

ALGORITHMS = {
    "ppo": (make_train_ppo, ActorCritic),
    "dqn": (make_train_dqn, Q),
    "oop-ppo": PPO,
    "oop-dqn": DQN
}

OPTIONS = {
    "n_total_timesteps": 1e5,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}

PPO_CONFIG = PPO.get_default_configuration()
DQN_CONFIG = DQN.get_default_configuration()

def prepare_training(algorithm, config, options):
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, options["n_envs"])

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
        max_length=config["buffer_size"],
        min_length=config["batch_size"],
        sample_batch_size=config["batch_size"],
        add_sequences=False,
        add_batch_size=config["num_envs"],
        priority_exponent=config["beta"]    
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
            options["n_total_timesteps"]
            // options["n_env_steps"]
            // options["n_envs"]
        )
        update_interval = np.ceil(total_updates / options["n_env_steps"])
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

def test_oop_ppo():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    agent = PPO(PPO_CONFIG, OPTIONS, env, env_params)
    runner_state = agent.init(rng)
    
    start = time.time()
    runner_state, _ = agent.train(runner_state)
    training_time = time.time() - start
    return (agent.eval(runner_state, OPTIONS["n_eval_episodes"]), training_time)

def test_oop_dqn():
    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    agent = DQN(DQN_CONFIG, OPTIONS, env, env_params)
    runner_state = agent.init(rng)
    
    start = time.time()
    runner_state, _ = agent.train(runner_state)
    training_time = time.time() - start
    return (agent.eval(runner_state, OPTIONS["n_eval_episodes"]), training_time)


print("# OOP PPO #")
print(test_oop_ppo())

print("# OOP DQN #")
print(test_oop_dqn())

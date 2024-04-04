import jax
import numpy as np
from arlbench.algorithms import (
    PPO,
    DQN
)

from arlbench.autorl_env.checkpointing import Checkpointer
from arlbench.utils import make_env


def test_read_write_buffer():
    CHECKPOINT_DIR = "/tmp/test_saves"
    CHECKPOINT_NAME = "unit_test"

    DQN_OPTIONS = {
        "n_total_timesteps": 1e6,
        "n_envs": 10,
        "n_env_steps": 500,
        "n_eval_episodes": 10,
        "track_metrics": False,
        "track_traj": False,
    }

    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env, env_params)
    runner_state, buffer_state = agent.init(rng)

    (runner_state, buffer_state), _ = agent.train(runner_state, buffer_state)
    buffer_checkpoint = Checkpointer.save_buffer(buffer_state, CHECKPOINT_DIR, CHECKPOINT_NAME)

    _, init_buffer_state = agent.init(rng)
    disk_buffer_state = Checkpointer.load_buffer(
        init_buffer_state,
        buffer_checkpoint["priority_state_path"],
        buffer_checkpoint["buffer_dir"],
        buffer_checkpoint["vault_uuid"]
    )

    assert np.allclose(buffer_state.experience.obs, disk_buffer_state.experience.obs, atol=1e-7)
    assert np.allclose(buffer_state.experience.last_obs, disk_buffer_state.experience.last_obs, atol=1e-7)
    assert np.allclose(buffer_state.experience.reward, disk_buffer_state.experience.reward, atol=1e-7)
    assert np.allclose(buffer_state.experience.action, disk_buffer_state.experience.action, atol=1e-7)

    (runner_state, disk_buffer_state), _ = agent.train(runner_state, disk_buffer_state)
    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    assert reward > 480


def test_checkpoints_dqn():
    C_STEP = 2
    C_EPISODE = 4
    DONE = False

    OPTIONS = {
        "algorithm": "dqn",
        "checkpoint_dir": "/tmp/test_saves",
        "checkpoint_name": "checkpoint_test",
        "checkpoint": ["opt_state", "policy", "buffer", "loss", "extras", "minibatches", "trajectory"],
        "load": "/tmp/test_saves/checkpoint_test/checkpoint_test_c_episode_4_step_2",
        "grad_obs": False
    }

    DQN_OPTIONS = {
        "n_total_timesteps": 1e6,
        "n_envs": 10,
        "n_env_steps": 500,
        "n_eval_episodes": 10
    }

    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = dict(DQN.get_default_hpo_config())
    algorithm = DQN(
        config,
        DQN_OPTIONS,
        env,
        env_params,
        track_metrics=True,
        track_trajectories=True,
    )
    runner_state, init_buffer_state = algorithm.init(rng)

    (runner_state, buffer_state), metrics = algorithm.train(runner_state, init_buffer_state)

    rewards = algorithm.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    Checkpointer.save(runner_state, buffer_state, OPTIONS, config, DONE, C_EPISODE, C_STEP, metrics)

    (
        config,
        c_step,
        c_episode
    ), algorithm_kw_args = Checkpointer.load(OPTIONS["load"], OPTIONS["algorithm"], init_buffer_state)

    assert c_step == C_STEP
    assert c_episode == C_EPISODE
    assert c_episode == C_EPISODE

    runner_state, _ = algorithm.init(rng, **algorithm_kw_args)
    rewards_reload = algorithm.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward_reload = np.mean(rewards_reload)

    assert np.abs(reward - reward_reload) < 50


def test_checkpoints_ppo():
    C_STEP = 2
    C_EPISODE = 4
    DONE = False

    OPTIONS = {
        "algorithm": "ppo",
        "checkpoint_dir": "/tmp/test_saves",
        "checkpoint_name": "checkpoint_test",
        "checkpoint": ["opt_state", "policy", "buffer", "loss", "extras", "minibatches", "trajectory"],
        "load": "/tmp/test_saves/checkpoint_test/checkpoint_test_c_episode_4_step_2",
        "grad_obs": False
    }

    PPO_OPTIONS = {
        "n_total_timesteps": 1e5,
        "n_envs": 10,
        "n_env_steps": 500,
        "n_eval_episodes": 10
    }

    env, env_params = make_env("gymnax", "CartPole-v1")
    rng = jax.random.PRNGKey(42)

    config = dict(PPO.get_default_hpo_config())
    algorithm = PPO(
        config,
        PPO_OPTIONS,
        env,
        env_params,
        track_metrics=True,
        track_trajectories=True,
    )
    runner_state, init_buffer_state = algorithm.init(rng)

    (runner_state, buffer_state), metrics = algorithm.train(runner_state, init_buffer_state)

    rewards = algorithm.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)
    Checkpointer.save(runner_state, buffer_state, OPTIONS, config, DONE, C_EPISODE, C_STEP, metrics)

    (
        config,
        c_step,
        c_episode
    ), algorithm_kw_args = Checkpointer.load(OPTIONS["load"], OPTIONS["algorithm"], init_buffer_state)

    assert c_step == C_STEP
    assert c_episode == C_EPISODE
    assert c_episode == C_EPISODE

    runner_state, _ = algorithm.init(rng, **algorithm_kw_args)
    rewards_reload = algorithm.eval(runner_state, PPO_OPTIONS["n_eval_episodes"])
    reward_reload = np.mean(rewards_reload)
    assert np.abs(reward - reward_reload) < 50      # PPO is more stochastic



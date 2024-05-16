from __future__ import annotations

import jax
import numpy as np

from arlbench.autorl import AutoRLEnv
from arlbench.autorl.checkpointing import Checkpointer
from arlbench.core.algorithms import DQN
from arlbench.core.environments import make_env


def test_read_write_buffer():
    CHECKPOINT_DIR = "/tmp/test_saves"
    CHECKPOINT_NAME = "unit_test"
    N_TOTAL_TIMESTEPS = 1e6
    EVAL_STEPS = 10
    EVAL_EPISODES = 10
    N_ENVS = 10
    SEED = 42

    env = make_env("gymnax", "CartPole-v1", n_envs=N_ENVS, seed=SEED)
    rng = jax.random.PRNGKey(SEED)

    hp_config = DQN.get_default_hpo_config()
    algorithm = DQN(hp_config, env)
    runner_state, buffer_state = algorithm.init(rng)

    runner_state, buffer_state, result = algorithm.train(
        runner_state,
        buffer_state,
        n_total_timesteps=N_TOTAL_TIMESTEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES,
    )
    buffer_checkpoint = Checkpointer.save_buffer(
        buffer_state, CHECKPOINT_DIR, CHECKPOINT_NAME
    )

    _, init_buffer_state = algorithm.init(rng)
    disk_buffer_state = Checkpointer.load_buffer(
        init_buffer_state,
        buffer_checkpoint["priority_state_path"],
        buffer_checkpoint["buffer_dir"],
        buffer_checkpoint["vault_uuid"],
    )

    assert np.allclose(
        buffer_state.experience.obs, disk_buffer_state.experience.obs, atol=1e-7
    )
    assert np.allclose(
        buffer_state.experience.last_obs,
        disk_buffer_state.experience.last_obs,
        atol=1e-7,
    )
    assert np.allclose(
        buffer_state.experience.reward, disk_buffer_state.experience.reward, atol=1e-7
    )
    assert np.allclose(
        buffer_state.experience.action, disk_buffer_state.experience.action, atol=1e-7
    )

    runner_state, disk_buffer_state, _ = algorithm.train(
        runner_state, disk_buffer_state, n_total_timesteps=1e6
    )
    rewards = algorithm.eval(runner_state, 10)
    reward = np.mean(rewards)
    assert reward > 480


def test_checkpointing_dqn():
    config = {
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "dqn",
        "checkpoint": ["all"],
        "checkpoint_name": "test_checkpointing_dqn",
    }

    env = AutoRLEnv(config=config)
    _ = env.reset()

    action = env.config_space.get_default_configuration()
    _, objectives, _, _, info = env.step(action)

    reward = objectives["reward_mean"]

    _ = env.reset(checkpoint_path=info["checkpoint"])

    new_reward = np.mean(env.eval(10))
    assert np.isclose(reward, new_reward, rtol=1.0)

    env.step(action, n_total_timesteps=10000)


def test_checkpointing_ppo():
    config = {
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "ppo",
        "checkpoint": ["all"],
        "checkpoint_name": "test_checkpointing_ppo",
    }

    env = AutoRLEnv(config=config)
    _ = env.reset()

    action = env.config_space.get_default_configuration()
    _, objectives, _, _, info = env.step(action)

    reward = objectives["reward_mean"]

    _ = env.reset(checkpoint_path=info["checkpoint"])

    new_reward = np.mean(env.eval(10))
    assert np.isclose(reward, new_reward, rtol=1.0)

    env.step(action, n_total_timesteps=10000)


def test_checkpointing_sac():
    config = {
        "env_framework": "gymnax",
        "env_name": "Pendulum-v1",
        "n_envs": 10,
        "algorithm": "sac",
        "checkpoint": ["all"],
        "checkpoint_name": "test_checkpointing_sac",
    }

    env = AutoRLEnv(config=config)
    _ = env.reset()

    action = env.config_space.get_default_configuration()
    _, objectives, _, _, info = env.step(action, n_total_timesteps=10000)

    reward = objectives["reward_mean"]

    _ = env.reset(checkpoint_path=info["checkpoint"])

    new_reward = np.mean(env.eval(10))
    assert np.isclose(reward, new_reward, rtol=1.0)

    env.step(action, n_total_timesteps=10000)


if __name__ == "__main__":
    test_checkpointing_dqn()
    test_checkpointing_ppo()
    test_checkpointing_sac()

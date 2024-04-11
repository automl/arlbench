import jax
import numpy as np

from arlbench.autorl.checkpointing import Checkpointer
from arlbench.core.algorithms import DQN, PPO, SAC
from arlbench.core.environments import make_env


N_TOTAL_TIMESTEPS = 1e6
EVAL_STEPS = 10
EVAL_EPISODES = 10
N_ENVS = 10
SEED = 42


def test_read_write_buffer():
    CHECKPOINT_DIR = "/tmp/test_saves"
    CHECKPOINT_NAME = "unit_test"

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
        n_eval_episodes=EVAL_EPISODES
    )
    buffer_checkpoint = Checkpointer.save_buffer(buffer_state, CHECKPOINT_DIR, CHECKPOINT_NAME)

    _, init_buffer_state = algorithm.init(rng)
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

    runner_state, disk_buffer_state, _ = algorithm.train(runner_state, disk_buffer_state, n_total_timesteps=1e6)
    rewards = algorithm.eval(runner_state, 10)
    reward = np.mean(rewards)
    assert reward > 480


def test_checkpoints_dqn():
    C_STEP = 2
    C_EPISODE = 4
    DONE = False

    OPTIONS = {
        "algorithm": "dqn",
        "checkpoint_dir": "/tmp/test_saves",
        "checkpoint_name": "checkpoint_test_dqn",
        "checkpoint": ["opt_state", "params", "loss"],
        "load": "/tmp/test_saves/checkpoint_test_dqn/checkpoint_test_c_episode_4_step_2",
    }

    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    hp_config = DQN.get_default_hpo_config()
    algorithm = DQN(
        hp_config,
        env,
        track_metrics=True,
        track_trajectories=False,
    )
    runner_state, init_buffer_state = algorithm.init(rng)

    runner_state, buffer_state, result = algorithm.train(
        runner_state,
        init_buffer_state,
        n_total_timesteps=N_TOTAL_TIMESTEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )

    eval_rng = jax.random.PRNGKey(42)
    runner_state._replace(rng=eval_rng)
    rewards = algorithm.eval(runner_state, EVAL_EPISODES)
    reward = np.mean(rewards)
    Checkpointer.save(
        algorithm.name,
        runner_state,
        buffer_state,
        OPTIONS,
        hp_config,
        DONE,
        C_EPISODE,
        C_STEP,
        result
    )
    (
        hp_config,
        c_step,
        c_episode
    ), algorithm_kw_args = Checkpointer.load(OPTIONS["load"], OPTIONS["algorithm"], init_buffer_state)

    assert c_step == C_STEP
    assert c_episode == C_EPISODE
    assert c_episode == C_EPISODE

    runner_state, _ = algorithm.init(rng, **algorithm_kw_args)
    
    eval_rng = jax.random.PRNGKey(42)
    runner_state._replace(rng=eval_rng)
    rewards_reload = algorithm.eval(runner_state, EVAL_EPISODES)
    reward_reload = np.mean(rewards_reload)

    assert np.abs(reward - reward_reload) < 5


def test_checkpoints_ppo():
    C_STEP = 2
    C_EPISODE = 4
    DONE = False

    OPTIONS = {
        "algorithm": "ppo",
        "checkpoint_dir": "/tmp/test_saves",
        "checkpoint_name": "checkpoint_test_ppo",
        "checkpoint": ["opt_state", "params", "loss"],
        "load": "/tmp/test_saves/checkpoint_test_ppo/checkpoint_test_c_episode_4_step_2",
    }

    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    hp_config = SAC.get_default_hpo_config()
    algorithm = SAC(
        hp_config,
        env,
        track_metrics=True,
        track_trajectories=False,
    )
    runner_state, init_buffer_state = algorithm.init(rng)

    runner_state, buffer_state, result = algorithm.train(
        runner_state,
        init_buffer_state,
        n_total_timesteps=N_TOTAL_TIMESTEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )

    eval_rng = jax.random.PRNGKey(42)
    runner_state._replace(rng=eval_rng)
    rewards = algorithm.eval(runner_state, EVAL_EPISODES)
    reward = np.mean(rewards)
    Checkpointer.save(
        algorithm.name,
        runner_state,
        buffer_state,
        OPTIONS,
        hp_config,
        DONE,
        C_EPISODE,
        C_STEP,
        result
    )
    (
        hp_config,
        c_step,
        c_episode
    ), algorithm_kw_args = Checkpointer.load(OPTIONS["load"], OPTIONS["algorithm"], init_buffer_state)

    assert c_step == C_STEP
    assert c_episode == C_EPISODE
    assert c_episode == C_EPISODE

    runner_state, _ = algorithm.init(rng, **algorithm_kw_args)
    
    eval_rng = jax.random.PRNGKey(42)
    runner_state._replace(rng=eval_rng)
    rewards_reload = algorithm.eval(runner_state, EVAL_EPISODES)
    reward_reload = np.mean(rewards_reload)

    assert np.abs(reward - reward_reload) < 50


def test_checkpoints_sac():
    C_STEP = 2
    C_EPISODE = 4
    DONE = False

    OPTIONS = {
        "algorithm": "sac",
        "checkpoint_dir": "/tmp/test_saves",
        "checkpoint_name": "checkpoint_test_ppo",
        "checkpoint": ["opt_state", "params", "loss"],
        "load": "/tmp/test_saves/checkpoint_test_ppo/checkpoint_test_c_episode_4_step_2",
    }

    env = make_env("gymnax", "Pendulum-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    hp_config = SAC.get_default_hpo_config()
    algorithm = SAC(
        hp_config,
        env,
        track_metrics=True,
        track_trajectories=False,
    )
    runner_state, init_buffer_state = algorithm.init(rng)

    runner_state, buffer_state, result = algorithm.train(
        runner_state,
        init_buffer_state,
        n_total_timesteps=N_TOTAL_TIMESTEPS,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )

    eval_rng = jax.random.PRNGKey(42)
    runner_state._replace(rng=eval_rng)
    rewards = algorithm.eval(runner_state, EVAL_EPISODES)
    reward = np.mean(rewards)
    Checkpointer.save(
        algorithm.name,
        runner_state,
        buffer_state,
        OPTIONS,
        hp_config,
        DONE,
        C_EPISODE,
        C_STEP,
        result
    )
    (
        hp_config,
        c_step,
        c_episode
    ), algorithm_kw_args = Checkpointer.load(OPTIONS["load"], OPTIONS["algorithm"], init_buffer_state)

    assert c_step == C_STEP
    assert c_episode == C_EPISODE
    assert c_episode == C_EPISODE

    runner_state, _ = algorithm.init(rng, **algorithm_kw_args)

    eval_rng = jax.random.PRNGKey(42)
    runner_state._replace(rng=eval_rng)
    rewards_reload = algorithm.eval(runner_state, EVAL_EPISODES)
    reward_reload = np.mean(rewards_reload)

    assert np.abs(reward - reward_reload) < 50


if __name__ == "__main__":
    test_checkpoints_sac()
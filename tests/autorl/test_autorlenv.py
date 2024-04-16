from arlbench import AutoRLEnv


def test_autorlenv_checkpoints_dqn():
    config = {
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "dqn",
        "checkpoint": [],
    }
    pass


def test_autorlenv_checkpoints_ppo():
    # TODO implement
    pass


def test_autorlenv_checkpoints_sac():
    # TODO implement
    pass
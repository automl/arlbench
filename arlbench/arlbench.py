from .utils import HandleTermination
from .autorl import AutoRLEnv


def cool_things(cfg):
    """As the name suggests, this does cool things."""
    # TODO this is a dummy
    with HandleTermination(AutoRLEnv({}, {})):
        print(f"Your current config is: {cfg}")
        

AUTORL_DEFAULTS = {
    "seed": 0,
    "algorithm": "dqn",
    "objectives": ["reward"],
    "checkpoint": [],
    "n_steps": 10,
    "n_eval_episodes": 10,
    "track_trajectories": False,
    "grad_obs": True
}

class AutoRLBenchmark:
    def __init__(self, config=None):
        """
        Initialize AutoRL Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        self.train_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.test_seeds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        if not self.config:
            self.config = AUTORL_DEFAULTS.copy()

        for key in AUTORL_DEFAULTS:
            if key not in self.config:
                self.config[key] = AUTORL_DEFAULTS[key]

    def get_environment(self):
        """
        Return AutoRL env with current configuration

        Returns
        -------
        AutoRLEnv
            AutoRL environment
        """
        raise NotImplementedError
        return AutoRLEnv(self.config)

    def get_benchmark(
        self,
        seed,
        name="everything",
        test=False,
        level="mix",
        dynamic=False,
        get_data=False,
    ):
        raise NotImplementedError

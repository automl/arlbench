from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arlbench.core.algorithms import Algorithm


def track_runtime(train_func: Callable, objectives: dict) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = train_func(*args, **kwargs)
        objectives["runtime"] = time.time() - start_time
        return result
    return wrapper

def track_reward(train_func: Callable, objectives: dict, algorithm: Algorithm, n_eval_episodes: int) -> Callable:
    def wrapper(*args, **kwargs):
        result = train_func(*args, **kwargs)
        (runner_state, _), _ = result
        rewards = algorithm.eval(runner_state, n_eval_episodes)
        objectives["reward_mean"] = np.mean(rewards)
        objectives["reward_std"] = np.std(rewards)

        # TODO add more stability metrics

        return result
    return wrapper

def track_emissions(train_func: Callable, objectives: dict) -> Callable:
    def wrapper(*args, **kwargs):
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(
            save_to_file=False,
            output_dir="/tmp",
            logging_logger=None
        )
        tracker.start()
        try:
            result = train_func(*args, **kwargs)
        finally:
            objectives["emissions"] = tracker.stop()
        return result
    return wrapper


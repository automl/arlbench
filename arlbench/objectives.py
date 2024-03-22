from typing import Callable
import time
from arlbench.algorithms import Algorithm


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
        objectives["reward"] = algorithm.eval(runner_state, n_eval_episodes)
        return result
    return wrapper 

def track_emissions(train_func: Callable, objectives: dict) -> Callable:
    def wrapper(*args, **kwargs):
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker()
        tracker.start()
        try:
            result = train_func(*args, **kwargs)
        finally:
            objectives["emissions"] = tracker.stop()
        return result
    return wrapper


from __future__ import annotations
from abc import ABC, abstractmethod

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, List

import numpy as np

from .autorl_env import AutoRLEnv

if TYPE_CHECKING:
    from ..core.algorithms import Algorithm

# SORTING RANKS
# Runtime = 0
# Emissions = 1
# Reward = 2

class Objective(ABC):
    RANK: int   # Sorting rank 

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance.__call__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def __call__(train_func: Callable, objectives: dict, env: AutoRLEnv) -> Callable:
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def get_spec() -> dict:
        raise NotImplementedError
    
    def __lt__(self, other: Objective) -> bool:
        return self.RANK < other.RANK
    

class RuntimeObjective(Objective):
    RANK = 0

    @staticmethod
    def __call__(train_func: Callable, objectives: dict, _) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = train_func(*args, **kwargs)
            objectives["runtime"] = time.time() - start_time
            return result
        return wrapper
    
    @staticmethod
    def get_spec() -> dict:
        return {
            "name": "runtime",
            "upper": None,
            "lower": 0.,
            "optimize": "lower"
        }


class RewardObjective(Objective):
    RANK = 2

    @staticmethod
    def __call__(train_func: Callable, objectives: dict, env: AutoRLEnv) -> Callable:
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)
            (runner_state, _), _ = result
            rewards = env.algorithm.eval(runner_state, env.config["reward_eval_episodes"])
            objectives["reward"] = np.mean(rewards)
            objectives["episode_rewards"] = list(rewards)

            return result
        return wrapper
    
    @staticmethod
    def get_spec() -> dict:
        return {
            "name": "reward",
            "upper": None,
            "lower": None,
            "optimize": "upper"
        }
    

class EmissionsObjective(Objective):
    RANK = 1

    @staticmethod
    def __call__(train_func: Callable, objectives: dict, _) -> Callable:
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
    
    @staticmethod
    def get_spec() -> dict:
        return {
            "name": "emissions",
            "upper": None,
            "lower": 0.,
            "optimize": "lower"
        }


OBJECTIVES = {
    "runtime": RuntimeObjective,
    "reward": RewardObjective,
    "emissions": EmissionsObjective
}

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gymnasium
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from arlbench.core.algorithms import TrainFunc


class StateFeature(ABC):
    KEY: str        # Unique identifier

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance.__call__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def __call__(train_func: TrainFunc, objectives: dict) -> TrainFunc:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_state_space() -> gymnasium.spaces.Space:
        raise NotImplementedError


class GradInfo(StateFeature):
    KEY = "grad_info"

    @staticmethod
    def __call__(train_func: TrainFunc, state_features: dict) -> TrainFunc:
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)

            if result.metrics and len(result.metrics) >= 3:
                grad_info = result.metrics[1]
            else:
                raise ValueError("Tying to extract grad_info but 'self.metrics' does not match.")

            grad_info = grad_info["params"]
            grad_info = {
                k: v
                for (k, v) in grad_info.items()
                if isinstance(v, dict)
            }

            grad_info = [
                grad_info[g][k] for g in grad_info for k in grad_info[g]
            ]

            grad_norm = np.mean([jnp.linalg.norm(g) for g in grad_info])
            grad_var = np.mean([jnp.var(g) for g in grad_info])

            state_features[GradInfo.KEY] = np.array([grad_norm, grad_var])

            return result
        return wrapper

    @staticmethod
    def get_state_space() -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(
            low=np.array([-np.inf, 0]),
            high=np.array([np.inf, np.inf])
        )


STATE_FEATURES = {
    o.KEY: o for o in [GradInfo]
}




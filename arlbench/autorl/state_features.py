"""State features for the AutoRL environment."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gymnasium
import jax.numpy as jnp
import numpy as np

from arlbench.core.algorithms import DQNMetrics, PPOMetrics, SACMetrics

if TYPE_CHECKING:
    from arlbench.core.algorithms import TrainFunc


class StateFeature(ABC):
    """An abstract state features for the AutoRL environment.

    It can be wrapped around the training function to calculate the state features.
    We do this be overriding the __new__() function. It allows us to imitate
    the behaviour of a basic function while keeping the advantages of a static class.
    """
    KEY: str  # Unique identifier

    def __new__(cls, *args, **kwargs) -> TrainFunc:
        """Creates a new instance of this state feature and directly wraps the train function.

        This is done by first creating an object and subsequently calling self.__call__().

        Returns:
            TrainFunc: Wrapped training function.
        """
        instance = super().__new__(cls)
        return instance.__call__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def __call__(train_func: TrainFunc, state_features: dict) -> TrainFunc:
        """Wraps the training function with the state feature calculation.

        Args:
            train_func (TrainFunc):  Training function to wrap.
            state_features (dict):  Dictionary to store state features.

        Returns:
            TrainFunc:  Wrapped training function.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_state_space() -> gymnasium.spaces.Space:
        """Returns a dictionary containing the specification of the state feature.

        Returns:
            dict: Specification.
        """
        raise NotImplementedError


class GradInfo(StateFeature):
    """Gradient information state feature for the AutoRL environment. It contains the grad norm during training."""

    KEY = "grad_info"

    @staticmethod
    def __call__(train_func: TrainFunc, state_features: dict) -> TrainFunc:
        """Wraps the training function with the gradient information calculation."""
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)

            _, train_result = result
            metrics = train_result.metrics

            if metrics is None:
                raise ValueError(
                    "Metrics in train_result are None. Can't compute gradient info without gradients."
                )

            if isinstance(metrics, DQNMetrics | PPOMetrics):
                grad_info = metrics.grads["params"]
            elif isinstance(metrics, SACMetrics):
                grad_info = metrics.actor_grads["params"]

            grad_info = {k: v for (k, v) in grad_info.items() if isinstance(v, dict)}

            grad_info = [grad_info[g][k] for g in grad_info for k in grad_info[g]]

            grad_norm = np.mean([jnp.linalg.norm(g) for g in grad_info])
            grad_var = np.mean([jnp.var(g) for g in grad_info])

            state_features[GradInfo.KEY] = np.array([grad_norm, grad_var])

            return result

        return wrapper

    @staticmethod
    def get_state_space() -> gymnasium.spaces.Space:
        """Returns state space."""
        return gymnasium.spaces.Box(
            low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf])
        )


STATE_FEATURES = {o.KEY: o for o in [GradInfo]}

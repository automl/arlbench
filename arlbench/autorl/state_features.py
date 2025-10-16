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

class LossInfo(StateFeature):
    """Loss information state feature for the AutoRL environment. It contains the loss mean and variance."""

    KEY = "loss_info"

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

            loss_info = metrics.loss
            loss_mean = np.mean(loss_info)
            loss_var = np.mean(loss_info)

            state_features[LossInfo.KEY] = np.array([loss_mean, loss_var])

            return result

        return wrapper

    @staticmethod
    def get_state_space() -> gymnasium.spaces.Space:
        """Returns state space."""
        return gymnasium.spaces.Box(
            low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf])
        )


class WeightInfo(StateFeature):
    """Weight information state feature for the AutoRL environment. It contains statistics about the weights during training."""

    KEY = "weight_info"

    @staticmethod
    def __call__(train_func: TrainFunc, state_features: dict) -> TrainFunc:
        """Wraps the training function with the gradient information calculation."""
        def wrapper(*args, **kwargs):
            def get_stats(params):
                weights = [v["kernel"].flatten() for (k, v) in params.items() if isinstance(v, dict)]
                biases = [v["bias"].flatten() for (k, v) in params.items() if isinstance(v, dict)]
                weights = jnp.concatenate(weights)
                biases = jnp.concatenate(biases)

                w_mean = jnp.mean(weights)
                w_var = jnp.var(weights)
                w_median = jnp.median(weights)
                b_mean = jnp.mean(biases)
                b_var = jnp.var(biases)
                b_median = jnp.median(biases)
                return np.array(
                        [w_mean, w_var, w_median, b_mean, b_var, b_median]
                    )
            result = train_func(*args, **kwargs)

            algo_state, metrics = result
            params = algo_state.runner_state.train_state.params["params"]
            params_stats = get_stats(params)

            if isinstance(metrics.metrics, DQNMetrics) and hasattr(algo_state.runner_state.train_state, "target_params"):
                target_params = algo_state.runner_state.train_state.target_params["params"]
                target_params_stats = get_stats(target_params)
                params_stats = np.concatenate((params_stats, target_params_stats))
            elif isinstance(metrics.metrics, SACMetrics):
                for state in ["actor_train_state", "critic_train_state", "alpha_train_state"]:
                    if hasattr(algo_state.runner_state, state):
                        state_params = getattr(algo_state.runner_state, state)["params"]
                        state_params_stats = get_stats(state_params)
                        params_stats = np.concatenate((params_stats, state_params_stats))
                        if hasattr(getattr(algo_state.runner_state, state), "target_params"):
                            target_state_params = getattr(algo_state.runner_state, state).target_params["params"]
                            target_state_params_stats = get_stats(target_state_params)
                            params_stats = np.concatenate((params_stats, target_state_params_stats))
            state_features[WeightInfo.KEY] = params_stats

            return result

        return wrapper

    @staticmethod
    def get_state_space() -> gymnasium.spaces.Space:
        """Returns state space."""
        # FIXME: This is technically not correct, but we don't know the exact size beforehand.
        # We could in principle pad this, but e.g. SAC is 4x larger than PPO, so that's also not ideal.
        return gymnasium.spaces.Box(
            low=np.ones(6)*-np.inf, high=np.ones(6)*np.inf)

class PredictionInfo(StateFeature):
    """Prediction information state feature for the AutoRL environment. It contains the predicted values and log probs."""

    KEY = "prediction_info"

    @staticmethod
    def __call__(train_func: TrainFunc, state_features: dict) -> TrainFunc:
        """Wraps the training function with the gradient information calculation."""
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)

            _, train_result = result
            trajectory = train_result.trajectories
            if isinstance(train_result.metrics, DQNMetrics):
                td_error = train_result.metrics.td_error
                value_mean = jnp.mean(td_error)
                value_var = jnp.var(td_error)
                log_probs_mean = 0
                log_probs_var = 0
            elif isinstance(train_result.metrics, PPOMetrics):
                log_probs = trajectory.log_probs
                value = train_result.value
                value_mean = jnp.mean(value)
                value_var = jnp.var(value)
                log_probs_mean = jnp.mean(log_probs)
                log_probs_var = jnp.var(log_probs)

            state_features[PredictionInfo.KEY] = np.array([value_mean, value_var, log_probs_mean, log_probs_var])

            return result

        return wrapper

    @staticmethod
    def get_state_space() -> gymnasium.spaces.Space:
        """Returns state space."""
        return gymnasium.spaces.Box(
            low=np.ones(4)*-np.inf, high=np.ones(4)*np.inf)

STATE_FEATURES = {o.KEY: o for o in [GradInfo, LossInfo, WeightInfo, PredictionInfo]}

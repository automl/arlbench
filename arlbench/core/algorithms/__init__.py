"""RL algorithms."""
from collections.abc import Callable

from flashbax.buffers.prioritised_trajectory_buffer import (
    PrioritisedTrajectoryBufferState,
)

from .algorithm import Algorithm
from .dqn import (
    DQN,
    DQNMetrics,
    DQNRunnerState,
    DQNState,
    DQNTrainingResult,
    DQNTrainReturnT,
)
from .ppo import (
    PPO,
    PPOMetrics,
    PPORunnerState,
    PPOState,
    PPOTrainingResult,
    PPOTrainReturnT,
)
from .sac import (
    SAC,
    SACMetrics,
    SACRunnerState,
    SACState,
    SACTrainingResult,
    SACTrainReturnT,
)

TrainResult = DQNTrainingResult | PPOTrainingResult | SACTrainingResult
TrainMetrics = DQNMetrics | PPOMetrics | SACMetrics
RunnerState = DQNRunnerState | PPORunnerState | SACRunnerState
BufferState = PrioritisedTrajectoryBufferState
TrainReturnT = DQNTrainReturnT | PPOTrainReturnT | SACTrainReturnT
TrainFunc = Callable[
    [RunnerState, BufferState, int | None, int | None, int | None], TrainReturnT
]
AlgorithmState = DQNState | PPOState | SACState

__all__ = ["Algorithm", "PPO", "DQN", "SAC"]

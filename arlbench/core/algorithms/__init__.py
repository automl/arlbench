from collections.abc import Callable
from typing import Optional, Union

from flashbax.buffers.prioritised_trajectory_buffer import \
    PrioritisedTrajectoryBufferState

from .algorithm import Algorithm
from .dqn import DQN, DQNRunnerState, DQNTrainingResult, DQNMetrics, DQNTrainReturnT
from .ppo import PPO, PPORunnerState, PPOTrainingResult, PPOMetrics, PPOTrainReturnT
from .sac import SAC, SACRunnerState, SACTrainingResult, SACMetrics, SACTrainReturnT

TrainResult = Union[DQNTrainingResult, PPOTrainingResult, SACTrainingResult]
TrainMetrics = Union[DQNMetrics, PPOMetrics, SACMetrics]
RunnerState = Union[DQNRunnerState, PPORunnerState, SACRunnerState]
BufferState = PrioritisedTrajectoryBufferState
TrainReturnT = Union[DQNTrainReturnT, PPOTrainReturnT, SACTrainReturnT]
TrainFunc = Callable[[RunnerState, BufferState, int | None, int | None, int | None], TrainReturnT]

__all__ = [
    "Algorithm",
    "PPO",
    "DQN",
    "SAC"
]

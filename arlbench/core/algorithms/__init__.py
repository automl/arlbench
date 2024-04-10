from .algorithm import Algorithm
from .dqn import DQN, DQNRunnerState, DQNTrainingResult
from .ppo import PPO, PPORunnerState, PPOTrainingResult
from .sac import SAC, SACRunnerState, SACTrainingResult
from typing import Union, Callable, Optional
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState

TrainResult = Union[DQNTrainingResult, PPOTrainingResult, SACTrainingResult]
RunnerState = Union[DQNRunnerState, PPORunnerState, SACRunnerState]
BufferState = PrioritisedTrajectoryBufferState
TrainFunc = Callable[[RunnerState, BufferState, Optional[int], Optional[int], Optional[int]], TrainResult]


__all__ = [
    "Algorithm",
    "PPO",
    "DQN",
    "SAC"
]

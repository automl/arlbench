from abc import ABC, abstractmethod
from typing import Tuple, Any
from chex import PRNGKey


class AutoRLEnv(ABC):
    def __init__(self, env: Any, n_envs: int):
        self.env = env
        self.n_envs = n_envs

    @abstractmethod
    def reset(self, rng: PRNGKey) -> Tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod
    def step(self, env_state: Any, action: Any, rng: PRNGKey) -> Tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod 
    def action_space(self):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self):
        raise NotImplementedError
    

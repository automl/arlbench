from abc import ABC, abstractmethod
from typing import Tuple, Any


class Environment(ABC):
    def __init__(self, env, n_envs):
        self.env = env
        self.n_envs = n_envs

    @abstractmethod
    def reset(self, rng) -> Tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod
    def step(self, env_state, action, rng) -> Tuple[Any, Any]:    # TODO improve typing
        raise NotImplementedError

    @abstractmethod 
    def action_space(self):
        raise NotImplementedError
    
    @abstractmethod 
    def sample_action(self, rng):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self):
        raise NotImplementedError
    

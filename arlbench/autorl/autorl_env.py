from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, Union

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from .checkpointing import Checkpointer
from .objectives import OBJECTIVES
from ..core.algorithms import (DQN, PPO, SAC, Algorithm, DQNRunnerState,
                                      PPORunnerState, SACRunnerState)
from ..utils import config_space_to_gymnasium_space

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from flashbax.buffers.prioritised_trajectory_buffer import \
        PrioritisedTrajectoryBufferState


class AutoRLEnv(gymnasium.Env):
    ALGORITHMS = {"ppo": PPO, "dqn": DQN, "sac": SAC}
    algorithm: Optional[Algorithm]
    get_obs: Callable[[], np.ndarray]
    runner_state: Optional[Union[PPORunnerState, DQNRunnerState, SACRunnerState]]
    buffer_state: Optional[PrioritisedTrajectoryBufferState]
    metrics: Optional[tuple]
    c_hp_config: Configuration

    def __init__(
            self,
            config,
            envs
        ) -> None:
        super().__init__()

        self.config = config
        self.seed = int(config["seed"])

        self.done = True
        self.c_step = 0     # current step
        self.c_episode = 0

        # Nested environments
        self.envs = envs
        self.c_env_id = 0   # TODO improve
        self.c_env = self.envs[self.c_env_id]["env"]
        self.c_env_options = self.envs[self.c_env_id]["env_options"]

        # Algorithm
        self.algorithm_cls = self.ALGORITHMS[config["algorithm"]]
        self.algorithm = None
        self.runner_state = None
        self.buffer_state = None
        self.metrics = None

        # Checkpointing
        self._checkpoints = []
        # TODO simplify this
        self.track_metrics = self.config["grad_obs"] or "loss" in config["checkpoint"] or "extras" in config["checkpoint"]
        self.track_trajectories = "minibatches" in config["checkpoint"] or "trajectories" in config["checkpoint"]
        
        # Objectives
        if len(config["objectives"]) == 0:
            raise ValueError("Please select at least one optimization objective.")
        
        self._objectives = []
        for o in config["objectives"]:
            if o not in OBJECTIVES.keys():
                raise ValueError(f"Invalid objective: {o}")
            if o == "reward" and "reward_eval_episodes" not in self.config:
                raise ValueError("Reward objective is selected, but 'reward_eval_episodes' is not defined in config.")
            self._objectives += [OBJECTIVES[o]]

            # Ensure right order of objectives, e.g. runtime is wrapped first
            self._objectives = sorted(self._objectives)

        # Define observation method and init observation space
        if "grad_obs" in config.keys() and config["grad_obs"]:
            self.get_obs = self.get_gradient_obs
            self.get_obs_space = self.get_gradient_obs_space
        else:
            self.get_obs = self.get_default_obs
            self.get_obs_space = self.get_default_obs_space

    def get_default_obs(self) -> np.ndarray:
        return np.array([self.c_step, self.c_step * self.c_env_options["n_total_timesteps"]])

    def get_default_obs_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([np.inf, np.inf]), 
            seed=self.seed
        )

    def get_gradient_obs(self) -> np.ndarray:
        if self.c_step == 0:
            grad_norm = 0
            grad_var = 0
        else:
            if self.metrics and len(self.metrics) >= 3:
                grad_info = self.metrics[1]
            else:
                raise ValueError(f"Tying to extract grad_info but 'self.metrics' does not match.")
    
            grad_info = grad_info["params"]
            grad_info = {
                k: v
                for (k, v) in grad_info.items()
                if isinstance(v, dict)
            }
            
            grad_info = [
                grad_info[g][k] for g in grad_info.keys() for k in grad_info[g].keys()
            ]

            grad_norm = np.mean([jnp.linalg.norm(g) for g in grad_info])
            grad_var = np.mean([jnp.var(g) for g in grad_info])
        return np.array(
            [
                self.c_step,
                self.c_step * self.c_env_options["n_total_timesteps"],
                grad_norm,
                grad_var,
            ]
        )

    def get_gradient_obs_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf]), 
            high=np.array([np.inf, np.inf, np.inf, np.inf]), 
            seed=self.seed
        )
    
    def step_(self):
        self.c_step += 1
        if self.c_step >= self.config["n_steps"]:
            return True
        return False
    
    def _instantiate_algorithm(self):
        self.algorithm = self.algorithm_cls(
            self.c_hp_config,
            self.c_env_options,
            self.c_env,
            track_trajectories=self.track_trajectories,
            track_metrics=self.track_metrics
        )

    def _initialize_algorithm(self, **init_kw_args):
        # Instantiate algorithm
        self._instantiate_algorithm()
        
        # Initialize algorithm (runner state and buffer) if static HPO or first step in DAC
        if self.runner_state is None and self.buffer_state is None:  # equal to c_step == 0
            rng = jax.random.PRNGKey(self.seed)
            rng, init_rng = jax.random.split(rng)
            self.runner_state, self.buffer_state = self.algorithm.init(init_rng, **init_kw_args)
    
    def _train(self):
        if self.algorithm is None:
            raise ValueError("You have to instantiate self.algorithm first.")
        
        objectives = {}     # result are stored here
        train_func = self.algorithm.train

        for o in self._objectives:
            train_func = o(train_func, objectives, self)

        # Track configuration + budgets using deepcave (https://github.com/automl/DeepCAVE)
        # TODO test
        if "deep_cave" in self.config.keys() and self.config["deep_cave"]:
            from deepcave import Recorder, Objective

            dc_objectives = [Objective(**o.get_spec()) for o in self._objectives]

            with Recorder(self.config_space, objectives=dc_objectives) as r:
                r.start(self.c_hp_config, self.c_env_options["n_total_timesteps"])
                (self.runner_state, self.buffer_state), self.metrics = train_func(self.runner_state, self.buffer_state)
                  
                r.end(costs=[objectives[o.name] for o in self._objectives])
        else:
            (self.runner_state, self.buffer_state), self.metrics = train_func(self.runner_state, self.buffer_state)
        return objectives

    def step(
        self,
        action: Configuration
    ) -> tuple[np.ndarray, dict[str, float], bool, bool, dict[str, Any]]:
        if len(action.keys()) == 0:     # no action provided
            warnings.warn("No agent configuration provided. Falling back to default configuration.")

        if self.done or self.algorithm is None:
            raise ValueError("Called step() before reset().")
        
        # Set done if max. number of steps in DAC is reached or classic (one-step) HPO is performed
        self.done = self.step_()
        info = {}
        
        # Apply changes to current hyperparameter configuration
        self.c_hp_config = action

        # Instantiate algorithm and apply current hyperparameter configuration
        self._initialize_algorithm()

        # Perform one iteration of training
        objectives = self._train()
        
        # Checkpointing
        if len(self.config["checkpoint"]) > 0:
            assert self.runner_state is not None
            assert self.buffer_state is not None

            checkpoint = self.save()
            self._checkpoints += [checkpoint]
            info["checkpoint"] = checkpoint
            
        return self.get_obs(), objectives, self.done, False, info
    
    def reset(
        self,
        seed: int | None = None,
        checkpoint_path: Optional[str] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:  # type: ignore
        self.done = False
        self.c_step = 0
        self.c_episode += 1
        self.runner_state = None
        self.buffer_state = None

        # Use default hyperparameter configuration
        self.c_hp_config = dict(self.algorithm_cls.get_default_hpo_config())
        self._instantiate_algorithm()

        if checkpoint_path:
            self.load(checkpoint_path)

        return self.get_obs(), {}
    
    def save(self, tag: Optional[str] = None) -> str:
        if self.runner_state is None or self.buffer_state is None:
            raise ValueError("Agent not initialized. Not able to save agent state.")
        
        return Checkpointer.save(
            self.runner_state,
            self.buffer_state,
            self.config,
            self.c_hp_config,
            self.done,
            self.c_episode,
            self.c_step,
            self.metrics,
            tag=tag
        )
    
    def load(self, checkpoint_path: str) -> None:
        _, dummy_buffer_state = self.algorithm.init(jax.random.PRNGKey(42))
        (
            self.c_hp_config,
            self.c_step,
            self.c_episode, 
        ), algorithm_kw_args = Checkpointer.load(checkpoint_path, self.config["algorithm"], dummy_buffer_state)

        self._initialize_algorithm(**algorithm_kw_args)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return config_space_to_gymnasium_space(self.algorithm_cls.get_hpo_config_space())
    
    @property
    def config_space(self) -> ConfigurationSpace:
        return self.algorithm_cls.get_hpo_config_space()
 
    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return self.get_obs_space()

    @property
    def checkpoints(self) -> list[str]:
        return list(self._checkpoints)
    
    @property
    def objectives(self) -> list[str]:
        return [o.__name__ for o in self._objectives]

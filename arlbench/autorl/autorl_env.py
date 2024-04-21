from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import gymnasium
import jax
import numpy as np

from arlbench.core.algorithms import (DQN, PPO, SAC, Algorithm, AlgorithmState,
                                      TrainResult, TrainReturnT)
from arlbench.core.environments import make_env
from arlbench.utils import config_space_to_gymnasium_space

from .checkpointing import Checkpointer
from .objectives import OBJECTIVES
from .state_features import STATE_FEATURES

ObservationT = dict[str, np.ndarray]
ObjectivesT = dict[str, float]
InfoT = dict[str, Any]

from ConfigSpace import Configuration, ConfigurationSpace

DEFAULT_AUTO_RL_CONFIG = {
    "seed": 42,
    "env_framework": "gymnax",
    "env_name": "CartPole-v1",
    "n_envs": 10,
    "algorithm": "dqn",
    "cnn_policy": False,
    "checkpoint": [],
    "checkpoint_name": "default_checkpoint",
    "checkpoint_dir": "/tmp",
    "objectives": ["reward_mean"],
    "optimize_objectives": "upper",
    "state_features": ["grad_info"],
    "n_steps": 10,
    "n_total_timesteps": 1e5,
    "n_eval_steps": 100,
    "n_eval_episodes": 10,
}


class AutoRLEnv(gymnasium.Env):
    ALGORITHMS = {"ppo": PPO, "dqn": DQN, "sac": SAC}
    _algorithm: Algorithm
    _get_obs: Callable[[], np.ndarray]
    _algorithm_state: AlgorithmState | None
    _train_result: TrainResult | None
    _hpo_config: Configuration
    _total_timesteps: int

    def __init__(
            self,
            config: dict | None = None
        ) -> None:
        super().__init__()

        self._config = DEFAULT_AUTO_RL_CONFIG.copy()

        if config:
            for k, v in config.items():
                if k in DEFAULT_AUTO_RL_CONFIG:
                    self._config[k] = v
                else:
                    warnings.warn(f"Invalid config key '{k}'. Falling back to default value.")

        self._seed = int(self._config["seed"])

        self._done = True
        self._total_timesteps = 0   # timesteps across calls of step()
        self._c_step = 0            # current step
        self._c_episode = 0         # current episode

        # Environment
        self._env = make_env(
            self._config["env_framework"],
            self._config["env_name"],
            cnn_policy=self._config["cnn_policy"],
            n_envs=self._config["n_envs"],
            seed=self._seed
        )

        # Algorithm
        self._algorithm_cls = self.ALGORITHMS[self._config["algorithm"]]
        self._config_space = self._algorithm_cls.get_hpo_config_space()

        # instantiate dummy algorithm
        self._algorithm = self._algorithm_cls(
            self._algorithm_cls.get_default_hpo_config(),
            self._env
        )
        self._algorithm_state = None
        self._train_result = None

        # Checkpointing
        self._checkpoints = []
        self._track_metrics = "all" in self._config["checkpoint"] or  "grad_info" in self._config["state_features"] or "loss" in self._config["checkpoint"]
        self._track_trajectories = "all" in self._config["checkpoint"] or "trajectories" in self._config["checkpoint"]

        # Objectives
        if len(self._config["objectives"]) == 0:
            raise ValueError("Please select at least one optimization objective.")

        self._objectives = []
        objectives = list(set(self._config["objectives"]))
        for o in objectives:
            if o not in OBJECTIVES:
                raise ValueError(f"Invalid objective: {o}")
            self._objectives += [OBJECTIVES[o]]

        # Ensure right order of objectives, e.g. runtime is wrapped first
        self._objectives = sorted(self._objectives, key=lambda o: o[1])

        # Extract online objective classes
        self._objectives = [o[0] for o in self._objectives]

        # State Features
        self._state_features = []
        state_features = list(set(self._config["state_features"]))
        for f in state_features:
            if f not in STATE_FEATURES:
                raise ValueError(f"Invalid state feature: {f}")
            self._state_features += [STATE_FEATURES[f]]

        self._observation_space = self._get_obs_space()

    def _get_obs_space(self) -> gymnasium.spaces.Dict:
        obs_space = {
            f.KEY: f.get_state_space() for f in self._state_features
        }

        obs_space["steps"] = gymnasium.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([np.inf, np.inf])
        )

        return gymnasium.spaces.Dict(obs_space)

    def step_(self):
        self._c_step += 1
        if self._c_step >= self._config["n_steps"]:
            return True
        return False
    
    def _make_algorithm(self):
        return self._algorithm_cls(
            self._hpo_config,
            self._env,
            track_trajectories=self._track_trajectories,
            track_metrics=self._track_metrics
        )

    def _train(
            self,
            **kw_args
        ) -> tuple[TrainReturnT, dict, dict]:
        assert self._algorithm_state is not None

        objectives = {}     # result are stored here
        obs = {}            # state features are stored here
        train_func = self._algorithm.train

        for o in self._objectives:
            train_func = o(train_func, objectives, self._config["optimize_objectives"])

        obs["steps"] = np.array([self._c_step, self._total_timesteps])
        for f in self._state_features:
            train_func = f(train_func, obs)

        # Track configuration + budgets using deepcave (https://github.com/automl/DeepCAVE)
        # TODO test
        if self._config.get("deep_cave"):
            from deepcave import Objective, Recorder

            dc_objectives = [Objective(**o.get_spec()) for o in self._objectives]

            with Recorder(self._config_space, objectives=dc_objectives) as r:
                r.start(self._hpo_config, self._config["n_total_timesteps"])
                result = train_func(
                    *self._algorithm_state,
                    **kw_args
                )

                r.end(costs=[objectives[o.name] for o in self._objectives])
        else:
            result = train_func(
                *self._algorithm_state,
                **kw_args
            )
        return result, objectives, obs

    def step(
        self,
        action: Configuration | dict,
        n_total_timesteps: int | None = None,
        n_eval_steps: int | None = None,
        n_eval_episodes: int | None = None,
        seed: int | None = None
    ) -> tuple[ObservationT, ObjectivesT, bool, bool, InfoT]:
        if len(action.keys()) == 0:     # no action provided
            warnings.warn("No agent configuration provided. Falling back to default configuration.")

        if self._done or self._algorithm is None or self._algorithm_state is None:
            raise ValueError("Called step() before reset().")

        # Set done if max. number of steps in DAC is reached or classic (one-step) HPO is performed
        self._done = self.step_()
        info = {}

        # Apply changes to current hyperparameter configuration and reinstantiate algorithm
        if isinstance(action, dict):
            action = Configuration(self.config_space, action)
        self._hpo_config = action

        # Update hyperparameter configuration
        self._algorithm = self._make_algorithm()


        if seed:
            print(self._algorithm_state)
            runner_state = self._algorithm_state._replace(rng=jax.random.key(seed))
            self._algorithm_state = self._algorithm_state._replace(runner_state=runner_state)

        # Training kwargs
        train_kw_args = {
            "n_total_timesteps": n_total_timesteps if n_total_timesteps else self._config["n_total_timesteps"],
            "n_eval_steps": n_eval_steps if n_eval_steps else self._config["n_eval_steps"],
            "n_eval_episodes": n_eval_episodes if n_eval_episodes else self._config["n_eval_episodes"]
        }

        # Perform one iteration of training
        result, objectives, obs = self._train(**train_kw_args)
        self._algorithm_state, self._train_result = result

        self._total_timesteps += train_kw_args["n_total_timesteps"]

        # Checkpointing
        if len(self._config["checkpoint"]) > 0:
            assert self._algorithm_state is not None

            checkpoint = self._save()
            self._checkpoints += [checkpoint]
            info["checkpoint"] = checkpoint

        return obs, objectives, False, self._done, info

    def reset(
        self,
        seed: int | None = None,
        checkpoint_path: str | None = None,
    ) -> tuple[ObservationT, InfoT]:  # type: ignore
        self._done = False
        self._c_step = 0
        self._c_episode += 1

        seed = seed if seed else self._seed

        # Use default hyperparameter configuration
        self._hpo_config = self._algorithm_cls.get_default_hpo_config()
        self._algorithm = self._make_algorithm()

        if checkpoint_path:
            self._load(checkpoint_path, seed)
        else:
            init_rng = jax.random.key(seed)
            self._algorithm_state = self._algorithm.init(init_rng)

        return {}, {}

    def _save(self, tag: str | None = None) -> str:
        if self._algorithm_state is None:
            warnings.warn("Agent not initialized. Not able to save agent state.")
            return ""

        if self._train_result is None:
            warnings.warn("No training performed, so there is nothing to save. Please run step() first.")
            return ""

        return Checkpointer.save(
            self._algorithm.name,
            self._algorithm_state,
            self._config,
            self._hpo_config,
            self._done,
            self._c_episode,
            self._c_step,
            self._train_result,
            tag=tag
        )

    def _load(self, checkpoint_path: str, seed: int) -> None:
        init_rng = jax.random.PRNGKey(seed)
        algorithm_state = self._algorithm.init(init_rng)
        (
            hpo_config,
            self._c_step,
            self._c_episode,
        ), algorithm_kw_args = Checkpointer.load(checkpoint_path, algorithm_state)
        self._hpo_config = Configuration(self._config_space, hpo_config)

        self._algorithm_state = self._algorithm.init(init_rng, **algorithm_kw_args)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return config_space_to_gymnasium_space(self._config_space)

    @property
    def config_space(self) -> ConfigurationSpace:
        return self._config_space

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return self._observation_space

    @property
    def hpo_config(self) -> Configuration:
        return self._hpo_config

    @property
    def checkpoints(self) -> list[str]:
        return list(self._checkpoints)

    @property
    def objectives(self) -> list[str]:
        return [o.__name__ for o in self._objectives]

    @property
    def config(self) -> dict:
        return self._config.copy()

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    def eval(self, num_eval_episodes) -> np.ndarray:
        if self._algorithm_state is None:
            raise ValueError("Agent not initialized. Not able to evaluate agent.")
        rewards = self._algorithm.eval(self._algorithm_state.runner_state, num_eval_episodes)
        return np.array(rewards)
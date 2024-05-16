from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import gymnasium
import jax
import numpy as np
import pandas as pd

from arlbench.core.algorithms import (
    DQN,
    PPO,
    SAC,
    Algorithm,
    AlgorithmState,
    TrainResult,
    TrainReturnT,
)
from arlbench.core.environments import make_env
from arlbench.utils import config_space_to_gymnasium_space

from .checkpointing import Checkpointer
from .objectives import OBJECTIVES, Objective
from .state_features import STATE_FEATURES, StateFeature

ObservationT = dict[str, np.ndarray]
ObjectivesT = dict[str, float]
InfoT = dict[str, Any]

from ConfigSpace import Configuration, ConfigurationSpace

DEFAULT_AUTO_RL_CONFIG = {
    "seed": 42,
    "env_framework": "gymnax",
    "env_name": "CartPole-v1",
    "env_kwargs": {},
    "eval_env_kwargs": {},
    "n_envs": 10,
    "algorithm": "dqn",
    "cnn_policy": False,
    "deterministic_eval": True,
    "nas_config": {},
    "checkpoint": [],
    "checkpoint_name": "default_checkpoint",
    "checkpoint_dir": "/tmp",
    "objectives": ["reward_mean"],
    "optimize_objectives": "upper",
    "state_features": [],
    "n_steps": 10,
    "n_total_timesteps": 1e5,
    "n_eval_steps": 100,
    "n_eval_episodes": 10,
}


class AutoRLEnv(gymnasium.Env):
    """
    Automated Reinforcement Learning (gynmasium-like) Environment.

    With each reset, the algorithm state is (re-)initialized.
    If a checkpoint path is passed to reset, the agent state is initialized with the checkpointed state.

    In each step, one iteration of training is performed with the current hyperparameter configuration (= action).
    """

    ALGORITHMS = {"ppo": PPO, "dqn": DQN, "sac": SAC}
    _algorithm: Algorithm
    _get_obs: Callable[[], np.ndarray]
    _algorithm_state: AlgorithmState | None
    _train_result: TrainResult | None
    _hpo_config: Configuration
    _total_training_steps: int

    def __init__(self, config: dict | None = None) -> None:
        """_summary_

        Args:
            config (dict | None, optional): _description_. Defaults to None.
        """
        super().__init__()

        self._config = DEFAULT_AUTO_RL_CONFIG.copy()

        if config:
            for k, v in config.items():
                if k in DEFAULT_AUTO_RL_CONFIG:
                    self._config[k] = v
                else:
                    warnings.warn(
                        f"Invalid config key '{k}'. This item will be ignored."
                    )

        self._seed = int(self._config["seed"])

        self._done = True
        self._total_training_steps = 0  # timesteps across calls of step()
        self._c_step = 0  # current step
        self._c_episode = 0  # current episode

        # Environment
        self._env = make_env(
            self._config["env_framework"],
            self._config["env_name"],
            n_envs=self._config["n_envs"],
            env_kwargs=self._config["env_kwargs"],
            cnn_policy=self._config["cnn_policy"],
            seed=self._seed,
        )

        self._eval_env = make_env(
            self._config["env_framework"],
            self._config["env_name"],
            n_envs=self._config["n_envs"],
            env_kwargs=self._config["eval_env_kwargs"],
            cnn_policy=self._config["cnn_policy"],
            seed=self._seed + 1,
        )

        # Checkpointing
        self._checkpoints = []
        self._track_metrics = (
            "all" in self._config["checkpoint"]
            or "grad_info" in self._config["state_features"]
            or "loss" in self._config["checkpoint"]
        )
        self._track_trajectories = (
            "all" in self._config["checkpoint"]
            or "trajectories" in self._config["checkpoint"]
        )

        # Algorithm
        self._algorithm_cls = self.ALGORITHMS[self._config["algorithm"]]
        self._config_space = self._algorithm_cls.get_hpo_config_space()

        # Instantiate algorithm with default hyperparameter configuration
        self._nas_config = self._algorithm_cls.get_default_nas_config()
        for k, v in self._config["nas_config"].items():
            self._nas_config[k] = v
        self._hpo_config = self._algorithm_cls.get_default_hpo_config()
        self._algorithm = self._make_algorithm()
        self._algorithm_state = None
        self._train_result = None

        # Optimization objectives
        self._objectives = self._get_objectives()

        # State Features
        self._state_features = self._get_state_features()

        self._observation_space = self._get_obs_space()

    def _get_objectives(self) -> list[Objective]:
        """_summary_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            list[Objective]: _description_
        """
        if len(self._config["objectives"]) == 0:
            raise ValueError("Please select at least one optimization objective.")

        objectives = []
        cfg_objectives = list(set(self._config["objectives"]))
        for o in cfg_objectives:
            if o not in OBJECTIVES:
                raise ValueError(f"Invalid objective: {o}")
            objectives += [OBJECTIVES[o]]

        # Ensure right order of objectives, e.g. runtime is wrapped first
        objectives = sorted(objectives, key=lambda o: o[1])

        # Extract online objective classes
        objectives = [o[0] for o in objectives]

        return objectives

    def _get_state_features(self) -> list[StateFeature]:
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            list[StateFeature]: _description_
        """
        state_features = []
        cfg_state_features = list(set(self._config["state_features"]))
        for f in cfg_state_features:
            if f not in STATE_FEATURES:
                raise ValueError(f"Invalid state feature: {f}")
            state_features += [STATE_FEATURES[f]]
        return state_features

    def _get_obs_space(self) -> gymnasium.spaces.Dict:
        """_summary_

        Returns:
            gymnasium.spaces.Dict: _description_
        """
        obs_space = {f.KEY: f.get_state_space() for f in self._state_features}

        obs_space["steps"] = gymnasium.spaces.Box(
            low=np.array([0, 0]), high=np.array([np.inf, np.inf])
        )

        return gymnasium.spaces.Dict(obs_space)

    def _step(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        self._c_step += 1
        if self._c_step >= self._config["n_steps"]:
            return True
        return False

    def _train(self, **train_kw_args) -> tuple[TrainReturnT, dict, dict]:
        """_summary_

        Returns:
            tuple[TrainReturnT, dict, dict]: _description_
        """
        assert self._algorithm_state is not None

        objectives = {}  # result are stored here
        obs = {}  # state features are stored here
        train_func = self._algorithm.train

        for o in self._objectives:
            train_func = o(train_func, objectives, self._config["optimize_objectives"])

        obs["steps"] = np.array([self._c_step, self._total_training_steps])
        for f in self._state_features:
            train_func = f(train_func, obs)

        # Track configuration + budgets using deepcave (https://github.com/automl/DeepCAVE)
        # TODO test
        if self._config.get("deep_cave"):
            from deepcave import Objective, Recorder

            dc_objectives = [Objective(**o.get_spec()) for o in self._objectives]

            with Recorder(self._config_space, objectives=dc_objectives) as r:
                r.start(self._hpo_config, self._config["n_total_timesteps"])
                result = train_func(*self._algorithm_state, **train_kw_args)

                r.end(costs=[objectives[o.KEY] for o in self._objectives])
        else:
            result = train_func(*self._algorithm_state, **train_kw_args)
        return result, objectives, obs

    def _make_algorithm(self) -> Algorithm:
        return self._algorithm_cls(
            self._hpo_config,
            self._env,
            nas_config=self._nas_config,
            eval_env=self._eval_env,
            track_metrics=self._track_metrics,
            track_trajectories=self._track_trajectories,
            cnn_policy=self._config["cnn_policy"],
            deterministic_eval=self._config["deterministic_eval"],
        )

    def step(
        self,
        action: Configuration | dict,
        checkpoint_path: str | None = None,
        n_total_timesteps: int | None = None,
        n_eval_steps: int | None = None,
        n_eval_episodes: int | None = None,
        seed: int | None = None,
    ) -> tuple[ObservationT, ObjectivesT, bool, bool, InfoT]:
        """_summary_

        Args:
            action (Configuration | dict): _description_
            n_total_timesteps (int | None, optional): _description_. Defaults to None.
            n_eval_steps (int | None, optional): _description_. Defaults to None.
            n_eval_episodes (int | None, optional): _description_. Defaults to None.
            seed (int | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: Error is raised if step() is called before reset() was called.

        Returns:
            tuple[ObservationT, ObjectivesT, bool, bool, InfoT]: _description_
        """
        if len(action.keys()) == 0:  # no action provided
            warnings.warn(
                "No agent configuration provided. Falling back to default configuration."
            )

        if self._done:
            raise ValueError("Called step() before reset().")
        
        # Set done if max. number of steps in DAC is reached or classic (one-step) HPO is performed
        self._done = self._step()
        info = {}
        
        # Apply changes to current hyperparameter configuration and reinstantiate algorithm
        if isinstance(action, dict):
            try:
                action = Configuration(self.config_space, action)
            except Exception as _:
                warnings.warn("Config does not match configuration space. Using dict config instead.")
        self._hpo_config = action

        seed = seed if seed else self._seed
        
        self._algorithm = self._make_algorithm()
        if checkpoint_path:
            print("#### LOADING ####")
            print("checkpoint_path:", checkpoint_path)
            try:
                self._algorithm_state = self._load(checkpoint_path, seed)
                print("#### LOADING successful ####")
            except Exception as e:
                print (e)
                init_rng = jax.random.key(seed)
                self._algorithm_state = self._algorithm.init(init_rng)
        else:
            init_rng = jax.random.key(seed)
            self._algorithm_state = self._algorithm.init(init_rng)

        # Training kwargs
        train_kw_args = {
            "n_total_timesteps": n_total_timesteps
            if n_total_timesteps
            else self._config["n_total_timesteps"],
            "n_eval_steps": n_eval_steps
            if n_eval_steps
            else self._config["n_eval_steps"],
            "n_eval_episodes": n_eval_episodes
            if n_eval_episodes
            else self._config["n_eval_episodes"],
        }

        # Perform one iteration of training
        result, objectives, obs = self._train(**train_kw_args)
        self._algorithm_state, self._train_result = result

        steps = (
            np.arange(1, train_kw_args["n_eval_steps"] + 1) * train_kw_args["n_total_timesteps"] // train_kw_args["n_eval_steps"]
        )
        returns = self._train_result.eval_rewards.mean(axis=1)

        info["train_info_df"] = pd.DataFrame({"steps": steps, "returns": returns})

        self._total_training_steps += train_kw_args["n_total_timesteps"]

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
    ) -> tuple[ObservationT, InfoT]:
        """_summary_

        Args:
            seed (int | None, optional): _description_. Defaults to None.
            checkpoint_path (str | None, optional): _description_. Defaults to None.

        Returns:
            tuple[ObservationT, InfoT]: _description_
        """
        self._done = False
        self._c_step = 0
        self._c_episode += 1

        return {}, {}

    def _save(self, tag: str | None = None) -> str:
        """_summary_

        Args:
            tag (str | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            str: _description_
        """
        if self._algorithm_state is None:
            warnings.warn("Agent not initialized. Not able to save agent state.")
            return ""

        if self._train_result is None:
            warnings.warn("No training performed, so there is nothing to save. Please run step() first.")

        return Checkpointer.save(
            self._algorithm.name,
            self._algorithm_state,
            self._config,
            self._hpo_config,
            self._done,
            self._c_episode,
            self._c_step,
            self._train_result,
            tag=tag,
        )

    def _load(self, checkpoint_path: str, seed: int) -> AlgorithmState:
        """_summary_

        Args:
            checkpoint_path (str): _description_
            seed (int | None, optional): _description_. Defaults to None.
        """
        init_rng = jax.random.PRNGKey(seed)
        algorithm_state = self._algorithm.init(init_rng)
        (
            (
                hpo_config,
                self._c_step,
                self._c_episode,
            ),
            algorithm_kw_args,
        ) = Checkpointer.load(checkpoint_path, algorithm_state)
        # hpo_config = Configuration(self._config_space, hpo_config)
        algorithm_state = self._algorithm.init(init_rng, **algorithm_kw_args)

        return algorithm_state

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        """_summary_

        Returns:
            gymnasium.spaces.Space: _description_
        """
        return config_space_to_gymnasium_space(self._config_space)

    @property
    def config_space(self) -> ConfigurationSpace:
        """_summary_

        Returns:
            ConfigurationSpace: _description_
        """
        return self._config_space

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        """_summary_

        Returns:
            gymnasium.spaces.Space: _description_
        """
        return self._observation_space

    @property
    def hpo_config(self) -> Configuration:
        """_summary_

        Returns:
            Configuration: _description_
        """
        return self._hpo_config

    @property
    def checkpoints(self) -> list[str]:
        """_summary_

        Returns:
            list[str]: _description_
        """
        return list(self._checkpoints)

    @property
    def objectives(self) -> list[str]:
        """_summary_

        Returns:
            list[str]: _description_
        """
        return [o.__name__ for o in self._objectives]

    @property
    def config(self) -> dict:
        """_summary_

        Returns:
            dict: _description_
        """
        return self._config.copy()

    def eval(self, num_eval_episodes: int) -> np.ndarray:
        """_summary_

        Args:
            num_eval_episodes (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            np.ndarray: _description_
        """
        if self._algorithm is None or self._algorithm_state is None:
            raise ValueError("Agent not initialized. Call reset() first.")
        rewards = self._algorithm.eval(
            self._algorithm_state.runner_state, num_eval_episodes
        )
        return np.array(rewards)

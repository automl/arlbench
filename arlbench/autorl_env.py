import numpy as np
import jax.numpy as jnp
import jax
from typing import Optional, Union, Any, Dict, Callable
from arlbench.algorithms import PPO, DQN, Algorithm, PPORunnerState, DQNRunnerState
import gymnasium
from arlbench.utils import config_space_to_gymnasium_space
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState
from ConfigSpace import Configuration
import warnings
from arlbench.utils.checkpointing import Checkpointer
from arlbench.objectives import track_emissions, track_reward, track_runtime


class AutoRLEnv(gymnasium.Env):
    ALGORITHMS = {"ppo": PPO, "dqn": DQN}
    algorithm: Optional[Algorithm]
    get_obs: Callable[[], np.ndarray]
    get_obs_space: Callable[[], gymnasium.spaces.Space]
    runner_state: Optional[Union[PPORunnerState, DQNRunnerState]]
    buffer_state: Optional[PrioritisedTrajectoryBufferState]
    metrics: Optional[tuple]

    def __init__(self, config, envs) -> None:
        super().__init__()

        self.config = config
        self.seed = int(config["seed"])

        # instances = environments
        self.done = True
        self.c_step = 0     # current step
        self.c_episode = 0
        self.envs = envs
        self.c_env_id = 0   # TODO improve
        self.c_env = self.envs[self.c_env_id]["env"]
        self.c_env_params = self.envs[self.c_env_id]["env_params"]
        self.c_env_options = self.envs[self.c_env_id]["env_options"]

        # init action space
        self.algorithm_cls = self.ALGORITHMS[config["algorithm"]]
        self.action_space = config_space_to_gymnasium_space(self.algorithm_cls.get_hpo_config_space())

        # algorithm state
        self.algorithm = None
        self.runner_state = None
        self.buffer_state = None
        self.metrics = None

        # checkpointing
        self.track_metrics = self.config["grad_obs"] or "loss" in config["checkpoint"] or "extras" in config["checkpoint"]
        self.track_trajectories = "minibatches" in config["checkpoint"] or "trajectories" in config["checkpoint"]
        
        # objectives
        self.objectives = [str(o) for o in config["objectives"]]
        if len(self.objectives) == 0:
            raise ValueError("Please select at least one optimization objective.")

        # define observation method and init observation space
        if "obs_method" in config.keys() and "obs_space_method" in config.keys():
            self.get_obs = config["obs_method"]
            self.get_obs_space = config["obs_space_method"]
        elif "grad_obs" in config.keys() and config["grad_obs"]:
            self.get_obs = self.get_gradient_obs
            self.get_obs_space = self.get_gradient_obs_space
        else:
            self.get_obs = self.get_default_obs
            self.get_obs_space = self.get_default_obs_space
        self.observation_space = self.get_obs_space()

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
    
    def instantiate_algorithm_(self):
        self.algorithm = self.algorithm_cls(
            self.c_hp_config,
            self.c_env_options,
            self.c_env,
            self.c_env_params,
            track_trajectories=self.track_trajectories,
            track_metrics=self.track_metrics
        )

    def initialize_algorithm_(self, **init_kw_args):
        # Instantiate algorithm
        self.instantiate_algorithm_()
        
        # Initialize algorithm (runner state and buffer) if static HPO or first step in DAC
        if self.runner_state is None and self.buffer_state is None:  # equal to c_step == 0
            rng = jax.random.PRNGKey(self.seed)
            rng, init_rng = jax.random.split(rng)
            self.runner_state, self.buffer_state = self.algorithm.init(init_rng, **init_kw_args)
    
    def train_(self):
        if self.algorithm is None:
            raise ValueError("You have to instantiate self.algorithm first.")
        
        objectives = {}
        train_func = self.algorithm.train
              
        if "runtime" in self.objectives:
            train_func = track_runtime(train_func, objectives)
        if "emissions" in self.objectives:
            train_func = track_emissions(train_func, objectives)
        if "reward" in self.objectives:
            train_func = track_reward(train_func, objectives, self.algorithm, self.config["n_eval_episodes"])
        
        (self.runner_state, self.buffer_state), self.metrics = train_func(self.runner_state, self.buffer_state)

        return objectives

    def step(
        self,
        action: Union[Configuration, dict]
    ) -> tuple[np.ndarray, dict[str, float], bool, bool, dict[str, Any]]:
        if len(action.keys()) == 0:     # no action provided
            warnings.warn("No agent configuration provided. Falling back to default configuration.")

        if self.done or self.algorithm is None:
            raise ValueError("Called step() before reset().")
        
        # Set done if max. number of steps in DAC is reached or classic (one-step) HPO is performed
        self.done = self.step_()
        info = {}
        
        # Apply changes to current hyperparameter configuration
        self.c_hp_config.update(action)

        # Instantiate algorithm and apply current hyperparameter configuration
        self.initialize_algorithm_()

        # Perform one iteration of training
        objectives = self.train_()
        
        # Checkpointing
        if len(self.config["checkpoint"]) > 0:
            assert self.runner_state is not None
            assert self.buffer_state is not None

            info["checkpoint"] = Checkpointer.save(
                self.runner_state,
                self.buffer_state,
                self.config,
                self.c_hp_config,
                self.done,
                self.c_episode,
                self.c_step,
                self.metrics
            )
            
        return self.get_obs(), objectives, self.done, False, info
    
    def reset(
        self,
        seed: int | None = None,
        checkpoint_path: str = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:  # type: ignore
        self.done = False
        self.c_step = 0
        self.c_episode += 1
        self.runner_state = None
        self.buffer_state = None

        # Use default hyperparameter configuration
        self.c_hp_config = dict(self.algorithm_cls.get_default_hpo_config())
        self.instantiate_algorithm_()

        if checkpoint_path:
            _, dummy_buffer_state = self.algorithm.init(jax.random.PRNGKey(42))
            (
                self.c_hp_config,
                self.c_step,
                self.c_episode, 
            ), algorithm_kw_args = Checkpointer.load(checkpoint_path, self.config["algorithm"], dummy_buffer_state)

            self.initialize_algorithm_(**algorithm_kw_args)

        return self.get_obs(), {}
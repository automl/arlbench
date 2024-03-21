import numpy as np
import jax.numpy as jnp
import jax
from typing import Optional, Union, Any, Dict, Callable
from arlbench.agents import PPO, DQN, Agent, PPORunnerState, DQNRunnerState
import gymnasium
from arlbench.utils import config_space_to_gymnasium_space
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState
from ConfigSpace import Configuration
import warnings


class AutoRLEnv(gymnasium.Env):
    ALGORITHMS = {"ppo": PPO, "dqn": DQN}
    algorithm: Optional[Agent]
    get_obs: Callable[[], np.ndarray]
    get_obs_space: Callable[[], gymnasium.spaces.Space]
    runner_state: Optional[Union[PPORunnerState, DQNRunnerState]]
    buffer_state: Optional[PrioritisedTrajectoryBufferState]

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
            grad_info = self.grad_info["params"]
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

    def instantiate_algorithm(
            self,
            agent_config
        ) -> Agent:
        return self.algorithm_cls(
            agent_config,
            self.c_env_options,
            self.c_env,
            self.c_env_params,
            track_trajectories=self.config["track_trajectories"],
            track_metrics=self.config["grad_obs"]
        )
    
    def get_algorithm_config(self, action):
        if isinstance(action, Configuration):
            action = dict(action)
        elif isinstance(action, dict):
            # all good
            pass
        else:
            raise ValueError(f"Illegal action type: {type(action)}")
        
        if self.algorithm is None:
            cur_config = dict(self.algorithm_cls.get_default_hpo_config())
        else:
            cur_config = dict(self.algorithm.hpo_config)

        cur_config.update(action)
        return cur_config

    def step(
        self,
        action: Union[Configuration, dict]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done or self.algorithm is None:
            raise ValueError("Called step() before reset().")
        
        if len(action.keys()) == 0:     # no action provided
            warnings.warn("No agent configuration provided. Falling back to default configuration.")

        # Initialize agent (runner state and buffer) if static HPO or first step in DAC
        if self.runner_state is None and self.buffer_state is None:  # equal to c_step == 0
            rng = jax.random.PRNGKey(self.seed)
            rng, init_rng = jax.random.split(rng)
            self.runner_state, self.buffer_state = self.algorithm.init(init_rng)

        # Perform one iteration of training
        (self.runner_state, self.buffer_state), metrics = self.algorithm.train(self.runner_state, self.buffer_state)
        
        if metrics:
            if self.config["track_trajectories"]:
                (
                    self.loss_info,
                    self.grad_info,
                    self.traj,
                    self.additional_info,
                ) = metrics
            elif self.config["grad_obs"]:
                (
                    self.loss_info,
                    self.grad_info,
                    self.additional_info,
                ) = metrics

        reward = self.algorithm.eval(self.runner_state, self.config["n_eval_episodes"])

        self.done = self.step_()
        return self.get_obs(), reward, self.done, False, {}
    
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:  # type: ignore
        self.done = False
        self.c_step = 0
        self.c_episode += 1
        self.runner_state = None
        self.buffer_state = None

        # instantiate default algorithm
        config = self.get_algorithm_config({})
        self.algorithm = self.instantiate_algorithm(config)

        return self.get_obs(), {}
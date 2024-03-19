import numpy as np
import jax.numpy as jnp
import jax
from typing import Optional, Union, Any, Dict, Callable
from arlbench.agents import PPO, DQN, Agent, PPORunnerState, DQNRunnerState
import gymnasium
from arlbench.utils import config_space_to_gymnasium_space


class AutoRLEnv(gymnasium.Env):
    ALGORITHMS = {"ppo": PPO, "dqn": DQN}
    get_obs: Callable[[], np.ndarray]
    get_obs_space: Callable[[], gymnasium.spaces.Space]
    agent_state: Optional[Union[PPORunnerState, DQNRunnerState]]

    def __init__(self, config, envs, seed=None) -> None:
        super().__init__()

        self.config = config
        self.seed = seed

        self.c_step = 0
        self.agent_state = None

        # instances = environments
        self.envs = envs
        self.c_env_id = 0   # TODO improve
        self.c_env = self.envs[self.c_env_id]["env"]
        self.c_env_params = self.envs[self.c_env_id]["env_params"]
        self.c_env_options = self.envs[self.c_env_id]["env_options"]

        # init action space
        self.algorithm_cls = self.ALGORITHMS[config["algorithm"]]
        self.action_space = config_space_to_gymnasium_space(self.algorithm_cls.get_configuration_space())
        
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
            grad_info = self.grad_info

            if self.config["algorithm"] == "ppo":
                import flax

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

    def make_agent(self, agent_config) -> tuple[Agent, Union[PPORunnerState, DQNRunnerState]]:
        agent = self.algorithm_cls(
            agent_config,
            self.c_env_options,
            self.c_env,
            self.c_env_params,
            track_trajectories=self.config["track_trajectories"],
            track_metrics=self.config["grad_obs"]
        )
        agent_rng = jax.random.PRNGKey(self.seed if self.seed else 0)
        runner_state = agent.init(agent_rng)
        return agent, runner_state

    def step(
        self,
        action: Dict
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        done = self.step_()

        agent, runner_state = self.make_agent(action)
        runner_state, metrics = agent.train(runner_state)

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

        reward = agent.eval(runner_state, self.config["n_eval_episodes"])
        return self.get_obs(), reward, done, False, {}
    
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:  # type: ignore
        self.c_step = 0

        return self.get_obs(), {}
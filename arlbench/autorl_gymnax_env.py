import jax
import jax.numpy as jnp
from flax import struct
import chex
from typing import Tuple, Union, Any, Dict
from gymnax.environments.environment import Environment
from algorithms import PPO, DQN
import functools
from gymnax.wrappers.purerl import FlattenObservationWrapper

@struct.dataclass
class ActionConfiguration:
    config: dict

@struct.dataclass
class EnvState:
    c_step: int
    episode: int
    runner_state: tuple
    c_env_id: int
    n_total_timesteps: int

@struct.dataclass
class EnvParams:
    options: dict
    n_steps: int


class AutoRLEnv(Environment):
    ALGORITHMS = {0: PPO, 1: DQN}

    def __init__(self, config, envs, n_envs=10) -> None:
        super().__init__()

        self.algorithm = self.ALGORITHMS[config["algorithm_id"]]
        self.envs = envs
        self.n_envs = n_envs

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        elif "grad_obs" in config.keys() and config["grad_obs"]:
            self.get_state = self.get_gradient_state
        else:
            self.get_state = self.get_default_state
    
    @functools.partial(jax.jit, static_argnums=0)
    def select_next_env(self, c_env_id):
        c_env_id = (c_env_id + 1) % len(self.envs)
    
    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        # TODO implement reset behavior

        state = EnvState(
            c_step=0,
            episode=0,
            runner_state=tuple(),
            c_env_id=0,
            n_total_timesteps=params.options["n_total_timesteps"]
        )

        return self.get_state(state, params), state

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: ActionConfiguration, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict]:
        def update_done(c_step, n_steps):
            return jax.lax.cond(
                c_step >= n_steps,
                lambda _: True,  # If condition is True, set done to True
                lambda _: False, # If condition is False, keep done as False
                None
            ) 

        c_step = state.c_step + 1

        done = update_done(c_step, params.n_steps)

        env, env_params = self.envs[0]
        rng, agent_rng = jax.random.split(rng)
        agent = PPO(action.config, params.options, env, env_params)
        runner_state = agent.init(agent_rng)

        runner_state, metrics = agent.train(runner_state)

        if "trajectory" in params.options["checkpoint"]:
            (
                loss_info,
                grad_info,
                traj,
                additional_info,
            ) = metrics
        elif params.options["grad_obs"]:
            (
                loss_info,
                grad_info,
                additional_info,
            ) = metrics

        if self.algorithm == "dqn":
            self.global_step = runner_state[5]

        reward = agent.eval(runner_state, params.options["n_eval_episodes"])

        state = EnvState(
            c_step=c_step,
            episode=state.episode + 1,
            runner_state=runner_state,
            c_env_id=self.select_next_env(state.c_env_id),
            n_total_timesteps=state.n_total_timesteps
        )

        return (
            jax.lax.stop_gradient(self.get_state(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {},
        )

    def switch_algorithm(self, new_algorithm):
        self.algorithm = new_algorithm
        self.reset()

    # Useful features could be: total deltas of grad norm and grad var, instance info...
    def get_default_state(self, state, _):
        return jnp.array(
            [state.c_step, state.c_step * state.n_total_timesteps]
        )

    def get_gradient_state(self, state, params):
        if state.c_step == 0:
            grad_norm = 0
            grad_var = 0
        else:
            grad_info = state.grad_info
            if params.config.algorithm == "ppo":
                import flax

                grad_info = grad_info["params"]
                grad_info = {
                    k: v
                    for (k, v) in grad_info.items()
                    if isinstance(v, flax.core.frozen_dict.FrozenDict)
                }
            grad_info = [
                grad_info[g][k] for g in grad_info.keys() for k in grad_info[g].keys()
            ]
            grad_norm = jnp.mean(jnp.array([jnp.linalg.norm(g) for g in grad_info]))
            grad_var = jnp.mean(jnp.array([jnp.var(g) for g in grad_info]))
        return jnp.array(
            [
                state.c_step,
                state.c_step * state.instance["total_timesteps"],
                grad_norm,
                grad_var,
            ]
        )

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training import orbax_utils
from flax.core.frozen_dict import FrozenDict
import gymnax
import orbax
import chex
from typing import Tuple, Union, Any, Dict
from gymnax.environments.environment import Environment
from agents import Agent, PPO, DQN
from .utils import make_env

@struct.dataclass
class EnvState:
    rng: chex.PRNGKey
    c_step: int
    episode: int
    env: Any
    env_params: Any
    runner_state: Any
    instance: Dict

@struct.dataclass
class EnvParams:
    # TODO add env parameters
    rng: chex.PRNGKey
    config: Dict
    options: Dict
    n_steps: int
    checkpoint: Any
    checkpoint_dir: str
    instance: Dict


class AutoRLEnv(Environment):
    ALGORITHMS = {"ppo": PPO, "dqn": DQN}

    def __init__(self, config) -> None:
        super().__init__()

        self.algorithm = config["algorithm"]

        # TODO add support for different reward functions
        # if "reward_function" in config.keys():
        #     self.get_reward = config["reward_function"]
        # else:
        #     self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        elif config.grad_obs:
            self.get_state = self.get_gradient_state
        else:
            self.get_state = self.get_default_state

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        # TODO do we want to instantiate env here? 
        # Alternatively, return a dummy obs and state and only init envs in step_env
        env, env_params = make_env(params.instance)

        rng, agent_rng = jax.random.split(params.rng)
        agent = self.ALGORITHMS[params.config["algorithm"]](params.config, env, env_params)
        runner_state = agent.init(agent_rng)

        # TODO adapt old loading code
        # if "load" in params.options.keys():
        #     checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #     restored = checkpointer.restore(params.options["load"])
        #     network_params = restored["params"]
        #     if isinstance(network_params, list):
        #         network_params = network_params[0]
        #     network_params = FrozenDict(network_params)
        #     if "buffer_obs" in restored.keys():
        #         obs = restored["buffer_obs"]
        #         next_obs = restored["buffer_next_obs"]
        #         actions = restored["buffer_actions"]
        #         rewards = restored["buffer_rewards"]
        #         dones = restored["buffer_dones"]
        #         weights = restored["buffer_weights"]
        #         self.buffer_state = buffer.add_batch_fn(
        #             self.buffer_state,
        #             ((obs, next_obs, actions, rewards, dones), weights),
        #         )

        #     instance = restored["config"]
        #     if "target" in restored.keys():
        #         target_params = restored["target"][0]
        #         if isinstance(target_params, list):
        #             target_params = target_params[0]
        #     try:
        #         opt_state = restored["opt_state"]
        #     except:
        #         opt_state = None
        # else:
        #     network_params = network.init(_rng, init_x)
        #     target_params = network.init(_rng, init_x)
        #     opt_state = None

        state = EnvState(
            rng=rng,
            c_step=0,
            episode=0,
            env=env,
            env_params=env_params,
            runner_state=runner_state,
            instance=params.instance
        )

        return self.get_state(state, params), state

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        done = False
        c_step = state.c_step + 1
        if state.c_step >= params.n_steps:
            done = True

        env, env_params = make_env(params.instance)
        rng, agent_rng = jax.random.split(params.rng)
        agent = self.ALGORITHMS[params.config["algorithm"]](params.config, env, env_params)
        runner_state = agent.init(agent_rng)

        _, _rng = jax.random.split(rng)
        # TODO adapt old loading code
        # if "load" in params.options.keys():
        #     checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #     restored = checkpointer.restore(params.options["load"])
        #     network_params = restored["params"]
        #     if isinstance(network_params, list):
        #         network_params = network_params[0]
        #     network_params = FrozenDict(network_params)
        #     if "buffer_obs" in restored.keys():
        #         obs = restored["buffer_obs"]
        #         next_obs = restored["buffer_next_obs"]
        #         actions = restored["buffer_actions"]
        #         rewards = restored["buffer_rewards"]
        #         dones = restored["buffer_dones"]
        #         weights = restored["buffer_weights"]
        #         self.buffer_state = buffer.add_batch_fn(
        #             self.buffer_state,
        #             ((obs, next_obs, actions, rewards, dones), weights),
        #         )

        #     instance = restored["config"]
        #     if "target" in restored.keys():
        #         target_params = restored["target"][0]
        #         if isinstance(target_params, list):
        #             target_params = target_params[0]
        #     try:
        #         opt_state = restored["opt_state"]
        #     except:
        #         opt_state = None
        # else:
        #     network_params = network.init(_rng, init_x)
        #     target_params = network.init(_rng, init_x)
        #     opt_state = None

        instance = params.instance.copy()
        instance.update(action)
        instance["track_traj"] = "trajectory" in params.config["checkpoint"]
        instance["track_metrics"] = params.config["grad_obs"]

        runner_state, metrics = agent.train(runner_state)

        if "trajectory" in params.config["checkpoint"]:
            (
                loss_info,
                grad_info,
                traj,
                additional_info,
            ) = metrics
        elif params.config["grad_obs"]:
            (
                loss_info,
                grad_info,
                additional_info,
            ) = metrics

        if self.algorithm == "dqn":
            self.global_step = runner_state[5]

        reward = agent.eval(runner_state, params.config["num_eval_episodes"])

        # TODO adapt this
        # reward = self.get_reward(self)

        # TODO adapt checkpointing etc
        # if params.config["checkpoint"]:
        #     # Checkpoint setup
        #     checkpoint_name = params.config["checkpoint_dir"] + "/"
        #     if "checkpoint_name" in params.config.keys():
        #         checkpoint_name += params.config["checkpoint_name"]
        #     else:
        #         if not done:
        #             checkpoint_name += f"_episode_{state.episode}_step_{state.c_step}"
        #         else:
        #             checkpoint_name += "_final"

        #     ckpt = {
        #         "config": params.instance,
        #     }

        #     if "opt_state" in params.config["checkpoint"]:
        #         ckpt["optimizer_state"] = opt_info

        #     if "policy" in params.config["checkpoint"]:
        #         ckpt["params"] = state.network_params
        #         if "target" in params.instance.keys():
        #             ckpt["target"] = state.target_params

        #     if "buffer" in params.config["checkpoint"]:
        #         ckpt["buffer_obs"] = state.buffer_state.storage.data[0]
        #         ckpt["buffer_next_obs"] = state.buffer_state.storage.data[1]
        #         ckpt["buffer_actions"] = state.buffer_state.storage.data[2]
        #         ckpt["buffer_rewards"] = state.buffer_state.storage.data[3]
        #         ckpt["buffer_dones"] = state.buffer_state.storage.data[4]
        #         ckpt["buffer_weights"] = state.buffer_state.storage.weights

        #     save_args = orbax_utils.save_args_from_target(ckpt)
        #     checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #     checkpointer.save(checkpoint_name, ckpt, save_args=save_args)

        #     if "loss" in params.config["checkpoint"]:
        #         ckpt = {}
        #         if params.config["algorithm"] == "ppo":
        #             ckpt["value_loss"] = jnp.concatenate(state.loss_info[0], axis=0)
        #             ckpt["actor_loss"] = jnp.concatenate(state.loss_info[1], axis=0)
        #         elif params.config["algorithm"] == "dqn":
        #             ckpt["loss"] = state.loss_info

        #         save_args = orbax_utils.save_args_from_target(ckpt)
        #         checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #         loss_checkpoint = checkpoint_name + "_loss"
        #         checkpointer.save(
        #             loss_checkpoint,
        #             ckpt,
        #             save_args=save_args,
        #         )

        #     additional_info = state.additional_info

        #     if (
        #         "minibatches" in params.config["checkpoint"]
        #         and "trajectory" in params.config["checkpoint"]
        #     ):
        #         ckpt = {}
        #         ckpt["minibatches"] = {}
        #         ckpt["minibatches"]["states"] = jnp.concatenate(
        #             additional_info["minibatches"][0].obs, axis=0
        #         )
        #         ckpt["minibatches"]["value"] = jnp.concatenate(
        #             additional_info["minibatches"][0].value, axis=0
        #         )
        #         ckpt["minibatches"]["action"] = jnp.concatenate(
        #             additional_info["minibatches"][0].action, axis=0
        #         )
        #         ckpt["minibatches"]["reward"] = jnp.concatenate(
        #             additional_info["minibatches"][0].reward, axis=0
        #         )
        #         ckpt["minibatches"]["log_prob"] = jnp.concatenate(
        #             additional_info["minibatches"][0].log_prob, axis=0
        #         )
        #         ckpt["minibatches"]["dones"] = jnp.concatenate(
        #             additional_info["minibatches"][0].done, axis=0
        #         )
        #         ckpt["minibatches"]["advantages"] = jnp.concatenate(
        #             additional_info["minibatches"][1], axis=0
        #         )
        #         ckpt["minibatches"]["targets"] = jnp.concatenate(
        #             additional_info["minibatches"][2], axis=0
        #         )
        #         save_args = orbax_utils.save_args_from_target(ckpt)
        #         checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #         minibatch_checkpoint = checkpoint_name + "_minibatches"
        #         checkpointer.save(
        #             minibatch_checkpoint,
        #             ckpt,
        #             save_args=save_args,
        #         )

        #     if "extras" in params.config["checkpoint"]:
        #         ckpt = {}
        #         for k in additional_info:
        #             if k == "param_history":
        #                 ckpt[k] = additional_info[k]
        #             elif k != "minibatches":
        #                 ckpt[k] = jnp.concatenate(additional_info[k], axis=0)
        #             elif "gradient_history" in params.config["checkpoint"]:
        #                 ckpt["gradient_history"] = state.grad_info["params"]

        #         save_args = orbax_utils.save_args_from_target(ckpt)
        #         checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #         extras_checkpoint = checkpoint_name + "_extras"
        #         checkpointer.save(
        #             extras_checkpoint,
        #             ckpt,
        #             save_args=save_args,
        #         )

        #     if "trajectory" in params.config["checkpoint"]:
        #         ckpt = {}
        #         ckpt["trajectory"] = {}
        #         ckpt["trajectory"]["states"] = jnp.concatenate(state.traj.obs, axis=0)
        #         ckpt["trajectory"]["action"] = jnp.concatenate(
        #             state.traj.action, axis=0
        #         )
        #         ckpt["trajectory"]["reward"] = jnp.concatenate(
        #             state.traj.reward, axis=0
        #         )
        #         ckpt["trajectory"]["dones"] = jnp.concatenate(state.traj.done, axis=0)
        #         if params.config["algorithm"] == "ppo":
        #             ckpt["trajectory"]["value"] = jnp.concatenate(
        #                 state.traj.value, axis=0
        #             )
        #             ckpt["trajectory"]["log_prob"] = jnp.concatenate(
        #                 state.traj.log_prob, axis=0
        #             )
        #         elif params.config["algorithm"] == "dqn":
        #             ckpt["trajectory"]["q_pred"] = jnp.concatenate(
        #                 state.traj.q_pred, axis=0
        #             )

        #         save_args = orbax_utils.save_args_from_target(ckpt)
        #         checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #         traj_checkpoint = checkpoint_name + "_trajectory"
        #         checkpointer.save(
        #             traj_checkpoint,
        #             ckpt,
        #             save_args=save_args,
        #         )

        state = EnvState(
            rng=rng,
            c_step=c_step,
            episode=state.episode + 1,
            env=env,
            env_params=env_params,
            runner_state=runner_state,
            instance=params.instance
        )

        return (
            jax.lax.stop_gradient(self.get_state(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {},
        )

    # def get_default_reward(self, state) -> float:
    #     return self.eval_func(state.rng, state.network_params)

    def switch_algorithm(self, new_algorithm):
        self.algorithm = new_algorithm
        self.reset()

    # Useful features could be: total deltas of grad norm and grad var, instance info...
    def get_default_state(self, state, _):
        return jnp.array(
            [state.c_step, state.c_step * state.instance["total_timesteps"]]
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

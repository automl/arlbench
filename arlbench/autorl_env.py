import jax
import jax.numpy as jnp
import numpy as np
import random
import flax
from flax import struct
from flax.training import orbax_utils
from flax.core.frozen_dict import FrozenDict
import gymnax
import orbax
import chex
from typing import Tuple, Union, Any, Dict
from utils import (
    make_train_ppo,
    make_train_dqn,
    make_eval,
    ActorCritic,
    Q,
    make_env,
    uniform_replay,
    UniformReplayBufferState,
)
from gymnax.environments.environment import Environment


@struct.dataclass
class EnvState:
    c_step: int
    global_step: int
    episode: int
    env: Any
    env_params: Any
    total_updates: int
    update_interval: int
    network: Any
    network_params: Any
    target_params: Any
    opt_state: Any
    opt_info: Any
    buffer_state: Any
    last_obsv: Any
    last_env_state: Any
    loss_info: Any
    grad_info: Any
    traj: Any
    additional_info: Any
    rng: Any


@struct.dataclass
class EnvParams:
    # TODO add env parameters
    instance: Any
    config: Dict
    options: Dict
    n_steps: int
    checkpoint: Any
    checkpoint_dir: str


class AutoRLEnv(Environment):
    ALGORITHMS = {"ppo": (make_train_ppo, ActorCritic), "dqn": (make_train_dqn, Q)}

    def __init__(self, config) -> None:
        super().__init__()

        # TODO access later on through EnvParams and EnvState
        # self.config = config
        # self.checkpoint = self.config["checkpoint"]
        # self.checkpoint_dir = self.config["checkpoint_dir"]
        # self.rng = jax.random.PRNGKey(self.config.seed)

        #  self.episode = 0

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        elif config.grad_obs:
            self.get_state = self.get_gradient_state
        else:
            self.get_state = self.get_default_state

        if "algorithm" in config.keys():
            self.make_train = self.ALGORITHMS[config.algorithm][0]
            self.network_cls = self.ALGORITHMS[config.algorithm][1]
        else:
            self.make_train = make_train_ppo
            self.network_cls = ActorCritic

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        # TODO move to new function that is shared among reset and step
        env, env_params = make_env(params.instance)

        # TODO how to access state.rng here?
        # for now, use initial rng based on config seed
        rng = jax.random.PRNGKey(params.config["seed"])
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, params.instance["num_envs"])

        last_obsv, last_env_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_rng, env_params
        )
        global_step = 0

        if isinstance(
            env.action_space(env_params), gymnax.environments.spaces.Discrete
        ):
            action_size = env.action_space(env_params).n
            action_buffer_size = 1
            discrete = True
        elif isinstance(env.action_space(env_params), gymnax.environments.spaces.Box):
            action_size = env.action_space(env_params).shape[0]
            if len(env.action_space(env_params).shape) > 1:
                action_buffer_size = [
                    env.action_space(env_params).shape[0],
                    env.action_space(env_params).shape[1],
                ]
            elif env.name == "BraxToGymnaxWrapper":
                action_buffer_size = [action_size, 1]
            else:
                action_buffer_size = action_size

            discrete = False
        else:
            raise NotImplementedError(
                f"Only Discrete and Box action spaces are supported, got {env.action_space(env_params)}."
            )

        network = self.network_cls(
            action_size,
            activation=params.instance["activation"],
            hidden_size=params.instance["hidden_size"],
            discrete=discrete,
        )
        # TODO use flashbax
        buffer = uniform_replay(
            max_size=int(params.instance["buffer_size"]), beta=params.instance["beta"]
        )
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        buffer_state = buffer.init_fn(
            (
                jnp.zeros(init_x.shape),
                jnp.zeros(init_x.shape),
                jnp.zeros(action_buffer_size),
                jnp.zeros(1),
                jnp.zeros(1),
            ),
            jnp.zeros(1),
        )

        _, _rng = jax.random.split(rng)
        if "load" in params.options.keys():
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = checkpointer.restore(params.options["load"])
            network_params = restored["params"]
            if isinstance(network_params, list):
                network_params = network_params[0]
            network_params = FrozenDict(network_params)
            if "buffer_obs" in restored.keys():
                obs = restored["buffer_obs"]
                next_obs = restored["buffer_next_obs"]
                actions = restored["buffer_actions"]
                rewards = restored["buffer_rewards"]
                dones = restored["buffer_dones"]
                weights = restored["buffer_weights"]
                self.buffer_state = buffer.add_batch_fn(
                    self.buffer_state,
                    ((obs, next_obs, actions, rewards, dones), weights),
                )

            instance = restored["config"]
            if "target" in restored.keys():
                target_params = restored["target"][0]
                if isinstance(target_params, list):
                    target_params = target_params[0]
            try:
                opt_state = restored["opt_state"]
            except:
                opt_state = None
        else:
            network_params = network.init(_rng, init_x)
            target_params = network.init(_rng, init_x)
            opt_state = None
        self.eval_func = make_eval(instance, network)
        if params.config["algorithm"] == "ppo":
            total_updates = (
                instance["total_timesteps"]
                // instance["num_steps"]
                // instance["num_envs"]
            )
            update_interval = np.ceil(total_updates / params.n_steps)
            if update_interval < 1:
                update_interval = 1
                print(
                    "WARNING: The number of iterations selected in combination with your timestep, num_env and num_step settings results in 0 steps per iteration. Rounded up to 1, this means more total steps will be executed."
                )
        else:
            self.update_interval = None

        state = EnvState(
            c_step=0,
            global_step=0,
            episode=0,  # TODO validate
            env=env,
            env_params=env_params,
            total_updates=total_updates,
            update_interval=update_interval,
            network=network,
            network_params=network_params,
            target_params=target_params,
            opt_state=opt_state,
            opt_info=None,
            buffer_state=buffer_state,
            last_obsv=last_obsv,
            last_env_state=last_env_state,
            loss_info=[],
            grad_info=[],
            traj=[],
            additional_info={},
            rng=rng,
        )

        return self.get_state(state, params), state

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        done = False
        state.c_step += 1
        if state.c_step >= params.n_steps:
            done = True

        # ------------------------------------------------------------------------
        # 1) Environment initialization
        # ------------------------------------------------------------------------
        env, env_params = make_env(params.instance)

        # TODO how to access state.rng here?
        # for now, use initial rng based on config seed
        rng = jax.random.PRNGKey(params.config["seed"])
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, params.instance["num_envs"])

        last_obsv, last_env_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_rng, env_params
        )
        global_step = 0

        if isinstance(
            env.action_space(env_params), gymnax.environments.spaces.Discrete
        ):
            action_size = env.action_space(env_params).n
            action_buffer_size = 1
            discrete = True
        elif isinstance(env.action_space(env_params), gymnax.environments.spaces.Box):
            action_size = env.action_space(env_params).shape[0]
            if len(env.action_space(env_params).shape) > 1:
                action_buffer_size = [
                    env.action_space(env_params).shape[0],
                    env.action_space(env_params).shape[1],
                ]
            elif env.name == "BraxToGymnaxWrapper":
                action_buffer_size = [action_size, 1]
            else:
                action_buffer_size = action_size

            discrete = False
        else:
            raise NotImplementedError(
                f"Only Discrete and Box action spaces are supported, got {env.action_space(env_params)}."
            )

        # ------------------------------------------------------------------------
        # 2) Network and buffer initialization
        # ------------------------------------------------------------------------
        network = self.network_cls(
            action_size,
            activation=params.instance["activation"],
            hidden_size=params.instance["hidden_size"],
            discrete=discrete,
        )
        # TODO use flashbax buffer
        buffer = uniform_replay(
            max_size=int(params.instance["buffer_size"]), beta=params.instance["beta"]
        )
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        buffer_state = buffer.init_fn(
            (
                jnp.zeros(init_x.shape),
                jnp.zeros(init_x.shape),
                jnp.zeros(action_buffer_size),
                jnp.zeros(1),
                jnp.zeros(1),
            ),
            jnp.zeros(1),
        )

        _, _rng = jax.random.split(rng)
        if "load" in params.options.keys():
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = checkpointer.restore(params.options["load"])
            network_params = restored["params"]
            if isinstance(network_params, list):
                network_params = network_params[0]
            network_params = FrozenDict(network_params)
            if "buffer_obs" in restored.keys():
                obs = restored["buffer_obs"]
                next_obs = restored["buffer_next_obs"]
                actions = restored["buffer_actions"]
                rewards = restored["buffer_rewards"]
                dones = restored["buffer_dones"]
                weights = restored["buffer_weights"]
                self.buffer_state = buffer.add_batch_fn(
                    self.buffer_state,
                    ((obs, next_obs, actions, rewards, dones), weights),
                )

            instance = restored["config"]
            if "target" in restored.keys():
                target_params = restored["target"][0]
                if isinstance(target_params, list):
                    target_params = target_params[0]
            try:
                opt_state = restored["opt_state"]
            except:
                opt_state = None
        else:
            network_params = network.init(_rng, init_x)
            target_params = network.init(_rng, init_x)
            opt_state = None
        self.eval_func = make_eval(instance, network)
        if params.config["algorithm"] == "ppo":
            total_updates = (
                instance["total_timesteps"]
                // instance["num_steps"]
                // instance["num_envs"]
            )
            update_interval = np.ceil(total_updates / params.n_steps)
            if update_interval < 1:
                update_interval = 1
                print(
                    "WARNING: The number of iterations selected in combination with your timestep, num_env and num_step settings results in 0 steps per iteration. Rounded up to 1, this means more total steps will be executed."
                )
        else:
            self.update_interval = None

        # ------------------------------------------------------------------------
        # 3) Training
        # ------------------------------------------------------------------------
        if "algorithm" in action.keys():
            print(
                f"Changing algorithm to {action['algorithm']} - attention, this will reinstantiate the network!"
            )
            self.switch_algorithm(action["algorithm"])

        params.instance.update(action)
        params.instance["track_traj"] = "trajectory" in params.config["checkpoint"]
        params.instance["track_metrics"] = params.config["grad_obs"]

        self.train_func = jax.jit(
            self.make_train(
                params.instance, state.env, state.network, state.update_interval
            )
        )

        train_args = (
            state.rng,
            state.env_params,
            state.network_params,
            state.opt_state,
            state.last_obsv,
            state.last_env_state,
            state.buffer_state,
        )
        if params.config["algorithm"] == "dqn":
            train_args = (
                state.rng,
                state.env_params,
                state.network_params,
                state.target_params,
                state.opt_state,
                state.last_obsv,
                state.last_env_state,
                state.buffer_state,
                state.global_step,
            )

        runner_state, metrics = self.train_func(*train_args)
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
        network_params = runner_state[0].params
        last_obsv = runner_state[2]
        last_env_state = runner_state[1]
        buffer_state = runner_state[4]
        opt_info = runner_state[0].opt_state
        if params.config["algorithm"] == "dqn":
            self.global_step = runner_state[5]
        reward = self.get_reward(self)
        if params.config["checkpoint"]:
            # Checkpoint setup
            checkpoint_name = params.config["checkpoint_dir"] + "/"
            if "checkpoint_name" in params.config.keys():
                checkpoint_name += params.config["checkpoint_name"]
            else:
                if not done:
                    checkpoint_name += f"_episode_{state.episode}_step_{state.c_step}"
                else:
                    checkpoint_name += "_final"

            ckpt = {
                "config": params.instance,
            }

            if "opt_state" in params.config["checkpoint"]:
                ckpt["optimizer_state"] = opt_info

            if "policy" in params.config["checkpoint"]:
                ckpt["params"] = state.network_params
                if "target" in params.instance.keys():
                    ckpt["target"] = state.target_params

            if "buffer" in params.config["checkpoint"]:
                ckpt["buffer_obs"] = state.buffer_state.storage.data[0]
                ckpt["buffer_next_obs"] = state.buffer_state.storage.data[1]
                ckpt["buffer_actions"] = state.buffer_state.storage.data[2]
                ckpt["buffer_rewards"] = state.buffer_state.storage.data[3]
                ckpt["buffer_dones"] = state.buffer_state.storage.data[4]
                ckpt["buffer_weights"] = state.buffer_state.storage.weights

            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            checkpointer.save(checkpoint_name, ckpt, save_args=save_args)

            if "loss" in params.config["checkpoint"]:
                ckpt = {}
                if params.config["algorithm"] == "ppo":
                    ckpt["value_loss"] = jnp.concatenate(state.loss_info[0], axis=0)
                    ckpt["actor_loss"] = jnp.concatenate(state.loss_info[1], axis=0)
                elif params.config["algorithm"] == "dqn":
                    ckpt["loss"] = state.loss_info

                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                loss_checkpoint = checkpoint_name + "_loss"
                checkpointer.save(
                    loss_checkpoint,
                    ckpt,
                    save_args=save_args,
                )

            additional_info = state.additional_info

            if (
                "minibatches" in params.config["checkpoint"]
                and "trajectory" in params.config["checkpoint"]
            ):
                ckpt = {}
                ckpt["minibatches"] = {}
                ckpt["minibatches"]["states"] = jnp.concatenate(
                    additional_info["minibatches"][0].obs, axis=0
                )
                ckpt["minibatches"]["value"] = jnp.concatenate(
                    additional_info["minibatches"][0].value, axis=0
                )
                ckpt["minibatches"]["action"] = jnp.concatenate(
                    additional_info["minibatches"][0].action, axis=0
                )
                ckpt["minibatches"]["reward"] = jnp.concatenate(
                    additional_info["minibatches"][0].reward, axis=0
                )
                ckpt["minibatches"]["log_prob"] = jnp.concatenate(
                    additional_info["minibatches"][0].log_prob, axis=0
                )
                ckpt["minibatches"]["dones"] = jnp.concatenate(
                    additional_info["minibatches"][0].done, axis=0
                )
                ckpt["minibatches"]["advantages"] = jnp.concatenate(
                    additional_info["minibatches"][1], axis=0
                )
                ckpt["minibatches"]["targets"] = jnp.concatenate(
                    additional_info["minibatches"][2], axis=0
                )
                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                minibatch_checkpoint = checkpoint_name + "_minibatches"
                checkpointer.save(
                    minibatch_checkpoint,
                    ckpt,
                    save_args=save_args,
                )

            if "extras" in params.config["checkpoint"]:
                ckpt = {}
                for k in additional_info:
                    if k == "param_history":
                        ckpt[k] = additional_info[k]
                    elif k != "minibatches":
                        ckpt[k] = jnp.concatenate(additional_info[k], axis=0)
                    elif "gradient_history" in params.config["checkpoint"]:
                        ckpt["gradient_history"] = state.grad_info["params"]

                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                extras_checkpoint = checkpoint_name + "_extras"
                checkpointer.save(
                    extras_checkpoint,
                    ckpt,
                    save_args=save_args,
                )

            if "trajectory" in params.config["checkpoint"]:
                ckpt = {}
                ckpt["trajectory"] = {}
                ckpt["trajectory"]["states"] = jnp.concatenate(state.traj.obs, axis=0)
                ckpt["trajectory"]["action"] = jnp.concatenate(
                    state.traj.action, axis=0
                )
                ckpt["trajectory"]["reward"] = jnp.concatenate(
                    state.traj.reward, axis=0
                )
                ckpt["trajectory"]["dones"] = jnp.concatenate(state.traj.done, axis=0)
                if params.config["algorithm"] == "ppo":
                    ckpt["trajectory"]["value"] = jnp.concatenate(
                        state.traj.value, axis=0
                    )
                    ckpt["trajectory"]["log_prob"] = jnp.concatenate(
                        state.traj.log_prob, axis=0
                    )
                elif params.config["algorithm"] == "dqn":
                    ckpt["trajectory"]["q_pred"] = jnp.concatenate(
                        state.traj.q_pred, axis=0
                    )

                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                traj_checkpoint = checkpoint_name + "_trajectory"
                checkpointer.save(
                    traj_checkpoint,
                    ckpt,
                    save_args=save_args,
                )

        state = EnvState(
            c_step=0,
            global_step=0,
            episode=0,  # TODO validate
            env=env,
            env_params=env_params,
            total_updates=total_updates,
            update_interval=update_interval,
            network=network,
            network_params=network_params,
            target_params=target_params,
            opt_state=opt_state,
            opt_info=opt_info,
            buffer_state=buffer_state,
            last_obsv=last_obsv,
            last_env_state=last_env_state,
            loss_info=loss_info,
            grad_info=grad_info,
            traj=traj,
            additional_info=additional_info,
            rng=state.rng,
        )

        return (
            jax.lax.stop_gradient(self.get_state(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {},
        )

    def get_default_reward(self, state) -> float:
        return self.eval_func(state.rng, state.network_params)

    def switch_algorithm(self, new_algorithm):
        self.make_train = self.ALGORITHMS[new_algorithm][0]
        self.network_cls = self.ALGORITHMS[new_algorithm][1]
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
            grad_norm = np.mean([jnp.linalg.norm(g) for g in grad_info])
            grad_var = np.mean([jnp.var(g) for g in grad_info])
        return jnp.array(
            [
                state.c_step,
                state.c_step * state.instance["total_timesteps"],
                grad_norm,
                grad_var,
            ]
        )

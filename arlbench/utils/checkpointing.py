from typing import Union, Optional, Any
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState
from flashbax.buffers.sum_tree import SumTreeState
from arlbench.agents import PPORunnerState, PPOTrainState, DQNRunnerState, DQNTrainState
from flashbax.vault import Vault
import numpy as np
import jax.numpy as jnp
import json
import os
import orbax
from flax.training import orbax_utils
import orbax.checkpoint as ocp
from flax.core.frozen_dict import FrozenDict


class Checkpointer:
    NODES_FILE = "nodes.npy"
    SCALARS_FILE = "scalars.json"
    MRP_FILE = "max_recorded_priority.npy"

    @staticmethod
    def _save_orbax_checkpoint(checkpoint: dict[str, Any], checkpoint_dir: str, checkpoint_name: str) -> None:
        save_args = orbax_utils.save_args_from_target(checkpoint)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            os.path.join(checkpoint_dir, checkpoint_name),
            checkpoint,
            save_args=save_args,
            force=True  # TODO debug, remove later on
        )

    @staticmethod
    def _load_orbax_checkpoint(checkpoint_dir: str, checkpoint_name: str) -> dict[str, Any]:
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(os.path.join(checkpoint_dir, checkpoint_name))

    @staticmethod
    def save(
        runner_state: Union[PPORunnerState, DQNRunnerState],
        buffer_state: PrioritisedTrajectoryBufferState,
        config: dict,
        hp_config: dict,
        done: bool,
        c_episode: int,
        c_step: int,
        loss_info: Any,
        metrics: tuple
    ) -> None:
        # Checkpoint setup
        checkpoint = config["checkpoint"]   # list of strings
        checkpoint_name = config["checkpoint_name"]
        checkpoint_dir = os.path.join(config["checkpoint_dir"], checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Structure: 
        # /.../checkpoint_name/checkpoint_name_c_episode_0_step_0
        # /.../checkpoint_name/checkpoint_name_c_episode_0_step_1
        # ...
        # /.../checkpoint_name/checkpoint_name_final
        if not done:
            checkpoint_name += f"_c_episode_{c_episode}_step_{c_step}"
        else:
            checkpoint_name += "_final"

        train_state = runner_state.train_state

        if "trajectory" in checkpoint:
            (
                loss_info,
                grad_info,
                traj,
                additional_info,
            ) = metrics

        
        network_params = train_state.params
        last_obsv = runner_state.obs
        last_env_state = runner_state.env_state
        opt_info = train_state.opt_state

        ckpt: dict[str, Any] = {
            "config": hp_config,
            "c_step": c_step,
            "c_episode": c_episode
        }

        if "opt_state" in checkpoint:
            ckpt["optimizer_state"] = opt_info

        if "policy" in checkpoint:
            ckpt["params"] = network_params
            if config["algorithm"] == "dqn":
                ckpt["target"] = train_state.target_params
            # TODO add SAC

        if "buffer" in checkpoint:
            ckpt["buffer"] = {}
            ckpt["buffer"]["experience"] = buffer_state.experience
            ckpt["buffer"]["current_index"] = buffer_state.current_index
            ckpt["buffer"]["is_full"] = buffer_state.is_full
            ckpt["buffer"]["priority_state"] = buffer_state.priority_state

            # Checkpointer.save_buffer(buffer_state, checkpoint_dir, checkpoint_name)

        Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name)

        if "loss" in checkpoint:
            ckpt = {}
            if config["algorithm"] == "ppo":
                ckpt["value_loss"] = jnp.concatenate(loss_info[0], axis=0)
                ckpt["actor_loss"] = jnp.concatenate(loss_info[1], axis=0)
            elif config["algorithm"]== "dqn":
                ckpt["loss"] = loss_info
            Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name + "_loss")

        if "minibatches" in checkpoint and "trajectory" in checkpoint:
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
            Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name + "_minibatches")

        if "extras" in checkpoint:
            ckpt = {}
            for k in additional_info:
                if k == "param_history":
                    ckpt[k] = additional_info[k]
                elif k != "minibatches":
                    ckpt[k] = jnp.concatenate(additional_info[k], axis=0)
                elif "gradient_history" in checkpoint:
                    ckpt["gradient_history"] = grad_info["params"]
            Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name + "_extras")

        if "trajectory" in checkpoint:
            ckpt = {}
            ckpt["trajectory"] = {}
            ckpt["trajectory"]["states"] = jnp.concatenate(traj.obs, axis=0)
            ckpt["trajectory"]["action"] = jnp.concatenate(traj.action, axis=0)
            ckpt["trajectory"]["reward"] = jnp.concatenate(traj.reward, axis=0)
            ckpt["trajectory"]["dones"] = jnp.concatenate(traj.done, axis=0)
            if config["algorithm"] == "ppo":
                ckpt["trajectory"]["value"] = jnp.concatenate(
                    traj.value, axis=0
                )
                ckpt["trajectory"]["log_prob"] = jnp.concatenate(
                    traj.log_prob, axis=0
                )
            elif config["algorithm"] == "dqn":
                ckpt["trajectory"]["q_pred"] = jnp.concatenate(
                    traj.q_pred, axis=0
                )
            # TODO add SAC

            Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name + "_trajectory")

    @staticmethod
    def load(options: dict[str, Any]) -> tuple[tuple, tuple]:
        if "load" in options.keys():
            checkpointer = ocp.PyTreeCheckpointer()
            restored = checkpointer.restore(options["load"])
            c_step = restored["c_step"]
            c_episode = restored["c_episode"]
            config = restored["config"]

            network_params = restored["params"]
            if isinstance(network_params, list):
                network_params = network_params[0]
            network_params = FrozenDict(network_params)

            if "target" in restored.keys():
                target_params = restored["target"][0]
                if isinstance(target_params, list):
                    target_params = target_params[0]
                target_params = FrozenDict(target_params)

            if "opt_state" in restored.keys():
                opt_state = restored["opt_state"]
            else:
                opt_state = None
        
        if config["algorithm"] == "ppo":
            return (c_step, c_episode), (config, network_params, opt_state)
        elif config["algorithm"] == "dqn":
            return (c_step, c_episode), (config, network_params, target_params, opt_state)
        else:
            raise ValueError(f"Invalid algorithm in checkpoint: {config['algorithm']}")
            
    @staticmethod
    def save_buffer(buffer_state: PrioritisedTrajectoryBufferState, checkpoint_dir: str, checkpoint_name: str) -> None:
        buffer_dir = os.path.join(checkpoint_dir, checkpoint_name + "_buffer_state")
        os.makedirs(buffer_dir, exist_ok=True)

        # write buffer
        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=buffer_state.experience,
            rel_dir=buffer_dir
        )
        v.write(buffer_state)

        Checkpointer._save_sum_tree_state(buffer_state.priority_state, os.path.join(buffer_dir, "buffer_priority_state"))

    @staticmethod
    def load_buffer(dummy_buffer_state: PrioritisedTrajectoryBufferState, checkpoint_dir: str, checkpoint_name: str) -> PrioritisedTrajectoryBufferState:
        buffer_dir = os.path.join(checkpoint_dir, checkpoint_name + "_buffer_state")

        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=dummy_buffer_state.experience,
            rel_dir=buffer_dir
        )
        buffer_state = v.read()
        print(buffer_state.experience.obs.shape)

        priority_state = Checkpointer._load_sum_tree_state(os.path.join(buffer_dir, "buffer_priority_state"))

        return PrioritisedTrajectoryBufferState(
            experience=buffer_state.experience,
            current_index=buffer_state.current_index,
            is_full=buffer_state.is_full,
            priority_state=priority_state
        )
    
    @staticmethod
    def _save_sum_tree_state(sum_tree_state: SumTreeState, directory: str) -> None:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Serialize and save the JAX arrays as .npy files
        np.save(os.path.join(directory, Checkpointer.NODES_FILE), np.array(sum_tree_state.nodes))
        np.save(os.path.join(directory, Checkpointer.MRP_FILE), np.array(sum_tree_state.max_recorded_priority))
        
        # Serialize scalar values to JSON
        scalar_values = {
            "tree_depth": sum_tree_state.tree_depth,
            "capacity": sum_tree_state.capacity,
        }
        with open(os.path.join(directory, Checkpointer.SCALARS_FILE), "w") as json_file:
            json.dump(scalar_values, json_file)

    @staticmethod
    def _load_sum_tree_state(directory: str) -> SumTreeState:
        # Load the JAX arrays from .npy files
        nodes = jnp.array(np.load(os.path.join(directory, Checkpointer.NODES_FILE)))
        max_recorded_priority = jnp.array(np.load(os.path.join(directory, Checkpointer.MRP_FILE)))
        
        # Load scalar values from JSON
        with open(os.path.join(directory, Checkpointer.SCALARS_FILE), "r") as json_file:
            scalar_values = json.load(json_file)
        
        # Reconstruct the SumTreeState object
        return SumTreeState(
            nodes=nodes,
            max_recorded_priority=max_recorded_priority,
            tree_depth=scalar_values["tree_depth"],
            capacity=scalar_values["capacity"],
        )
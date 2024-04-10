from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flashbax.buffers.prioritised_trajectory_buffer import \
    PrioritisedTrajectoryBufferState
from flashbax.buffers.sum_tree import SumTreeState
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.vault import Vault
from flax.core.frozen_dict import FrozenDict

if TYPE_CHECKING:
    from ConfigSpace import Configuration

    from arlbench.core.algorithms import RunnerState


class Checkpointer:
    NODES_FILE = "nodes.npy"
    SCALARS_FILE = "scalars.json"
    MRP_FILE = "max_recorded_priority.npy"

    @staticmethod
    def _save_orbax_checkpoint(checkpoint: dict[str, Any], checkpoint_dir: str, checkpoint_name: str) -> None:
        checkpointer = ocp.StandardCheckpointer()
        # save_args = orbax_utils.save_args_from_target(checkpoint)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            os.path.join(checkpoint_dir, checkpoint_name),
            checkpoint,
            #save_args=save_args,
            force=True  # TODO debug, remove later on
        )

    @staticmethod
    def _load_orbax_checkpoint(checkpoint_dir: str, checkpoint_name: str) -> dict[str, Any]:
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(os.path.join(checkpoint_dir, checkpoint_name))

    @staticmethod
    def save(
        runner_state: RunnerState,
        buffer_state: PrioritisedTrajectoryBufferState,
        options: dict,
        hp_config: Configuration,
        done: bool,
        c_episode: int,
        c_step: int,
        metrics: tuple | None,
        tag: str | None = None
    ) -> str:
        # Checkpoint setup
        checkpoint = options["checkpoint"]   # list of strings
        checkpoint_name = options["checkpoint_name"]
        checkpoint_dir = os.path.join(options["checkpoint_dir"], checkpoint_name)
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

        # append tag for error/sigterm
        if tag is not None:
            checkpoint_name += f"_{tag}"


        # TODO change this for SAC
        train_state = runner_state.train_state

        if "minibatches" in checkpoint or "trajectories" in checkpoint:
            if metrics and len(metrics) == 4:
                (
                    loss_info,
                    grad_info,
                    traj,
                    additional_info,
                ) = metrics
            else:
                raise ValueError("'trajectories' in checkpoint but 'metrics' does not match.")
        elif "loss" in checkpoint or "extras" in checkpoint:
            if metrics and len(metrics) == 3:
                (
                    loss_info,
                    grad_info,
                    additional_info,
                ) = metrics
            else:
                raise ValueError("'loss' or 'extras' in checkpoint but 'metrics' does not match.")

        network_params = train_state.params
        opt_info = train_state.opt_state

        ckpt: dict[str, Any] = {
            "config": dict(hp_config),
            "options": options,
            "c_step": c_step,
            "c_episode": c_episode
        }

        if "opt_state" in checkpoint or checkpoint == "all":
            ckpt["optimizer_state"] = opt_info

        if "policy" in checkpoint:
            ckpt["params"] = network_params
            if options["algorithm"] == "dqn":
                ckpt["target"] = train_state.target_params
            # TODO add SAC

        if "buffer" in checkpoint:
            ckpt["buffer"] = Checkpointer.save_buffer(buffer_state, checkpoint_dir, checkpoint_name)

        Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name)

        if "loss" in checkpoint:
            ckpt = {}
            if options["algorithm"] == "ppo":
                ckpt["value_loss"] = jnp.concatenate(loss_info[0], axis=0)
                ckpt["actor_loss"] = jnp.concatenate(loss_info[1], axis=0)
            elif options["algorithm"]== "dqn":
                ckpt["loss"] = loss_info
            # TODO add SAC
            Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name + "_loss")

        if "minibatches" in checkpoint and "trajectory" in checkpoint and options["algorithm"] == "ppo":
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

        if "trajectories" in checkpoint and options["algorithm"] == "dqn":
            ckpt = {}
            ckpt["trajectories"] = {}
            ckpt["trajectories"]["states"] = jnp.concatenate(traj.obs, axis=0)
            ckpt["trajectories"]["action"] = jnp.concatenate(traj.action, axis=0)
            ckpt["trajectories"]["reward"] = jnp.concatenate(traj.reward, axis=0)
            ckpt["trajectories"]["dones"] = jnp.concatenate(traj.done, axis=0)
            if options["algorithm"] == "ppo":
                ckpt["trajectories"]["value"] = jnp.concatenate(
                    traj.value, axis=0
                )
                ckpt["trajectories"]["log_prob"] = jnp.concatenate(
                    traj.log_prob, axis=0
                )
            elif options["algorithm"] == "dqn":
                ckpt["trajectories"]["q_pred"] = jnp.concatenate(
                    traj.q_pred, axis=0
                )
            # TODO add SAC

            Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name + "_trajectory")

        return os.path.join(checkpoint_dir, checkpoint_name)

    @staticmethod
    def load(
        checkpoint_path: str,
        algorithm: str,
        dummy_buffer_state: PrioritisedTrajectoryBufferState
    ) -> tuple[tuple[dict[str, Any], int, int], dict]:
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(checkpoint_path)
        c_step = restored["c_step"]
        c_episode = restored["c_episode"]
        config = restored["config"]

        if "buffer" in restored:
            buffer_state = Checkpointer.load_buffer(
                dummy_buffer_state,
                restored["buffer"]["priority_state_path"],
                restored["buffer"]["buffer_dir"],
                restored["buffer"]["vault_uuid"]
            )

        network_params = restored["params"]
        if isinstance(network_params, list):
            network_params = network_params[0]
        network_params = FrozenDict(network_params)

        if "target" in restored:
            target_params = restored["target"]
            if isinstance(target_params, list):
                target_params = target_params[0]
            target_params = FrozenDict(target_params)

        opt_state = restored.get("opt_state", None)

        common = (config, c_step, c_episode)
        if algorithm == "ppo":
            algorithm_kw_args = {
                "buffer_state": buffer_state,
                "network_params": network_params,
                "opt_state": opt_state
            }
        elif algorithm == "dqn":
            algorithm_kw_args = {
                "buffer_state": buffer_state,
                "network_params": network_params,
                "target_params": target_params,
                "opt_state": opt_state
            }
        else:
            raise ValueError(f"Invalid algorithm in checkpoint: {config['algorithm']}")
        return common, algorithm_kw_args

    @staticmethod
    def save_buffer(buffer_state: TrajectoryBufferState | PrioritisedTrajectoryBufferState, checkpoint_dir: str, checkpoint_name: str) -> dict:
        buffer_dir = os.path.join(checkpoint_dir, checkpoint_name + "_buffer_state")
        os.makedirs(buffer_dir, exist_ok=True)

        if isinstance(buffer_state, PrioritisedTrajectoryBufferState):
            priority_state_path = os.path.join(buffer_dir, "buffer_priority_state")
            Checkpointer._save_sum_tree_state(buffer_state.priority_state, priority_state_path)
        else:
            priority_state_path = ""

        vault_uuid = datetime.now().strftime("%Y%m%d%H%M%S")
        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=buffer_state.experience,
            rel_dir=buffer_dir,
            vault_uid=vault_uuid
        )

        # write buffer
        _fbx_shape = jax.tree_util.tree_leaves(buffer_state.experience)[0].shape
        buffer_size = _fbx_shape[1]
        source_interval = (0, buffer_size)
        v.write(buffer_state, source_interval=source_interval)

        return {
            "vault_uuid": vault_uuid,
            "buffer_dir": buffer_dir,
            "priority_state_path": priority_state_path
        }

    @staticmethod
    def load_buffer(dummy_buffer_state: PrioritisedTrajectoryBufferState, priority_state_path: str, buffer_dir: str, vault_uuid: str) -> TrajectoryBufferState | PrioritisedTrajectoryBufferState:
        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=dummy_buffer_state.experience,
            rel_dir=buffer_dir,
            vault_uid=vault_uuid
        )
        buffer_state = v.read()

        if priority_state_path != "":
            priority_state = Checkpointer._load_sum_tree_state(priority_state_path)

            return PrioritisedTrajectoryBufferState(
                experience=buffer_state.experience,
                current_index=buffer_state.current_index,
                is_full=buffer_state.is_full,
                priority_state=priority_state
            )
        else:
            return TrajectoryBufferState(
                experience=buffer_state.experience,
                current_index=buffer_state.current_index,
                is_full=buffer_state.is_full,
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
        with open(os.path.join(directory, Checkpointer.SCALARS_FILE)) as json_file:
            scalar_values = json.load(json_file)

        # Reconstruct the SumTreeState object
        return SumTreeState(
            nodes=nodes,
            max_recorded_priority=max_recorded_priority,
            tree_depth=scalar_values["tree_depth"],
            capacity=scalar_values["capacity"],
        )
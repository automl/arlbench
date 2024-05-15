from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flashbax.buffers.prioritised_trajectory_buffer import (
    PrioritisedTrajectoryBufferState,
)
from flashbax.buffers.sum_tree import SumTreeState
from flashbax.vault import Vault
from flax.core.frozen_dict import FrozenDict

from optax import ScaleByAdamState, EmptyState
import warnings
from arlbench.core.algorithms import DQN, PPO, SAC
from arlbench.core.algorithms.dqn import DQNRunnerState, DQNTrainingResult
from arlbench.core.algorithms.ppo import PPORunnerState, PPOTrainingResult
from arlbench.core.algorithms.sac import SACRunnerState, SACTrainingResult

if TYPE_CHECKING:
    from ConfigSpace import Configuration
    from flashbax.buffers.trajectory_buffer import TrajectoryBufferState

    from arlbench.core.algorithms import AlgorithmState, TrainResult


VALID_CHECKPOINTS = ["opt_state", "params", "loss", "buffer"]


class Checkpointer:
    NODES_FILE = "nodes.npy"
    SCALARS_FILE = "scalars.json"
    MRP_FILE = "max_recorded_priority.npy"

    @staticmethod
    def _save_orbax_checkpoint(
        checkpoint: dict[str, Any], checkpoint_dir: str, checkpoint_name: str
    ) -> None:
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            os.path.join(checkpoint_dir, checkpoint_name),
            checkpoint,
            force=True,
        )

    @staticmethod
    def _load_orbax_checkpoint(
        checkpoint_dir: str, checkpoint_name: str
    ) -> dict[str, Any]:
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(os.path.join(checkpoint_dir, checkpoint_name))

    @staticmethod
    def save(
        algorithm: str,
        algorithm_state: AlgorithmState,
        autorl_config: dict,
        hp_config: Configuration,
        done: bool,
        c_episode: int,
        c_step: int,
        train_result: TrainResult | None,
        tag: str | None = None,
    ) -> str:
        # Checkpoint setup
        checkpoint = autorl_config["checkpoint"]  # list of strings
        if "all" in checkpoint:
            checkpoint = ["all"]

        checkpoint_name = autorl_config["checkpoint_name"]
        checkpoint_dir = os.path.join(autorl_config["checkpoint_dir"], checkpoint_name)
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

        runner_state = algorithm_state.runner_state

        if (
            algorithm == "dqn"
            and isinstance(runner_state, DQNRunnerState)
            and isinstance(train_result, DQNTrainingResult)
        ):
            algorithm_ckpt = DQN.get_checkpoint_factory(runner_state, train_result)
        elif (
            algorithm == "ppo"
            and isinstance(runner_state, PPORunnerState)
            and isinstance(train_result, PPOTrainingResult)
        ):
            algorithm_ckpt = PPO.get_checkpoint_factory(runner_state, train_result)
        elif (
            algorithm == "sac"
            and isinstance(runner_state, SACRunnerState)
            and isinstance(train_result, SACTrainingResult)
        ):
            algorithm_ckpt = SAC.get_checkpoint_factory(runner_state, train_result)
        else:
            raise ValueError(
                f"Invalid type of runner state or training result for {algorithm}: {type(runner_state)}, {type(train_result)}."
            )

        ckpt: dict[str, Any] = {
            "autorl_config": autorl_config,
            "hp_config": dict(hp_config),
            "c_step": c_step,
            "c_episode": c_episode,
        }

        if checkpoint == ["all"]:
            # use all available checkpoint options
            for key in algorithm_ckpt:
                ckpt[key] = algorithm_ckpt[
                    key
                ]()  # get actual checkpoint by calling factory function
        else:
            # only use selected checkpoint options
            for key in checkpoint:
                if key == "buffer":
                    if algorithm in ["dqn", "sac"]:
                        continue
                    else:
                        warnings.warn(
                            f"Invalid checkpoint for algorithm {algorithm}: {key}. Valid keys are {list(algorithm_ckpt.keys())!s}. Skipping key."
                        )

                # find all algorithm checkpoints that contain the requested key
                found_key = False
                for algorithm_key in algorithm_ckpt:
                    if key in algorithm_key:
                        found_key = True
                        ckpt[algorithm_key] = algorithm_ckpt[
                            algorithm_key
                        ]()  # get actual checkpoint by calling factory function
                if not found_key:
                    warnings.warn(
                        f"Invalid checkpoint for algorithm {algorithm}: {key}. Valid keys are {list(algorithm_ckpt.keys())!s}. Skipping key."
                    )

        Checkpointer._save_orbax_checkpoint(ckpt, checkpoint_dir, checkpoint_name)

        if "buffer" in checkpoint and algorithm_state.buffer_state is not None:
            ckpt["buffer"] = Checkpointer.save_buffer(
                algorithm_state.buffer_state, checkpoint_dir, checkpoint_name
            )

        return os.path.join(checkpoint_dir, checkpoint_name)

    @staticmethod
    def _load_buffer(
        ckpt: dict[str, Any], dummy_buffer_state: PrioritisedTrajectoryBufferState
    ) -> PrioritisedTrajectoryBufferState | None:
        if "buffer" in ckpt:
            return Checkpointer.load_buffer(
                dummy_buffer_state,
                ckpt["buffer"]["priority_state_path"],
                ckpt["buffer"]["buffer_dir"],
                ckpt["buffer"]["vault_uuid"],
            )
        else:
            return None

    @staticmethod
    def _load_params(ckpt: dict[str, Any], key: str) -> FrozenDict | None:
        if key not in ckpt or ckpt[key] is None:
            return None
        else:
            params = ckpt[key]
            if isinstance(params, list):
                params = params[0]
            return FrozenDict(params)

    @staticmethod
    def _load_adam_opt_state(ckpt: dict[str, Any], key: str) -> tuple | None:
        def apply(func, t: dict | tuple):
            if isinstance(t, tuple) or isinstance(t, list):
                return tuple(apply(func, item) for item in t)
            else:
                return func(t)

        def make_opt_state(opt_state: dict | None) -> ScaleByAdamState | EmptyState:
            if opt_state is None:
                return EmptyState()
            else:
                return ScaleByAdamState(
                    count=opt_state["count"],
                    mu=FrozenDict(opt_state["mu"]),
                    nu=FrozenDict(opt_state["nu"]),
                )

        if key not in ckpt or ckpt[key] is None:
            return None
        else:
            opt_state = ckpt[key]

            return apply(make_opt_state, opt_state)

    @staticmethod
    def load(
        checkpoint_path: str, algorithm_state: AlgorithmState
    ) -> tuple[tuple[dict[str, Any], int, int], dict]:
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(checkpoint_path)

        c_step = restored["c_step"]
        c_episode = restored["c_episode"]
        autorl_config = restored["autorl_config"]
        hp_config = restored["hp_config"]

        if algorithm_state.buffer_state is not None:
            buffer_state = Checkpointer._load_buffer(
                restored, algorithm_state.buffer_state
            )
        else:
            buffer_state = None

        common = (hp_config, c_step, c_episode)

        if autorl_config["algorithm"] == "ppo":
            algorithm_kw_args = {
                "network_params": Checkpointer._load_params(restored, "params"),
                "opt_state": Checkpointer._load_adam_opt_state(restored, "opt_state"),
            }
        elif autorl_config["algorithm"] == "dqn":
            algorithm_kw_args = {
                "buffer_state": buffer_state,
                "network_params": Checkpointer._load_params(restored, "params"),
                "target_params": Checkpointer._load_params(restored, "target_params"),
                "opt_state": Checkpointer._load_adam_opt_state(restored, "opt_state"),
            }
        elif autorl_config["algorithm"] == "sac":
            algorithm_kw_args = {
                "buffer_state": buffer_state,
                "actor_network_params": Checkpointer._load_params(
                    restored, "actor_network_params"
                ),
                "critic_network_params": Checkpointer._load_params(
                    restored, "critic_network_params"
                ),
                "critic_target_params": Checkpointer._load_params(
                    restored, "critic_target_params"
                ),
                "alpha_network_params": Checkpointer._load_params(
                    restored, "alpha_network_params"
                ),
                "actor_opt_state": Checkpointer._load_adam_opt_state(
                    restored, "actor_opt_state"
                ),
                "critic_opt_state": Checkpointer._load_adam_opt_state(
                    restored, "critic_opt_state"
                ),
                "alpha_opt_state": Checkpointer._load_adam_opt_state(
                    restored, "alpha_opt_state"
                ),
            }
        else:
            raise ValueError(
                f"Invalid algorithm in checkpoint: {autorl_config['algorithm']}"
            )
        return common, algorithm_kw_args

    @staticmethod
    def save_buffer(
        buffer_state: TrajectoryBufferState | PrioritisedTrajectoryBufferState,
        checkpoint_dir: str,
        checkpoint_name: str,
    ) -> dict:
        buffer_dir = os.path.join(checkpoint_dir, checkpoint_name + "_buffer_state")
        os.makedirs(buffer_dir, exist_ok=True)

        if isinstance(buffer_state, PrioritisedTrajectoryBufferState):
            priority_state_path = os.path.join(buffer_dir, "buffer_priority_state")
            Checkpointer._save_sum_tree_state(
                buffer_state.priority_state, priority_state_path
            )
        else:
            priority_state_path = ""

        vault_uuid = datetime.now().strftime("%Y%m%d%H%M%S")
        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=buffer_state.experience,
            rel_dir=buffer_dir,
            vault_uid=vault_uuid,
        )

        # write buffer
        _fbx_shape = jax.tree_util.tree_leaves(buffer_state.experience)[0].shape
        buffer_size = _fbx_shape[1]
        source_interval = (0, buffer_size)
        v.write(buffer_state, source_interval=source_interval)

        return {
            "vault_uuid": vault_uuid,
            "buffer_dir": buffer_dir,
            "priority_state_path": priority_state_path,
        }

    @staticmethod
    def load_buffer(
        dummy_buffer_state: PrioritisedTrajectoryBufferState,
        priority_state_path: str,
        buffer_dir: str,
        vault_uuid: str,
    ) -> PrioritisedTrajectoryBufferState:
        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=dummy_buffer_state.experience,
            rel_dir=buffer_dir,
            vault_uid=vault_uuid,
        )
        buffer_state = v.read()
        priority_state = Checkpointer._load_sum_tree_state(priority_state_path)

        return PrioritisedTrajectoryBufferState(
            experience=buffer_state.experience,
            current_index=buffer_state.current_index,
            is_full=buffer_state.is_full,
            priority_state=priority_state,
        )

    @staticmethod
    def _save_sum_tree_state(sum_tree_state: SumTreeState, directory: str) -> None:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Serialize and save the JAX arrays as .npy files
        np.save(
            os.path.join(directory, Checkpointer.NODES_FILE),
            np.array(sum_tree_state.nodes),
        )
        np.save(
            os.path.join(directory, Checkpointer.MRP_FILE),
            np.array(sum_tree_state.max_recorded_priority),
        )

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
        max_recorded_priority = jnp.array(
            np.load(os.path.join(directory, Checkpointer.MRP_FILE))
        )

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

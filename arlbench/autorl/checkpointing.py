"""Contains all checkpointing-related methods for the AutoRL environment."""
from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
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
from omegaconf import DictConfig, ListConfig, OmegaConf
from optax import EmptyState, ScaleByAdamState

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
    """Contains all checkpointing-related methods for the AutoRL environment."""
    NODES_FILE = "nodes.npy"
    SCALARS_FILE = "scalars.json"
    MRP_FILE = "max_recorded_priority.npy"

    @staticmethod
    def _save_orbax_checkpoint(
        checkpoint: dict[str, Any], checkpoint_dir: str, checkpoint_name: str
    ) -> None:
        """Stores a dictionary using orbax.

        Args:
            checkpoint (dict[str, Any]): Dictionary to store.
            checkpoint_dir (str): Checkpoint directory.
            checkpoint_name (str): Checkpoint name.
        """
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
        """Load a dictionary using orbax.

        Args:
            checkpoint_dir (str): Checkpoint directory.
            checkpoint_name (str): Checkpoint name.

        Returns:
            dict[str, Any]: Dictionary to load.
        """
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
        """Saves the current state of a AutoRL environment.

        Args:
            algorithm (str): Name of the algorithm.
            algorithm_state (AlgorithmState): Current algorithm state.
            autorl_config (dict): AutoRL configuration.
            hp_config (Configuration): Hyperparameter configuration of the algorithm.
            done (bool): Whether the environment is done.
            c_episode (int): Current episode of the AutoRL environment.
            c_step (int):  Current step of the AutoRL environment.
            train_result (TrainResult | None): Last training result of the algorithm.
            tag (str | None, optional): Checkpoint tag which is appended to the checkpoint name. Defaults to None.

        Returns:
            str: Path of the checkpoint.
        """
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

        # Append tag for error/sigterm
        if tag is not None:
            checkpoint_name += f"_{tag}"

        runner_state = algorithm_state.runner_state

        # We get the factory functions depending on the algorithm at hand
        # This allows us to just call all functions without knowing which algorithm we are using
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

        # First, we store the basic information of the AutoRL environment
        for k in autorl_config:
            if isinstance(autorl_config[k], DictConfig | ListConfig):
                autorl_config[k] = OmegaConf.to_container(autorl_config[k], resolve=True)
        ckpt: dict[str, Any] = {
            "autorl_config": autorl_config,
            "hp_config": dict(hp_config),
            "c_step": c_step,
            "c_episode": c_episode,
        }

        if checkpoint == ["all"]:
            # Use all available checkpoint options
            for key in algorithm_ckpt:
                # Get actual checkpoint by calling factory function
                ckpt[key] = algorithm_ckpt[
                    key
                ]()
        else:
            # Only use selected checkpoint options
            for key in checkpoint:
                if key == "buffer":
                    if algorithm in ["dqn", "sac"]:
                        continue
                    else:
                        warnings.warn(
                            f"Invalid checkpoint for algorithm {algorithm}: {key}. Valid keys are {list(algorithm_ckpt.keys())!s}. Skipping key."
                        )

                # Find all algorithm checkpoints that contain the requested key
                found_key = False
                for algorithm_key in algorithm_ckpt:
                    if key in algorithm_key:
                        found_key = True
                        # Get actual checkpoint by calling factory function
                        ckpt[algorithm_key] = algorithm_ckpt[
                            algorithm_key
                        ]()
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
        """Loads the buffer state from a given checkpoint.

        Args:
            ckpt (dict[str, Any]): Checkpoint containing the buffer info.
            dummy_buffer_state (PrioritisedTrajectoryBufferState): Dummy state that is required to know the buffer dimensions.

        Returns:
            PrioritisedTrajectoryBufferState | None: Buffer state if contained in the checkpoint.
        """
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
        """Load network parameters from a give checkpoint.

        Args:
            ckpt (dict[str, Any]):  Checkpoint containing the parameter info.
            key (str): Key of the parameters in checkpoint dictionary.

        Returns:
            FrozenDict | None: Network parameters.
        """
        if key not in ckpt or ckpt[key] is None:
            return None
        else:
            params = ckpt[key]
            if isinstance(params, list):
                # Sometimes we have a list of parameters which only contains one element
                # so we get rid of the list
                params = params[0]
            return FrozenDict(params)

    @staticmethod
    def _load_adam_opt_state(ckpt: dict[str, Any], key: str) -> tuple | None:
        """Load the state of a adam optimizer from a given checkpoint.

        Args:
            ckpt (dict[str, Any]): Checkpoint dictionary containing optimizer state.
            key (str): Key of the optimizer state in the checkpoint dictionary.

        Returns:
            tuple | None: Optimizer state.
        """
        def apply(func: Callable, t: dict | list | tuple) -> tuple| Any:
            """Applies a given fuction to each element of a dict/list/tuple.

            Args:
                func (Callable): Function to apply.
                t (dict | tuple): Tuple/list/dict to apply function to.

            Returns:
                tuple| Any: Input element with function applied to each element of list/tuple or dict itself.
            """
            if isinstance(t, list | tuple):
                return tuple(apply(func, item) for item in t)
            else:
                return func(t)

        def make_opt_state(opt_state: dict | None) -> ScaleByAdamState | EmptyState:
            """Converts a dictionary containing an optimizer state to an actual OptState.

            Args:
                opt_state (dict | None): Dictionary containing optimizer state.

            Returns:
                ScaleByAdamState | EmptyState: Actual optimizer state.
            """
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
        """Loads a AutoRL environment checkpoint.

        Args:
            checkpoint_path (str): Path of the checkpoint.
            algorithm_state (AlgorithmState): Current algorithm state, certain attributes will be overriden by checkpoint.

        Returns:
            tuple[tuple[dict[str, Any], int, int], dict]: Common AutoRL environment attributes as well as dictionary to restored algorithm state: (hp_config, c_step, c_episode), algorithm_kw_args
        """
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
        """Saves the buffer state of an algorithm.

        Args:
            buffer_state (TrajectoryBufferState | PrioritisedTrajectoryBufferState): Buffer state.
            checkpoint_dir (str): Checkpoint directory.
            checkpoint_name (str): Checkpoint name.

        Returns:
            dict: Dictionary containing the identifiers of single parts of the buffer. Required to load the checkpoint.
        """
        buffer_dir = os.path.join(checkpoint_dir, checkpoint_name + "_buffer_state")
        os.makedirs(buffer_dir, exist_ok=True)

        if isinstance(buffer_state, PrioritisedTrajectoryBufferState):
            # In this case, we have to store the priorities separately since
            # the vault does not store them
            priority_state_path = os.path.join(buffer_dir, "buffer_priority_state")
            Checkpointer._save_sum_tree_state(
                buffer_state.priority_state, priority_state_path
            )
        else:
            priority_state_path = ""

        # We are using a vault to store the buffer itself
        # However, this does not include the priorities
        vault_uuid = datetime.now().strftime("%Y%m%d%H%M%S")
        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=buffer_state.experience,
            rel_dir=buffer_dir,
            vault_uid=vault_uuid,
        )

        # Now we are writing the actual buffer state to disk
        _fbx_shape = jax.tree_util.tree_leaves(buffer_state.experience)[0].shape
        buffer_size = _fbx_shape[1]
        source_interval = (0, buffer_size)
        v.write(buffer_state, source_interval=source_interval)

        # This is required in the load_buffer() method to reconstructed all
        # the bits and pieces
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
        """Loads the buffer state from a checkpoint.

        Args:
            dummy_buffer_state (PrioritisedTrajectoryBufferState): Dummy buffer state. This is required to know the size/data types of the buffer.
            priority_state_path (str): Path where the priorities are stored.
            buffer_dir (str): The directory where the buffer data is stored.
            vault_uuid (str): The unique ID of the vault containing the buffer data.

        Returns:
            PrioritisedTrajectoryBufferState: The buffer state that was loaded from disk.
        """
        # Using the vault we can easily load the data of the buffer
        # As described in the part of saving the buffer, this does
        # not contain the priorities
        v = Vault(
            vault_name="buffer_state_vault",
            experience_structure=dummy_buffer_state.experience,
            rel_dir=buffer_dir,
            vault_uid=vault_uuid,
        )
        buffer_state = v.read()

        # We have to load the priorities separately from the file we stored them
        priority_state = Checkpointer._load_sum_tree_state(priority_state_path)

        # Now we can reassamble the buffer state
        return PrioritisedTrajectoryBufferState(
            experience=buffer_state.experience,
            current_index=buffer_state.current_index,
            is_full=buffer_state.is_full,
            priority_state=priority_state,
        )

    @staticmethod
    def _save_sum_tree_state(sum_tree_state: SumTreeState, directory: str) -> None:
        """Saves the state of a JAX sum stree to disk.

        Args:
            sum_tree_state (SumTreeState): State of a JAX sum tree.
            directory (str): Directory to store state in.
        """
        os.makedirs(directory, exist_ok=True)

        # We store each property of the tree state separately since it
        # is not serializable as a whole
        np.save(
            os.path.join(directory, Checkpointer.NODES_FILE),
            np.array(sum_tree_state.nodes),
        )
        np.save(
            os.path.join(directory, Checkpointer.MRP_FILE),
            np.array(sum_tree_state.max_recorded_priority),
        )

        # We do the same for scalar values
        scalar_values = {
            "tree_depth": sum_tree_state.tree_depth,
            "capacity": sum_tree_state.capacity,
        }
        with open(os.path.join(directory, Checkpointer.SCALARS_FILE), "w") as json_file:
            json.dump(scalar_values, json_file)

    @staticmethod
    def _load_sum_tree_state(directory: str) -> SumTreeState:
        """Loads the state of a JAX sum tree.

        Args:
            directory (str): Directory containing the saved tree state.

        Returns:
            SumTreeState: JAX sum tree state.
        """
        # First, we load the nodes and maximum recorded priority from file
        # and convert them to JAX arrays again
        nodes = jnp.array(np.load(os.path.join(directory, Checkpointer.NODES_FILE)))
        max_recorded_priority = jnp.array(
            np.load(os.path.join(directory, Checkpointer.MRP_FILE))
        )

        # Scalars can be read easily
        with open(os.path.join(directory, Checkpointer.SCALARS_FILE)) as json_file:
            scalar_values = json.load(json_file)

        # Now we can manually reconstruct the SumTreeState object
        return SumTreeState(
            nodes=nodes,
            max_recorded_priority=max_recorded_priority,
            tree_depth=scalar_values["tree_depth"],
            capacity=scalar_values["capacity"],
        )

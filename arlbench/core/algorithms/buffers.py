# From https://github.com/instadeepai/flashbax/blob/main/flashbax/buffers/prioritised_trajectory_buffer.py
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flashbax import utils
from flashbax.buffers import sum_tree
from flashbax.buffers.prioritised_flat_buffer import (
    ExperiencePair, PrioritisedTransitionSample, TransitionSample)
from flashbax.buffers.prioritised_trajectory_buffer import (
    Experience, PrioritisedTrajectoryBufferSample,
    PrioritisedTrajectoryBufferState, _get_sample_trajectories,
    get_invalid_indices)
from flashbax.buffers.trajectory_buffer import calculate_uniform_item_indices

if TYPE_CHECKING:
    import chex


# adapted from flashbax.buffers.trajectory_buffer.sample
def uniform_sample(
    state: PrioritisedTrajectoryBufferState[Experience],
    rng_key: chex.PRNGKey,
    batch_size: int,
    sequence_length: int,
    period: int,
) -> TransitionSample:
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )
    # Calculate the indices of the items that will be sampled.
    item_indices = calculate_uniform_item_indices(
        state,
        rng_key,
        batch_size,
        sequence_length,
        period,
        add_batch_size,
        max_length_time_axis,
    )

    trajectory = _get_sample_trajectories(
        item_indices, max_length_time_axis, period, sequence_length, state
    )

    # There is an edge case where experience from the sum-tree has probability 0.
    # To deal with this we overwrite indices with probability zero with
    # the index that is the most probable within the batch of indices. This slightly biases
    # the sampling, however as this is an edge case it is unlikely to have a significant effect.
    priorities = sum_tree.get(state.priority_state, item_indices)
    most_probable_in_batch_index = jnp.argmax(priorities)
    item_indices = jnp.where(
        priorities == 0, item_indices[most_probable_in_batch_index], item_indices
    )
    priorities = jnp.where(
        priorities == 0, priorities[most_probable_in_batch_index], priorities
    )

    # We get the indices of the items that will be invalid when sampling from the buffer state.
    # If the sampled indices are in the invalid indices, then we replace them with the
    # most probable index in the batch. As with above this is unlikely to occur.
    invalid_item_indices = get_invalid_indices(
        state, sequence_length, period, add_batch_size, max_length_time_axis
    )
    invalid_item_indices = invalid_item_indices.flatten()
    item_indices = jnp.where(
        jnp.any(item_indices[:, None] == invalid_item_indices, axis=-1),
        item_indices[most_probable_in_batch_index],
        item_indices,
    )
    priorities = jnp.where(
        jnp.any(item_indices[:, None] == invalid_item_indices, axis=-1),
        priorities[most_probable_in_batch_index],
        priorities,
    )

    sampled_batch = PrioritisedTrajectoryBufferSample(
        experience=trajectory, indices=item_indices, priorities=priorities
    )
    first = jax.tree_util.tree_map(lambda x: x[:, 0], sampled_batch.experience)
    second = jax.tree_util.tree_map(lambda x: x[:, 1], sampled_batch.experience)
    return PrioritisedTransitionSample(
        experience=ExperiencePair(first=first, second=second),
        indices=sampled_batch.indices,
        priorities=sampled_batch.priorities,
    )

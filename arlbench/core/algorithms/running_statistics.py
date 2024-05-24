# This file includes code from brax, licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

from typing import Any, Optional, Tuple, Iterable, Mapping, Union
import dataclasses

from flax import struct
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Array:
  """Describes a numpy array or scalar shape and dtype.

  Similar to dm_env.specs.Array.
  """
  shape: Tuple[int, ...]
  dtype: jnp.dtype


# Define types for nested arrays and tensors.
NestedArray = jnp.ndarray
NestedTensor = Any

# pytype: disable=not-supported-yet
NestedSpec = Union[
  Array,
  Iterable['NestedSpec'],
  Mapping[Any, 'NestedSpec'],
]
# pytype: enable=not-supported-yet

Nest = Union[NestedArray, NestedTensor, NestedSpec]


def _zeros_like(nest: Nest, dtype=None) -> Nest:
  return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def _ones_like(nest: Nest, dtype=None) -> Nest:
  return jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


@struct.dataclass
class NestedMeanStd:
  """A container for running statistics (mean, std) of possibly nested data."""
  mean: Nest
  std: Nest


@struct.dataclass
class RunningStatisticsState(NestedMeanStd):
  """Full state of running statistics computation."""
  count: jnp.ndarray
  summed_variance: Nest


def init_state(nest: Nest) -> RunningStatisticsState:
  """Initializes the running statistics for the given nested structure."""
  dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

  return RunningStatisticsState(
      count=jnp.zeros((), dtype=dtype),
      mean=_zeros_like(nest, dtype=dtype),
      summed_variance=_zeros_like(nest, dtype=dtype),
      # Initialize with ones to make sure normalization works correctly
      # in the initial state.
      std=_ones_like(nest, dtype=dtype))


def _validate_batch_shapes(batch: NestedArray,
                           reference_sample: NestedArray,
                           batch_dims: Tuple[int, ...]) -> None:
  """Verifies shapes of the batch leaves against the reference sample.

  Checks that batch dimensions are the same in all leaves in the batch.
  Checks that non-batch dimensions for all leaves in the batch are the same
  as in the reference sample.

  Arguments:
    batch: the nested batch of data to be verified.
    reference_sample: the nested array to check non-batch dimensions.
    batch_dims: a Tuple of indices of batch dimensions in the batch shape.

  Returns:
    None.
  """
  def validate_node_shape(reference_sample: jnp.ndarray,
                          batch: jnp.ndarray) -> None:
    expected_shape = batch_dims + reference_sample.shape
    assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'

  jax.tree_util.tree_map(validate_node_shape, reference_sample, batch)


def update(state: RunningStatisticsState,
           batch: Nest,
           *,
           weights: Optional[jnp.ndarray] = None,
           std_min_value: float = 1e-6,
           std_max_value: float = 1e6,
           pmap_axis_name: Optional[str] = None,
           validate_shapes: bool = True) -> RunningStatisticsState:
  """Updates the running statistics with the given batch of data.

  Note: data batch and state elements (mean, etc.) must have the same structure.

  Note: by default will use int32 for counts and float32 for accumulated
  variance. This results in an integer overflow after 2^31 data points and
  degrading precision after 2^24 batch updates or even earlier if variance
  updates have large dynamic range.
  To improve precision, consider setting jax_enable_x64 to True, see
  https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

  Arguments:
    state: The running statistics before the update.
    batch: The data to be used to update the running statistics.
    weights: Weights of the batch data. Should match the batch dimensions.
      Passing a weight of 2. should be equivalent to updating on the
      corresponding data point twice.
    std_min_value: Minimum value for the standard deviation.
    std_max_value: Maximum value for the standard deviation.
    pmap_axis_name: Name of the pmapped axis, if any.
    validate_shapes: If true, the shapes of all leaves of the batch will be
      validated. Enabled by default. Doesn't impact performance when jitted.

  Returns:
    Updated running statistics.
  """
  # We require exactly the same structure to avoid issues when flattened
  # batch and state have different order of elements.
  assert jax.tree_util.tree_structure(batch) == jax.tree_util.tree_structure(state.mean)
  batch_shape = jax.tree_util.tree_leaves(batch)[0].shape
  # We assume the batch dimensions always go first.
  batch_dims = batch_shape[:len(batch_shape) -
                           jax.tree_util.tree_leaves(state.mean)[0].ndim]
  batch_axis = range(len(batch_dims))
  if weights is None:
    step_increment = jnp.prod(jnp.array(batch_dims))
  else:
    step_increment = jnp.sum(weights)
  if pmap_axis_name is not None:
    step_increment = jax.lax.psum(step_increment, axis_name=pmap_axis_name)
  count = state.count + step_increment

  # Validation is important. If the shapes don't match exactly, but are
  # compatible, arrays will be silently broadcasted resulting in incorrect
  # statistics.
  if validate_shapes:
    if weights is not None:
      if weights.shape != batch_dims:
        raise ValueError(f'{weights.shape} != {batch_dims}')
    _validate_batch_shapes(batch, state.mean, batch_dims)

  def _compute_node_statistics(
      mean: jnp.ndarray, summed_variance: jnp.ndarray,
      batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert isinstance(mean, jnp.ndarray), type(mean)
    assert isinstance(summed_variance, jnp.ndarray), type(summed_variance)
    # The mean and the sum of past variances are updated with Welford's
    # algorithm using batches (see https://stackoverflow.com/q/56402955).
    diff_to_old_mean = batch - mean
    if weights is not None:
      expanded_weights = jnp.reshape(
          weights,
          list(weights.shape) + [1] * (batch.ndim - weights.ndim))
      diff_to_old_mean = diff_to_old_mean * expanded_weights
    mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / count
    if pmap_axis_name is not None:
      mean_update = jax.lax.psum(
          mean_update, axis_name=pmap_axis_name)
    mean = mean + mean_update

    diff_to_new_mean = batch - mean
    variance_update = diff_to_old_mean * diff_to_new_mean
    variance_update = jnp.sum(variance_update, axis=batch_axis)
    if pmap_axis_name is not None:
      variance_update = jax.lax.psum(variance_update, axis_name=pmap_axis_name)
    summed_variance = summed_variance + variance_update
    return mean, summed_variance

  updated_stats = jax.tree_util.tree_map(_compute_node_statistics, state.mean,
                                         state.summed_variance, batch)
  # Extract `mean` and `summed_variance` from `updated_stats` nest.
  mean = jax.tree_util.tree_map(lambda _, x: x[0], state.mean, updated_stats)
  summed_variance = jax.tree_util.tree_map(lambda _, x: x[1], state.mean,
                                           updated_stats)

  def compute_std(summed_variance: jnp.ndarray,
                  std: jnp.ndarray) -> jnp.ndarray:
    assert isinstance(summed_variance, jnp.ndarray)
    # Summed variance can get negative due to rounding errors.
    summed_variance = jnp.maximum(summed_variance, 0)
    std = jnp.sqrt(summed_variance / count)
    std = jnp.clip(std, std_min_value, std_max_value)
    return std

  std = jax.tree_util.tree_map(compute_std, summed_variance, state.std)

  return RunningStatisticsState(
      count=count, mean=mean, summed_variance=summed_variance, std=std)


def normalize(batch: NestedArray,
              mean_std: NestedMeanStd,
              max_abs_value: Optional[float] = None) -> NestedArray:
  """Normalizes data using running statistics."""

  def normalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                     std: jnp.ndarray) -> jnp.ndarray:
    # Only normalize inexact
    if not jnp.issubdtype(data.dtype, jnp.inexact):
      return data
    data = (data - mean) / std
    if max_abs_value is not None:
      # TODO: remove pylint directive
      data = jnp.clip(data, -max_abs_value, +max_abs_value)
    return data

  return jax.tree_util.tree_map(normalize_leaf, batch, mean_std.mean, mean_std.std)


def denormalize(batch: NestedArray,
                mean_std: NestedMeanStd) -> NestedArray:
  """Denormalizes values in a nested structure using the given mean/std.

  Only values of inexact types are denormalized.
  See https://numpy.org/doc/stable/_images/dtype-hierarchy.png for Numpy type
  hierarchy.

  Args:
    batch: a nested structure containing batch of data.
    mean_std: mean and standard deviation used for denormalization.

  Returns:
    Nested structure with denormalized values.
  """

  def denormalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                       std: jnp.ndarray) -> jnp.ndarray:
    # Only denormalize inexact
    if not jnp.issubdtype(data.dtype, jnp.inexact):
      return data
    return data * std + mean

  return jax.tree_util.tree_map(denormalize_leaf, batch, mean_std.mean, mean_std.std)


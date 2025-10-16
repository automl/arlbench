"""SAC models for the actor and critic networks."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax.nn.initializers import constant, ones, orthogonal, zeros

PRNGKey = Any
Array = Any
Shape = tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]

class BatchRenorm(nn.Module):
  """BatchRenorm Module from the original CrossQ Code, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
  and adapted from Flax's BatchNorm implementation:
  https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228.


  Attributes:
    use_running_average: if True, the statistics stored in batch_stats will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch
      statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  use_running_average: bool | None = None
  axis: int = -1
  momentum: float = 0.999
  epsilon: float = 0.001
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = ones
  axis_name: str | None = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True

  @nn.compact
  def __call__(self, x, use_running_average: bool | None = None):
    """Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    use_running_average = nn.merge_param(
        "use_running_average", self.use_running_average, use_running_average
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    ra_mean = self.variable(
        "batch_stats",
        "mean",
        lambda s: jnp.zeros(s, jnp.float32),
        feature_shape,
    )
    ra_var = self.variable(
        "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
    )

    r_max = self.variable(
        "batch_stats",
        "r_max",
        lambda s: s,
        3,
    )
    d_max = self.variable(
        "batch_stats",
        "d_max",
        lambda s: s,
        5,
    )
    steps = self.variable(
        "batch_stats",
        "steps",
        lambda s: s,
        0,
    )

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
      custom_mean = mean
      custom_var = var
    else:
      mean, var = _compute_stats(
          x,
          reduction_axes,
          dtype=self.dtype,
          axis_name=self.axis_name if not self.is_initializing() else None,
          axis_index_groups=self.axis_index_groups,
          use_fast_variance=self.use_fast_variance,
      )
      custom_mean = mean
      custom_var = var
      if not self.is_initializing():
        # The code below is implemented following the Batch Renormalization paper
        r = 1
        d = 0
        std = jnp.sqrt(var + self.epsilon)
        ra_std = jnp.sqrt(ra_var.value + self.epsilon)
        r = jax.lax.stop_gradient(std / ra_std)
        r = jnp.clip(r, 1 / r_max.value, r_max.value)
        d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
        d = jnp.clip(d, -d_max.value, d_max.value)
        tmp_var = var / (r**2)
        tmp_mean = mean - d * jnp.sqrt(custom_var) / r

        # Warm up batch renorm for 100_000 steps to build up proper running statistics
        warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
        custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
        custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

        ra_mean.value = (
            self.momentum * ra_mean.value + (1 - self.momentum) * mean
        )
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
        steps.value += 1



    return _normalize(
        self,
        x,
        custom_mean,
        custom_var,
        reduction_axes,
        feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        self.use_bias,
        self.use_scale,
        self.bias_init,
        self.scale_init,
    )

class TanhTransformedDistribution(distrax.Transformed):  # type: ignore[name-defined]
    """Tanh transformation of a distrax distribution."""

    def __init__(self, distribution):  # type: ignore[name-defined]
        """Initializes the Tanh transformation of a distribution."""
        super().__init__(
            distribution=distribution, bijector=distrax.Block(distrax.Tanh(), 1)
        )

    def mode(self) -> jnp.ndarray:
        """Returns the mode of the distribution."""
        return self.bijector.forward(self.distribution.mode())


class AlphaCoef(nn.Module):
    """Alpha coefficient for SAC."""

    alpha_init: float = 1.0

    def setup(self):
        """Initializes the alpha coefficient."""
        self.log_alpha = self.param(
            "log_alpha", init_fn=lambda rng: jnp.full((), jnp.log(self.alpha_init)) # noqa: ARG005
        )

    def __call__(self) -> jnp.ndarray:
        """Returns the alpha coefficient."""
        return jnp.exp(self.log_alpha)


class SACMLPActor(nn.Module):
    """An MLP-based actor network for SAC."""

    action_dim: int
    activation: int
    hidden_size: int = 64
    log_std_min: float = -20
    log_std_max: float = 2

    def setup(self):
        """Initializes the actor network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.dense0 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.dense1 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.mean_out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )
        self.log_std_out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

    def __call__(self, x):
        """Applies the actor to the input."""
        actor_hidden = self.dense0(x)
        actor_hidden = self.activation_func(actor_hidden)
        actor_hidden = self.dense1(actor_hidden)
        actor_hidden = self.activation_func(actor_hidden)
        actor_mean = self.mean_out_layer(actor_hidden)
        actor_logstd = self.log_std_out_layer(actor_hidden)
        actor_logstd = jnp.clip(actor_logstd, self.log_std_min, self.log_std_max)

        return TanhTransformedDistribution(
            distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        )


class SACCNNActor(nn.Module):
    """A CNN-based actor network for SAC. Based on NatureCNN https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L48."""

    action_dim: int
    activation: int
    hidden_size: int = 64
    log_std_min: float = -20
    log_std_max: float = 2

    def setup(self):
        """Initializes the actor network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.conv0 = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv1 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv2 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.dense = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.mean_out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )
        self.log_std_out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

    def __call__(self, x):
        """Applies the actor to the input."""
        x = x / 255.0
        x = jnp.transpose(x, (0, 2, 3, 1))
        actor_hidden = self.actor_conv0(x)
        actor_hidden = self.activation_func(actor_hidden)
        actor_hidden = self.actor_conv1(actor_hidden)
        actor_hidden = self.activation_func(actor_hidden)
        actor_hidden = self.actor_conv2(actor_hidden)
        actor_hidden = self.activation_func(actor_hidden)
        actor_hidden = actor_hidden.reshape((actor_hidden.shape[0], -1))  # flatten
        actor_mean = self.mean_out_layer(actor_hidden)
        actor_logstd = self.log_std_out_layer(actor_hidden)
        actor_logstd = jnp.clip(actor_logstd, self.log_std_min, self.log_std_max)

        return TanhTransformedDistribution(
            distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        )


class SACMLPCritic(nn.Module):
    """An MLP-based critic network for SAC."""

    action_dim: int
    activation: int
    hidden_size: int = 64

    def setup(self):
        """Initializes the critic network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.critic0 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic1 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_out = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, x, action):
        """Applies the critic to the input."""
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, action], -1)
        critic = self.critic0(x)
        critic = self.activation_func(critic)
        critic = self.critic1(critic)
        critic = self.activation_func(critic)
        critic = self.critic_out(critic)

        return jnp.squeeze(critic, axis=-1)

class SACCrossQCritic(nn.Module):
    """An MLP-based critic network for SAC."""

    action_dim: int
    activation: int
    hidden_size: int = 64
    batch_norm_momentum: float = 0.99

    def setup(self):
        """Initializes the critic network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.critic0 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic1 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_out = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )
        self.batch_renorm1 = BatchRenorm(momentum=self.batch_norm_momentum, use_running_average=False)
        self.batch_renorm2 = BatchRenorm(momentum=self.batch_norm_momentum)
        self.batch_renorm3 = BatchRenorm(momentum=self.batch_norm_momentum)

    def __call__(self, x, action, train=True):
        """Applies the critic to the input."""
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, action], -1)
        x = self.batch_renorm1(x)
        critic = self.critic0(x)
        critic = self.activation_func(critic)
        critic = self.batch_renorm2(critic, use_running_average=not train)
        critic = self.critic1(critic)
        critic = self.activation_func(critic)
        critic = self.batch_renorm3(critic, use_running_average=not train)
        critic = self.critic_out(critic)

        return jnp.squeeze(critic, axis=-1)


class SACCNNCritic(nn.Module):
    """A CNN-based critic network for SAC. Based on NatureCNN https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L48."""

    action_dim: int
    activation: int
    hidden_size: int = 512

    def setup(self):
        """Initializes the critic network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.conv0 = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv1 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv2 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.dense = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x, action):
        """Applies the critic to the input."""
        x = x / 255.0
        x = jnp.transpose(x(0, 2, 3, 1))
        x = jnp.concatenate([x, action], -1)
        critic = self.conv0(x)
        critic = self.activation_func(critic)
        critic = self.conv1(critic)
        critic = self.activation_func(critic)
        critic = self.conv2(critic)
        critic = self.activation_func(critic)
        critic = critic.reshape((critic.shape[0], -1))  # flatten
        critic = self.dense(critic)
        critic = self.activation_func(critic)
        critic = self.out(critic)

        return jnp.squeeze(critic, axis=-1)

class SACCrossQCNNCritic(nn.Module):
    """A CNN-based critic network for SAC. Based on NatureCNN https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L48."""

    action_dim: int
    activation: int
    hidden_size: int = 512
    batch_norm_momentum: float = 0.99

    def setup(self):
        """Initializes the critic network."""
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.conv0 = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv1 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv2 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALUE",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.dense = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x, action, train=True):
        """Applies the critic to the input."""
        x = x / 255.0
        x = jnp.transpose(x(0, 2, 3, 1))
        x = jnp.concatenate([x, action], -1)
        x = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        critic = self.conv0(x)
        critic = self.activation_func(critic)
        critic = self.conv1(critic)
        critic = self.activation_func(critic)
        critic = self.conv2(critic)
        critic = self.activation_func(critic)
        critic = critic.reshape((critic.shape[0], -1))  # flatten
        critic = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(critic)
        critic = self.dense(critic)
        critic = self.activation_func(critic)
        critic = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(critic)
        critic = self.out(critic)

        return jnp.squeeze(critic, axis=-1)


class SACVectorCritic(nn.Module):
    """A vectorized critic network for SAC."""
    critic: type[SACMLPCritic] | type[SACCNNCritic]
    action_dim: int
    activation: int
    hidden_size: int = 64
    n_critics: int = 2

    @nn.compact
    def __call__(self, x, action):
        """Applies the critic to the input."""
        vmap_critic = nn.vmap(
            self.critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )(self.action_dim, self.activation, self.hidden_size)
        return vmap_critic(x, action)


class SACCrossQVectorCritic(nn.Module):
    """A vectorized critic network for SAC."""
    critic: type[SACCrossQCritic] | type[SACCrossQCNNCritic]
    action_dim: int
    activation: int
    hidden_size: int = 64
    n_critics: int = 2

    @nn.compact
    def __call__(self, x, action, train=True):
        """Applies the critic to the input."""
        vmap_critic = nn.vmap(
            self.critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )(self.action_dim, self.activation, self.hidden_size)
        return vmap_critic(x, action, train)
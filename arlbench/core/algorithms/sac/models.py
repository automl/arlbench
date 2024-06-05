from __future__ import annotations

import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import constant, orthogonal


class TanhTransformedDistribution(distrax.Transformed):  # type: ignore[name-defined]
    """Tanh transformation of a distrax distribution."""

    def __init__(self, distribution):  # type: ignore[name-defined]
        super().__init__(
            distribution=distribution, bijector=distrax.Block(distrax.Tanh(), 1)
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class AlphaCoef(nn.Module):
    """Alpha coefficient for SAC."""

    alpha_init: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha", init_fn=lambda rng: jnp.full((), jnp.log(self.alpha_init))
        )

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.log_alpha)


class SACMLPActor(nn.Module):
    """An MLP-based actor network for PPO."""

    action_dim: int
    activation: int
    hidden_size: int = 64
    log_std_min: float = -20
    log_std_max: float = 2

    def setup(self):
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
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, action], -1)
        critic = self.critic0(x)
        critic = self.activation_func(critic)
        critic = self.critic1(critic)
        critic = self.activation_func(critic)
        critic = self.critic_out(critic)

        return jnp.squeeze(critic, axis=-1)


class SACCNNCritic(nn.Module):
    """A CNN-based critic network for SAC. Based on NatureCNN https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L48."""

    action_dim: int
    activation: int
    hidden_size: int = 512

    def setup(self):
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


class SACVectorCritic(nn.Module):
    critic: type[SACMLPCritic] | type[SACCNNCritic]
    action_dim: int
    activation: int
    hidden_size: int = 64
    n_critics: int = 2

    @nn.compact
    def __call__(self, x, action):
        vmap_critic = nn.vmap(
            self.critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )(self.action_dim, self.activation, self.hidden_size)
        return vmap_critic(x, action)

from __future__ import annotations

from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class MLPActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_size: int = 64
    discrete: bool = True

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
        self.out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

        self.actor_logtstd = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,)
        )

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

    def __call__(self, x):
        actor_mean = self.dense0(x)
        actor_mean = self.activation_func(actor_mean)
        actor_mean = self.dense1(actor_mean)
        actor_mean = self.activation_func(actor_mean)
        actor_mean = self.out_layer(actor_mean)
        if self.discrete:
            pi = distrax.Categorical(logits=actor_mean)
        else:
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.actor_logtstd))

        critic = self.critic0(x)
        critic = self.activation_func(critic)
        critic = self.critic1(critic)
        critic = self.activation_func(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class CNNActorCritic(nn.Module):
    """Based on NatureCNN https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L48."""
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_size: int = 64
    discrete: bool = True

    def setup(self):
        if self.activation == "tanh":
            self.activation_func = nn.tanh
        elif self.activation == "relu":
            self.activation_func = nn.relu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.actor_conv1 = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_conv2 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_conv3 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_dense = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_out = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )
        self.actor_logtstd = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,)
        )

        self.critic_conv1 = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_conv2 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_conv3 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_dense = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_out = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, x):
        actor_mean = self.actor_conv1(x)
        actor_mean = self.activation_func(actor_mean)
        actor_mean = self.actor_conv2(actor_mean)
        actor_mean = self.activation_func(actor_mean)
        actor_mean = self.actor_conv3(actor_mean)
        actor_mean = self.activation_func(actor_mean)
        actor_mean = self.actor_dense(actor_mean)
        actor_mean = self.activation_func(actor_mean)
        actor_mean = self.out_layer(actor_mean)
        if self.discrete:
            pi = distrax.Categorical(logits=actor_mean)
        else:
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.actor_logtstd))

        critic = self.critic_conv1(x)
        critic = self.activation_func(critic)
        critic = self.critic_conv2(critic)
        critic = self.activation_func(critic)
        critic = self.critic_conv3(critic)
        critic = self.activation_func(critic)
        critic = self.critic_dense(critic)
        critic = self.activation_func(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1)

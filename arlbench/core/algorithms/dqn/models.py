from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class CNNQ(nn.Module):
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
        
        self.conv1 = nn.Conv(
            features=32,
            kernel_size=(8, 8), 
            strides=(4, 4),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv2 = nn.Conv(
            features=64, 
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.conv3 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.dense = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.out_layer = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, x):
        q = self.conv1(x)
        q = self.activation_func(q)
        q = self.conv2(q)
        q = self.activation_func(q)
        q = self.conv3(q)
        q = self.activation_func(q)
        q = self.dense(q)
        q = self.activation_func(q)
        return self.out_layer(q)


class MLPQ(nn.Module):
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
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, x):
        q = self.dense0(x)
        q = self.activation_func(q)
        q = self.dense1(q)
        q = self.activation_func(q)
        return self.out_layer(q)


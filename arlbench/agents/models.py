import jax.numpy as jnp
from typing import Sequence
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

from enum import Enum

ACTIVATION = {
    0: nn.tanh,
    1: nn.relu
}


class SACActor(nn.Module):
    action_dim: Sequence[int]
    activation: int
    hidden_size: int = 64
    log_std_min: float = -20
    log_std_max: float = 2

    def setup(self):
        if self.activation not in ACTIVATION.keys():
            raise ValueError(f"Invalid activation_func id: {self.activation}")
        else:
            self.activation_func = ACTIVATION[self.activation]

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

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        return pi


class SACCritic(nn.Module):
    action_dim: Sequence[int]
    activation: int
    hidden_size: int = 64

    def setup(self):
        if self.activation not in ACTIVATION.keys():
            raise ValueError(f"Invalid activation_func id: {self.activation}")
        else:
            self.activation_func = ACTIVATION[self.activation]

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
        #x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, action], -1)
        critic = self.critic0(x)
        critic = self.activation_func(critic)
        critic = self.critic1(critic)
        critic = self.activation_func(critic)
        critic = self.critic_out(critic)

        return jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: int
    hidden_size: int = 64
    discrete: bool = True

    def setup(self):
        if self.activation not in ACTIVATION.keys():
            raise ValueError(f"Invalid activation_func id: {self.activation}")
        else:
            self.activation_func = ACTIVATION[self.activation]

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
        actor_mean = self.mean_out_layer(actor_mean)
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


class Q(nn.Module):
    action_dim: Sequence[int]
    activation: int         # 0 = tanh, 1 = relu
    hidden_size: int = 64
    discrete: bool = True

    def setup(self):
        if self.activation not in ACTIVATION.keys():
            raise ValueError(f"Invalid activation_func id: {self.activation}")
        else:
            self.activation_func = ACTIVATION[self.activation]

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
        q = self.out_layer(q)

        return q

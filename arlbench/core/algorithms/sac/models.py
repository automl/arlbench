from __future__ import annotations

from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


class TanhTransformedDistribution(distrax.Transformed):  # type: ignore[name-defined]
    def __init__(self, distribution):  # type: ignore[name-defined]
        super().__init__(distribution=distribution, bijector=distrax.Block(distrax.Tanh(), 1))

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class AlphaCoef(nn.Module):
    alpha_init: float = 1.0

    def setup(self):
        self.log_alpha = self.param("log_alpha", init_fn=lambda rng: jnp.full((), jnp.log(self.alpha_init)))

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.log_alpha)



class SACActor(nn.Module):
    action_dim: Sequence[int]
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
            #kernel_init=orthogonal(jnp.sqrt(2)),
            #bias_init=constant(0.0),
        )
        self.dense1 = nn.Dense(
            self.hidden_size,
            #kernel_init=orthogonal(jnp.sqrt(2)),
            #bias_init=constant(0.0),
        )
        self.mean_out_layer = nn.Dense(
            self.action_dim, #kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )
        self.log_std_out_layer = nn.Dense(
            self.action_dim, #kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )


    def __call__(self, x):
        actor_hidden = self.dense0(x)
        actor_hidden = self.activation_func(actor_hidden)
        actor_hidden = self.dense1(actor_hidden)
        actor_hidden = self.activation_func(actor_hidden)
        actor_mean = self.mean_out_layer(actor_hidden)
        actor_logstd = self.log_std_out_layer(actor_hidden)
        actor_logstd = jnp.clip(actor_logstd, self.log_std_min, self.log_std_max)

        return TanhTransformedDistribution(distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd)))


class SACCritic(nn.Module):
    action_dim: Sequence[int]
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
            #kernel_init=orthogonal(jnp.sqrt(2)),
            #bias_init=constant(0.0),
        )
        self.critic1 = nn.Dense(
            self.hidden_size,
            #kernel_init=orthogonal(jnp.sqrt(2)),
            #bias_init=constant(0.0),
        )
        self.critic_out = nn.Dense(
            1, #kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, x, action):
        #x = x.reshape((x.shape[0], -1))

        # @Julian this helped with an error for discrete action spaces
        if len(action.shape) == 1:
            action = action[:, jnp.newaxis]

        x = jnp.concatenate([x, action], -1)
        critic = self.critic0(x)
        critic = self.activation_func(critic)
        critic = self.critic1(critic)
        critic = self.activation_func(critic)
        critic = self.critic_out(critic)

        return jnp.squeeze(critic, axis=-1)


class SACVectorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: int
    hidden_size: int = 64
    n_critics: int = 2

    @nn.compact
    def __call__(self, x, action):
        vmap_critic = nn.vmap(
            SACCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )(self.action_dim, self.activation, self.hidden_size)
        return vmap_critic(x, action)

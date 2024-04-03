import jax
import gymnax
import functools
import jax.numpy as jnp


class VectorWrapper:
    def __init__(self, env, n_envs: int = -1, env_params=None):
        super().__init__()
        self.is_gymnax = isinstance(env, gymnax.environments.environment.Environment)
        self.env = env
        self.xla_step_ = None

        if self.is_gymnax:
            # TODO validate that env_params is not None
            self.env_params = env_params
        else:
            self.handle0_, _, _, self.xla_step_ = self.env.xla()

    @functools.partial(jax.jit, static_argnums=0)
    def reset_gymnax_(self, rng):
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            rng, self.env_params
        )
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=0)
    def reset_envpool_(self):
        obs, _ = self.env.reset()
        return self.handle0_, obs

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, env_params=None, rng=None):
        if self.is_gymnax:
            return self.reset_gymnax_(rng)
        else:
            return self.reset_envpool_()

    @functools.partial(jax.jit, static_argnums=0)
    def step_gymnax_(self, env_state, action, rng):
        obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(rng, env_state, action, self.env_params)

        return env_state, (obs, reward, done, info)

    @functools.partial(jax.jit, static_argnums=0)
    def step_envpool_(self, handle, action):
        handle1, (obs, reward, term, trunc, info) = self.step(handle, action)
        done = jnp.logical_or(term, trunc)
        return handle1, (obs, reward, done, info)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state, action, rng=None):
        if self.is_gymnax:
            return self.step_gymnax_(env_state, action, rng=rng)
        else:
            return self.step_envpool_(env_state, action)

    @property
    def action_space(self):
        if self.is_gymnax:
            return self.env.action_space(self.env_params)
        else:
            return self.env.action_space

    @property
    def observation_space(self):
        if self.is_gymnax:
            return self.env.observation_space(self.env_params)
        else:
            return self.env.observation_space
    
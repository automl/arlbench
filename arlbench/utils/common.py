import jax
import gymnax
import numpy as np
import jax.numpy as jnp
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import gymnasium as gym
from gymnasium.wrappers import AutoResetWrapper, FlattenObservation
import chex
from typing import Union
from gymnax.environments import EnvState, EnvParams
from minigrid.wrappers import RGBImgObsWrapper


ENVS = {
    0: ("gymnax", "CartPole-v1")
}

class ImageExtractionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["image"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["image"], info

    def step(self, action):
        obs, reward, tr, te, info = self.env.step(action)
        return obs["image"], reward, tr, te, info


def make_env(env_id: int):
    if env_id not in ENVS.keys():
        raise ValueError(f"Invalid env_id: {env_id}")
    env_framework, env_name = ENVS[env_id]

    if env_framework == "gymnax":
        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)
    elif env_framework == "brax":
        from brax import envs

        env = envs.get_environment(env_name, backend="generalized")
        env = envs.training.wrap(env)
        env = BraxToGymnaxWrapper(env)
        env_params = None
    else:
        if env_name.startswith("procgen"):
            import procgen
            import gym as old_gym

            env = old_gym.make(env_name)
            env = GymToGymnasiumWrapper(env)
        elif env_name.lower().startswith("minigrid"):
            env = gym.make(env_name)
            env = RGBImgObsWrapper(env)
            env = ImageExtractionWrapper(env)
        else:
            env = gym.make(env_name)

        # Gymnax does autoreset anyway
        env = AutoResetWrapper(env)
        env = FlattenObservation(env)
        env = GymToGymnaxWrapper(env)
        env_params = None
        
    env = LogWrapper(env)
    return env, env_params


def to_gymnasium_space(space):
    import gym as old_gym

    if isinstance(space, old_gym.spaces.Box):
        new_space = gym.spaces.Box(
            low=space.low, high=space.high, dtype=space.low.dtype
        )
    elif isinstance(space, old_gym.spaces.Discrete):
        new_space = gym.spaces.Discrete(space.n)
    else:
        raise NotImplementedError
    return new_space


class BraxToGymnaxWrapper(gymnax.environments.environment.Environment):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.max_steps_in_episode = 1000
        self.step = jax.jit(self.internal_step)
        self.reset = jax.jit(self.internal_reset)

    def internal_step(self, key, state, action, params):
        state = self.env.step(state, jnp.array([action]))
        return state.obs[0], state, state.reward[0], state.done[0], {}

    def internal_reset(self, key: chex.PRNGKey, params: EnvParams):
        state = self.env.reset(rng=jnp.array([key]))
        return state.obs[0], state

    @property
    def default_params(self):
        return None

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.action_size,)
        )

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return gymnax.environments.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.observation_size,)
        )


class GymToGymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = to_gymnasium_space(self.env.action_space)
        self.observation_space = to_gymnasium_space(self.env.observation_space)

    def reset(self, seed=None, options={}):
        return self.env.reset(), {}

    def step(self, action):
        s, r, d, i = self.env.step(action)
        return s, r, d, False, i


class JaxifyGymOutput(gym.Wrapper):
    def step(self, action):
        s, r, te, tr, _ = self.env.step(action)
        r = np.ones(s.shape) * r
        d = np.ones(s.shape) * int(te or tr)
        return np.stack([s, r, d]).astype(np.float32)


def make_bool(data):
    return np.array([bool(data)])


class GymToGymnaxWrapper(gymnax.environments.environment.Environment):
    def __init__(self, env):
        super().__init__()
        self.done = False
        self.env = JaxifyGymOutput(env)
        self.state = None
        self.state_type = None

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ):
        """Environment-specific step transition."""
        # TODO: this is obviously super wasteful for large state spaces - how can we fix it?
        result_shape = jax.core.ShapedArray(
            np.repeat(self.state[None, ...], 3, axis=0).shape, jnp.float32
        )
        result = jax.pure_callback(self.env.step, result_shape, action)
        s = result[0].astype(self.state_type)
        r = result[1].mean()
        d = result[2].mean()
        result_shape = jax.core.ShapedArray((1,), bool)
        self.done = jax.pure_callback(make_bool, result_shape, d)[0]
        return s, {}, r, self.done, {}

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        """Environment-specific reset."""
        self.done = False
        self.state, _ = self.env.reset()
        self.state_type = self.state.dtype
        return self.state, {}

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        return self.done

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        if isinstance(self.env.action_space, gym.spaces.Box):
            return len(self.env.action_space.low)
        else:
            return self.env.action_space.n

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        if isinstance(self.env.action_space, gym.spaces.Box):
            return gymnax.environments.spaces.Box(
                self.env.action_space.low,
                self.env.action_space.high,
                self.env.action_space.low.shape,
            )
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            return gymnax.environments.spaces.Discrete(self.env.action_space.n)
        else:
            raise NotImplementedError(
                "Only Box and Discrete action spaces are supported."
            )

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return gymnax.environments.spaces.Box(
            self.env.observation_space.low,
            self.env.observation_space.high,
            self.env.observation_space.low.shape,
        )

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return gymnax.environments.spaces.Dict({})

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(500)
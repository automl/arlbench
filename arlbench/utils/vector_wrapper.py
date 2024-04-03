import chex
import jax
import gymnax
from typing import Union, Any
from flax import struct

# handle0, states = loop_var
#   action = policy(states)
#   # for gym < 0.26
#   handle1, (new_states, rew, done, info) = step(handle0, action)
#   # for gym >= 0.26
#   # handle1, (new_states, rew, term, trunc, info) = step(handle0, action)
#   # for dm
#   # handle1, new_states = step(handle0, action)
#   return (handle1, new_states)


@struct.dataclass
class EnvState:
    handle: Any     # TODO add type of "handle" from env.xla()
    step: Any       # TODO add type of "step" from env.xla()


@struct.dataclass
class EnvParams:
    dummy: int


class VectorWrapper(gymnax.environments.environment.Environment):
    def __init__(self, env, n_envs: int):
        super().__init__()
        self.is_gymnax = isinstance(env, gymnax.environments.environment.Environment)
        self.env = env

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ):
        """Environment-specific step transition."""
        new_handle, (obs, r, done, info) = state.step(state.handle, action)
        new_state = EnvState(handle=new_handle, step=state.step)

        return obs,new_state, r, done, info

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        """Environment-specific reset."""
        handle, _, _, step = self.env.xla()
        state = EnvState(handle=handle, step=step)
        return state, {}

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return self.env.action_space

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return gymnax.environments.spaces.Box(
            self.env.observation_space.low,
            self.env.observation_space.high,
            self.env.observation_space.low.shape,
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(500)

import chex

@chex.dataclass(frozen=True)
class TimeStep:
    last_obs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
from __future__ import annotations

import chex


@chex.dataclass(frozen=True)
class TimeStep:
    """A timestep capturing an environment interaction."""
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

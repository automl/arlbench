import chex
from typing import Union
from flax.training.train_state import TrainState

@chex.dataclass(frozen=True)
class TimeStep:
    last_obs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class ExtendedTrainState(TrainState):
    target_params: Union[None, chex.Array, dict] = None
    opt_state = None

    @classmethod
    def create_with_opt_state(cls, *, apply_fn, params, target_params, tx, opt_state, **kwargs):
        if opt_state is None:
            opt_state = tx.init(params)
        obj = cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
        return obj
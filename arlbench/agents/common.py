import chex
from typing import Union
from flax.training.train_state import TrainState


class ExtendedTrainState(TrainState):
    target_params: Union[None, chex.Array, dict] = None
    opt_state = None

    @classmethod
    def create_with_opt_state(cls, *, apply_fn, params, tx, opt_state, **kwargs):
        if opt_state is None:
            opt_state = tx.init(params)
        obj = cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
        return obj
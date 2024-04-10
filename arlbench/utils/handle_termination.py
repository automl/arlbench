"""Main module."""
from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from ..autorl import AutoRLEnv

logger = logging.getLogger(__name__)


class HandleTermination:
    """This is a context manager that handles different termination signals.
    The comments show how to save a model and optimizer states
    even if there's an error in the code.
    """

    def __init__(self, env: AutoRLEnv):
        """Initialize the context manager with logdir."""
        self.env = env

    def __enter__(self):
        self.old_sigterm_handler = signal.signal(signal.SIGTERM, self.handle_sigterm)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """An exception was raised, so save the model and optimizer states
        with an exception tag before re-raising.
        """
        signal.signal(signal.SIGTERM, self.old_sigterm_handler)
        if exc_type is not None:
            logger.info("Oh no, there was an exception!")
            path = self.env.save(tag="exc")
            logger.info(f"Saving checkpoint to {path}")

            # torch.save({
            #    'model_state_dict': self.model.state_dict(),
            #    'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self.directory / f'checkpoint_exc.pth')

        # Everything ran perfectly, so save the final model and optimizer states
        if exc_type is None:
            logger.info("All clear!")

        return False

    def handle_sigterm(self, signum, frame): # noqa: ARG002
        """Save the model and optimizer states before exiting."""
        path = self.env.save(tag="sigterm")
        logger.info(f"Saving checkpoint to {path}")
        sys.exit(0)
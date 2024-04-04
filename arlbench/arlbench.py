from arlbench.utils import HandleTermination
import os


def cool_things(cfg):
    """As the name suggests, this does cool things."""
    with HandleTermination(os.getcwd()):
        print(f"Your current config is: {cfg}")
        


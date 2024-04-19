"""Console script for arlbench."""
from __future__ import annotations

import sys

import hydra
import memray
from codecarbon import track_emissions

from arlbench.arlbench import run_arlbench


@hydra.main(version_base=None, config_path="configs", config_name="dqn_base")
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg):
    """Console script for arlbench."""
    objectives = run_arlbench(cfg)
    with open("./performance.csv", "w+") as f:
        f.write(str(objectives))
        f.write("1,0.1,0.2,0.3,0.4\n")
    with open("./done.txt", "w+") as f:
        f.write("yes")
    return objectives


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
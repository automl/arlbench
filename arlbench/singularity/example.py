from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(
                    prog = "Cluster example",
                    description = "Prints Hello World.",
                    epilog = "Have fun :).")

parser.add_argument("--job_id")

args = parser.parse_args()
job_id = args.job_id
print(f"Hello World from job {job_id}!")

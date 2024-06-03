import os
import pandas as pd
import re
from datetime import datetime


RESULTS_DIR = "results"
RESULTS_COMBINED_DIR = "results_combined"


def extract_training_runtime(arlbench_log_path: str) -> float:
    with open(arlbench_log_path, "r") as f:
        lines = f.readlines()

    start_time = None
    end_time = None

    for line in lines:
        if "Training started" in line:
            start_match = re.search(r'\[(.*?)\]', line)
            if start_match:
                start_time = datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
        elif "Training finished" in line:
            end_match = re.search(r'\[(.*?)\]', line)
            if end_match:
                end_time = datetime.strptime(end_match.group(1), '%Y-%m-%d %H:%M:%S,%f')

        if start_time and end_time:
            runtime = (end_time - start_time).total_seconds()
            return runtime

    return None


def read_arlbench_runtimes(approach: str) -> pd.DataFrame:
    runtime_data = []

    approach_path = os.path.join(RESULTS_DIR, approach)
    for exp in os.listdir(approach_path):
        print(f"Reading experiment {exp}")
        exp_path = os.path.join(approach_path, exp)
        if not os.path.isdir(exp_path):
            continue

        for approach_seed in os.listdir(exp_path):
            approach_seed_path = os.path.join(exp_path, approach_seed)
            if not os.path.isdir(approach_seed_path):
                continue

            for run_seed in os.listdir(approach_seed_path):
                run_seed_path = os.path.join(approach_seed_path, run_seed)
                for run_id in os.listdir(run_seed_path):
                    run_path = os.path.join(run_seed_path, run_id)
                    log_file_path = os.path.join(run_path, "run_arlbench.log")

                    if not os.path.isfile(log_file_path):
                        continue

                    runtime = extract_training_runtime(log_file_path)

                    splitted_filename = exp.split("_")
                    algorithm = splitted_filename[0]
                    environment = "_".join(splitted_filename[1:])
                    
                    runtime_data += [{
                        "algorithm": algorithm.upper(),
                        "environment": environment,
                        "run_id": run_id,
                        "run_seed": run_seed,
                        "approach_seed": approach_seed,
                        "runtime": runtime
                    }]

    return pd.DataFrame(runtime_data)


def fetch_cleanrl_runtimes(algorithms, environments):
    import wandb
    api = wandb.Api()

    data = []

    runs = api.runs(path="openrlbenchmark/cleanrl")
    for run in runs:
        algorithm = run.config["exp_name"]
        if not ("ppo" in algorithm or "dqn" in algorithm or "sac" in algorithm):
            continue

        environment = run.config["env_id"]
        summary = run.summary

        if not "_runtime" in summary:
            continue

        data += [{
            "algorithm": algorithm,
            "environment": environment,
            "seed": run.config["seed"],
            "runtime": summary["_runtime"]
        }]

    return pd.DataFrame(data)


if __name__ == "__main__":
    #approach = "rs"
    #data = read_arlbench_runtimes(approach)
    #data.to_csv(os.path.join(RESULTS_COMBINED_DIR, approach, "runtimes.csv"))
    # data = pd.read_csv(os.path.join(RESULTS_COMBINED_DIR, approach, "runtimes.csv"))
    data = fetch_cleanrl_runtimes([], ["CartPole-v1"])
    print(data)
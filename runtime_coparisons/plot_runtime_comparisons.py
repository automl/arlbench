import os
import pandas as pd
import re
from datetime import datetime
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt


sns.set_style("whitegrid")
sns.set_palette("colorblind")


RESULTS_DIR = "results"
RESULTS_COMBINED_DIR = "results_combined"

RUNTIMES_DIR = "results_combined/runtime_comparisons"

GPU_ENVS = ["ant", "halfcheetah", "hopper", "humanoid", "BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "QBert-v5", "Phoenix-v5"]
ATARI_ENVS = ["BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "QBert-v5", "Phoenix-v5"]

N_SOBOL_RUNS = 256 * 10         # 256 * 10 seeds
N_SOBOL_RUNS_ATARI = 128 * 10   # 128 * 10 seeds

N_OPT_RUNS = 4 * 3 * 32 * 4     # 4 optimizers * 3 optimizer seeds * 32 * 3 RL seeds

ENV_CATEGORIES = {
    "ppo": [
        "Atari", "Atari", "Atari", "Atari", "Atari", "Box2D", "Box2D", "MuJoCo", "MuJoCo", "MuJoCo", "MuJoCo",
        "Classic Control", "Classic Control", "Classic Control", "Classic Control", "Classic Control",
        "XLand", "XLand", "XLand", "XLand"
        ],
    "dqn": [
        "Atari", "Atari", "Atari", "Atari", "Atari", "Box2D", "Classic Control", "Classic Control", "Classic Control", 
        "XLand", "XLand", "XLand", "XLand"
    ],
    "sac": [
        "Box2D", "Box2D", "MuJoCo", "MuJoCo", "MuJoCo", "MuJoCo",
        "Classic Control", "Classic Control"
        ],
}

SUBSET_CATEGORIES = {
    "ppo": ["Box2D", "MuJoCo", "Atari", "XLand", "XLand"],
    "dqn": ["Classic Control", "XLand", "Atari", "XLand"],
    "sac": ["Box2D", "MuJoCo", "Classic Control", "Classic Control"],
}

CATEGORY = {
    "ant": "brax",
    "Ant-v4": "MuJoCo",
    "CartPole-v1": "Classic Control",
    "LunarLander-v2": "Box2D",
    "LunarLanderContinuous-v2": "Box2D",
    "Pendulum-v1": "Classic Control",
    "Pong-v5": "Atari",
    "MiniGrid-DoorKey-5x5": "XLand"
}


def extract_training_runtime(arlbench_log_path: str) -> float:
    with open(arlbench_log_path, "r") as f:
        lines = f.readlines()

    start_time = None
    end_time = None

    for line in lines:
        if "Training started" in line:
            start_match = re.search(r"\[(.*?)\]", line)
            if start_match:
                start_time = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
        elif "Training finished" in line:
            end_match = re.search(r"\[(.*?)\]", line)
            if end_match:
                end_time = datetime.strptime(end_match.group(1), "%Y-%m-%d %H:%M:%S,%f")

        if start_time and end_time:
            runtime = (end_time - start_time).total_seconds()
            return runtime

    return None


def read_arlbench_runtimes(approach: str) -> pd.DataFrame:
    runtime_data_path = os.path.join(RESULTS_COMBINED_DIR, approach, "runtimes.csv")
    if os.path.isfile(runtime_data_path):
        return pd.read_csv(runtime_data_path)

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
                if not os.path.isdir(run_seed_path):
                    continue

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

    runtime_data = pd.DataFrame(runtime_data)
    runtime_data.to_csv(runtime_data_path, index=False)
    return runtime_data


def plot_runtime_comparisons():
    runtime_results = []

    for exp in os.listdir(RUNTIMES_DIR):
        exp_path = os.path.join(RUNTIMES_DIR, exp)
        runtime_data = pd.read_csv(exp_path)

        exp_name = exp.replace(".csv", "")
        framework, environment, algorithm = exp_name.split("_")

        for algorithm_framework, data in runtime_data.groupby("framework"):
            runtime = data["runtime"].mean()
            runtime_results += [{
                "algorithm_framework": algorithm_framework,
                "environment": environment,
                "algorithm": algorithm,
                "category": CATEGORY[environment],
                "runtime": runtime
            }]

    runtime_data = pd.DataFrame(runtime_results)
    
    print(runtime_data)

    result = {}
    for index, row in runtime_data.iterrows():
        algorithm = row["algorithm"]
        category = row["category"]
        runtime = row["runtime"]
        
        if algorithm not in result:
            result[algorithm] = {}
        
        if category not in result[algorithm]:
            result[algorithm][category] = {"ARLBench": None, "SB3": None}
    
        result[algorithm][category][row["algorithm_framework"]] = runtime

    all_runtimes = []
    for set_name, env_categories in zip(["All", "Subset"], [ENV_CATEGORIES, SUBSET_CATEGORIES]):
        for algorithm, categories in env_categories.items():        
            for category in categories:
                total_runtime_arlb = 0
                total_runtime_sb3 = 0

                if category in result.get(algorithm, {}):
                    if category == "mujoco":
                        total_runtime_arlb += result[algorithm]["brax"]["ARLBench"]
                    else:
                        total_runtime_arlb += result[algorithm][category]["ARLBench"]
                    total_runtime_sb3 += result[algorithm][category]["SB3"]
            
                all_runtimes.append({"algorithm": algorithm, "category": category, "set": f"{set_name} (ARLBench)", "runtime": total_runtime_arlb })
                all_runtimes.append({"algorithm": algorithm, "category": category, "set": f"{set_name} (SB3)", "runtime": total_runtime_sb3 })

    all_runtimes = pd.DataFrame(all_runtimes)
    all_runtimes = all_runtimes.groupby(["algorithm", "set", "category"]).sum().reset_index()

    # Fill missing values with zeros
    all_runtimes["runtime"] = all_runtimes["runtime"].fillna(0)

    # Set seaborn style
    sns.set_style("whitegrid")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    fig.subplots_adjust(top=0.85)

    set_order = ["All (SB3)", "All (ARLBench)", "Subset (SB3)", "Subset (ARLBench)"]
    hue_order = ["Atari", "Box2D", "Classic Control", "XLand", "MuJoCo"]

    all_combinations = pd.MultiIndex.from_product([all_runtimes["algorithm"].unique(), all_runtimes["set"].unique(), hue_order], names=["algorithm", "set", "category"])
    all_runtimes = all_runtimes.set_index(["algorithm", "set", "category"]).reindex(all_combinations).reset_index()
    all_runtimes["runtime"] = all_runtimes["runtime"].fillna(0)

    print(all_runtimes)

    all_runtimes["runtime"] /= 3600
    
    for i, algorithm in enumerate(env_categories.keys()):
        runtime_data = all_runtimes[all_runtimes["algorithm"] == algorithm]
        print(f"### {algorithm.upper()} ###")
        print(runtime_data.groupby(["algorithm", "set"]).sum())

        plot = (
            so.Plot(
                runtime_data,
                x="runtime",
                y="set",
                color="category",
            ).add(
                so.Bar(),
                so.Agg("sum"),
                so.Norm(func="sum", by=["x"]),
                so.Stack()
            ).scale(
                color="colorblind",
                y=so.Nominal(order=set_order),
            )
        )
        plot.on(axes[i]).show()

        axes[i].set_title(algorithm.upper())
        axes[i].set_xlabel("Total Runtime [h]")
        if i == 0:
            axes[i].set_ylabel("Environments")
        else:
            axes[i].set_ylabel("")
            axes[i].set_yticklabels([])

    for l in fig.legends:
        l.set_visible(False)
    legend = fig.legends[0]
    fig.legend(legend.legend_handles, [t.get_text() for t in legend.texts], loc="upper center", bbox_to_anchor=(0.5, 0.105), ncol=5, fancybox=False, shadow=False, frameon=False)
    plt.tight_layout(pad=1.3)
    plt.savefig("plots/runtime_experiments/set_comparison.png", dpi=500)
    plt.close()
    

def compute_total_runtime():
    # We use the random search data for the runtimes
    runtimes = read_arlbench_runtimes("rs")

    runtimes["platform"] = "cpu"
    runtimes.loc[runtimes["environment"].isin(GPU_ENVS), "platform"] = "gpu"

    runtimes = runtimes.groupby(['algorithm', 'environment', 'platform'])['runtime'].mean().reset_index()

    runtimes["n_runs"] = N_SOBOL_RUNS + N_OPT_RUNS
    runtimes.loc[runtimes["environment"].isin(ATARI_ENVS), "n_runs"] = N_SOBOL_RUNS_ATARI + N_OPT_RUNS

    runtimes["total_runtime"] = runtimes["n_runs"] * runtimes["runtime"]
    runtimes["total_runtime"] /= 3600   # to hours

    print(runtimes)
    total_runtimes = runtimes.groupby(['platform'])['runtime'].sum().reset_index()
    print(total_runtimes)



if __name__ == "__main__":
    plot_runtime_comparisons()
    # compute_total_runtime()
    

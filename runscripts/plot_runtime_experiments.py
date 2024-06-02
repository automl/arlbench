import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


def capitalize(s: str) -> str:
    return s[0].upper() + s[1:]


def format_framework(s: str) -> str:
    return "ARLBench" if s == "arlbench" else "SB3"


def plot(experiments: list[dict], dir: str):
    for experiment in experiments:
        for env_name in experiment["env_names"]:
            env_framework = experiment["env_framework"]
            algorithm_frameworks = experiment["algorithm_frameworks"]
            algorithm_name = experiment["algorithm_name"]

            os.makedirs(f"plots/{dir}/{env_framework}_{env_name}", exist_ok=True)

            plot_experiment(env_framework, env_name, algorithm_frameworks, algorithm_name, dir)


def plot_experiment(env_framework: str, env_name: str, algorithm_frameworks: list[str], algorithm_name: str, dir: str):
    all_data = pd.DataFrame()
    runtimes = []

    for algorithm_framework in algorithm_frameworks:
        folder_path = f"results/{dir}/{env_framework}_{env_name}/{algorithm_framework}_{algorithm_name}"
        folders = os.listdir(folder_path)

        for folder in folders:
            eval_path = os.path.join(folder_path, folder, "evaluation.csv")
            if os.path.exists(eval_path):
                data = pd.read_csv(eval_path)
                data["id"] = folder
                data["framework"] = format_framework(algorithm_framework)
                all_data = pd.concat([all_data, data])

            info_path = os.path.join(folder_path, folder, "info")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    info_data = f.read()

                    time_pattern = r"time:\s*(\d+\.\d+)"
                    matches = re.search(time_pattern, info_data)
                    if matches:
                        time_value = float(matches.group(1))
                        runtimes += [{
                            "framework": format_framework(algorithm_framework),
                            "id": folder,
                            "runtime": time_value
                        }]

    runtimes = pd.DataFrame(runtimes)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 2.5), gridspec_kw={"width_ratios": [2, 1]})

    sns.lineplot(ax=axes[0], x="steps", y="returns", hue="framework", data=all_data, errorbar=("ci", 95), estimator="mean")
    axes[0].set_xlabel("Steps")
    #axes[0].set_xscale("log")
    axes[0].set_ylabel("Evaluation Return")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=False, shadow=False, frameon=False)


    sns.boxplot(ax=axes[1], x="framework", y="runtime", data=runtimes)
    axes[1].set_xlabel("Framework")
    axes[1].set_ylabel("Runtime [s]")
    axes[1].set_ylim(0, None)
    plt.suptitle(f"{algorithm_name.upper()} on {capitalize(env_name)} ({capitalize(env_framework)})", y=0.95)
    fig.subplots_adjust(bottom=0.65)
    plt.tight_layout()
    plt.savefig(f"plots/{dir}/{env_framework}_{env_name}/{algorithm_name}.png", dpi=500)
    plt.close()


if __name__ == "__main__":
    dir = "runtime_experiments"

    experiments_gpu = [
        {   
            "env_framework": "envpool",
            "env_names": ["CartPole-v1", "LunarLander-v2", "Pong-v5"],
            "algorithm_frameworks": ["arlbench",  "sb3"],
            "algorithm_name": "dqn"
        },
        {   
            "env_framework": "envpool",
            "env_names": ["CartPole-v1", "Pong-v5", "LunarLander-v2", "LunarLanderContinuous-v2", "Pendulum-v1", "Ant-v4"],
            "algorithm_frameworks": ["arlbench",  "sb3"],
            "algorithm_name": "ppo"
        },
        {
            "env_framework": "envpool",
            "env_names": ["Pendulum-v1", "LunarLanderContinuous-v2", "Ant-v4"],
            "algorithm_frameworks": ["arlbench",  "sb3"],
            "algorithm_name": "sac"
        },
        {
            "env_framework": "brax",
            "env_names": ["ant"],
            "algorithm_frameworks": ["arlbench",  "brax"],
            "algorithm_name": "ppo"
        },
        {
            "env_framework": "brax",
            "env_names": ["ant"],
            "algorithm_frameworks": ["arlbench",  "brax"],
            "algorithm_name": "sac"
        },
    ]

    #dir_cpu = "runtime_experiments/normal"

    #experiments_cpu = [
    #    {
    #        "env_framework": "gymnax",
    #        "env_names": ["CartPole-v1"],
    #        "algorithm_frameworks": ["purejaxrl",  "arlbench"],
    #        "algorithm_name": "dqn"
    #    },
    #    {
    #        "env_framework": "gymnax",
    #        "env_names": ["CartPole-v1"],
    #        "algorithm_frameworks": ["purejaxrl",  "arlbench"],
    #        "algorithm_name": "ppo"
    #    },
    #]

    plot(experiments_gpu, dir)

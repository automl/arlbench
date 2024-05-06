import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


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
                data['id'] = folder 
                data['framework'] = algorithm_framework
                all_data = pd.concat([all_data, data])

            info_path = os.path.join(folder_path, folder, "info")
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info_data = f.read()

                    time_pattern = r'time:\s*(\d+\.\d+)'
                    matches = re.search(time_pattern, info_data)
                    if matches:
                        time_value = float(matches.group(1))
                        runtimes += [{
                            "framework": algorithm_framework,
                            "id": folder,
                            "runtime": time_value
                        }]

    runtimes = pd.DataFrame(runtimes)

    plt.figure()
    sns.lineplot(x='steps', y='returns', hue='framework', data=all_data, errorbar='sd')
    plt.xlabel('Steps')
    plt.xscale('log')
    plt.ylabel('Return')
    plt.title(f'{env_framework} {env_name} {algorithm_name.upper()}')
    plt.legend(title='Framework', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{dir}/{env_framework}_{env_name}/{algorithm_name}.png')

    plt.figure()
    sns.boxplot(x='framework', y='runtime', data=runtimes)
    plt.xlabel('Framework')
    plt.ylabel('Runtime')
    plt.title(f'{env_framework} {env_name} {algorithm_name.upper()}')
    plt.tight_layout()
    plt.savefig(f'plots/{dir}/{env_framework}_{env_name}/{algorithm_name}_runtime.png')


if __name__ == "__main__":
    dir_gpu = "runtime_experiments/gpu"

    experiments_gpu = [
        {   
            "env_framework": "gymnax",
            "env_names": ["CartPole-v1"],
            "algorithm_frameworks": ["purejaxrl",  "arlbench-prio-buffer", "arlbench-no-prio-buffer"],
            "algorithm_name": "dqn"
        },
        {   
            "env_framework": "gymnax",
            "env_names": ["CartPole-v1"],
            "algorithm_frameworks": ["purejaxrl",  "arlbench"],
            "algorithm_name": "ppo"
        },
    ]

    dir_cpu = "runtime_experiments/normal"

    experiments_cpu = [
        {   
            "env_framework": "gymnax",
            "env_names": ["CartPole-v1"],
            "algorithm_frameworks": ["purejaxrl",  "arlbench"],
            "algorithm_name": "dqn"
        },
        {   
            "env_framework": "gymnax",
            "env_names": ["CartPole-v1"],
            "algorithm_frameworks": ["purejaxrl",  "arlbench"],
            "algorithm_name": "ppo"
        },
    ]

    plot(experiments_cpu, dir_cpu)

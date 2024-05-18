import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


RAW_SOBOL_RESULTS = "runscripts/configs/initial_design"
DATA_DIR = "subset_selection/data/"


def inspect_buffer_sizes():
    # Maintain one DataFrame of results per algorithm
    merged_results = {}

    for result_file in os.listdir(RAW_SOBOL_RESULTS):
        result_path = os.path.join(RAW_SOBOL_RESULTS, result_file)
        if os.path.isdir(result_path) or "ppo" in result_file:
            continue

        # Extract properties from the file name of the result
        splitted_filename = result_file.replace(".csv", "").split("_")
        algorithm = splitted_filename[-1]
        environment = "_".join(splitted_filename[:-1])

        result_raw = pd.read_csv(result_path)

        # We don't need the particular hyperparameter configuration
        try:
            result_filtered = result_raw[["run_id", "hp_config.buffer_size"]]
        except:
            continue
        if algorithm in merged_results:
            merged_results[algorithm][environment] = result_filtered["hp_config.buffer_size"]
        else:
            merged_results[algorithm] = result_filtered.rename(columns={"hp_config.buffer_size": environment})

    for algorithm, df in merged_results.items():
        # For atari, we only have 128 configurations
        df = df.iloc[:128, :]
        df = df.drop("run_id", axis=1)
        correlation_matrix = df.corr(method='spearman')

        plt.figure(figsize=(10, 8))

        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.title(f"Spearman Correlation Matrix of Buffer Sizes for {algorithm}")
        plt.tight_layout()
        plt.savefig(f"subset_selection/plots/buffer_size_corr_{algorithm}.png", dpi=500)
        plt.show()


if __name__ == "__main__":
    inspect_buffer_sizes()

import pandas as pd
import os
from offline_evaluations import OfflineEvaluations


RAW_SOBOL_RESULTS = "runscripts/configs/initial_design"
DATA_DIR = "subset_selection/data/"


def prepare_dataset():
    # Maintain one DataFrame of results per algorithm
    merged_results = {}

    for result_file in os.listdir(RAW_SOBOL_RESULTS):
        result_path = os.path.join(RAW_SOBOL_RESULTS, result_file)
        if os.path.isdir(result_path):
            continue

        result_raw = pd.read_csv(result_path)

        # We don't need the particular hyperparameter configuration
        result_filtered = result_raw[["run_id", "performance", "seed"]]

        # Match desired format
        result_filtered = result_filtered.rename(
            columns={"run_id": "Configuration", "performance": "Score", "seed": "Seed"}
        )

        # Modify score s.t. higher is better
        # During the data collection we tried to minimize the cost, i.e. the core
        result_filtered["Score"] *= -1

        # Extract properties from the file name of the result
        splitted_filename = result_file.replace(".csv", "").split("_")
        algorithm = splitted_filename[-1]
        environment = "_".join(splitted_filename[:-1])

        # Insert the environment name as an extra column
        result_filtered["Environment"] = environment

        # Sort for consistency
        result_filtered = result_filtered.sort_values(["Configuration", "Seed"])
        
        # TODO should we only use the first 128 observations since we only have 128 for Atari?
        result_filtered = result_filtered[result_filtered["Configuration"] < 128]

        # Prepare DataFrame to be merged
        merged_results[algorithm] = pd.concat(
            [
                merged_results[algorithm] if algorithm in merged_results else None,
                result_filtered,
            ],
            axis=0,
        )

    # Save everything to the data directory
    for algorithm in merged_results:
        merged_result_dir = os.path.join(DATA_DIR, algorithm)
        os.makedirs(merged_result_dir, exist_ok=True)

        # This is not required for the subset generation but we want to keep it
        merged_result_path = os.path.join(merged_result_dir, "performances.csv")
        merged_results[algorithm].to_csv(merged_result_path, index=False)

        # We use the average ranked scores over seeds as target 
        target = merged_results[algorithm].groupby(["Configuration", "Environment"])["Score"].mean().reset_index()
        
        # Sine reward ranges might vary drastically across environments we normalize by ranking
        # Note: NaNs will be propagated trough the mean. That means if a configuration returns NaN
        # for one seed (e.g. due to vanishing/exploding gradients) the configuration receives the
        # score NaN for all seeds. This ensures that these configurations still receive the lowest rank
        # since they are unstable
        pivot_target = target.pivot(index="Configuration", columns="Environment", values="Score")
        target = pivot_target.rank(axis=0, pct=True)

        # If the score is NaN we set it to the lowest rank
        target = target.fillna(0)

        # The target is the average rank of a configuration across all environments
        target = target.mean(axis=1).values
        
        # target = merged_results[algorithm].loc[:, ["Configuration", "Score"]].groupby("Configuration").mean()["Score"].to_dict()

        # This is object will be used for the subset generation
        evaluations = OfflineEvaluations.from_dataframe(
            df=merged_results[algorithm],
            model_col="Configuration",
            dataset_col="Environment",
            seed_col="Seed",
            iteration_col=None,
            target=target,
            metric_col="Score",
        )
        evaluations.save(merged_result_dir)


if __name__ == "__main__":
    prepare_dataset()

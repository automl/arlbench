import pandas as pd
import os
import numpy as np
import pandas as pd
from typing import Callable, List
from sklearn.model_selection import train_test_split, KFold
import itertools
from parallel_for import parallel_for
import scipy
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


RAW_SOBOL_RESULTS = "results_combined/sobol"


def read_arlb_dataset():
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

    return merged_results


def normalize_ranks(pivot_table: pd.DataFrame) -> pd.DataFrame:        
    pivot_table = pivot_table.rank(axis=0, pct=True)

    # If the score is NaN we set it to the lowest rank
    pivot_table = pivot_table.fillna(0)

    return pivot_table

def normalize_z_score(pivot_table: pd.DataFrame) -> pd.DataFrame:  
    def z_score(col):
        # We assign the highest score, i.e., the lowest performance to NaN
        col = col.fillna(col.max())
        return (col - col.mean()) / col.std()

    return pivot_table.apply(z_score)      


def normalize_min_max(pivot_table: pd.DataFrame) -> pd.DataFrame:        
    def z_score(col):
        # We assign the highest score, i.e., the lowest performance to NaN
        col = col.fillna(col.max())
        return (col - col.min()) / (col.max() - col.min())

    return pivot_table.apply(z_score)    


def prepare_dataset(
        dataset: dict,
        normalization_func: Callable[[pd.DataFrame], pd.DataFrame] = normalize_ranks
    ) -> dict:
    merged = {}

    # Save everything to the data directory
    for algorithm, data_frame in dataset.items():
        # First, we average across seeds. This does not depend on the normalization strategy
        # Note: NaNs will be propagated trough the mean. That means if a configuration returns NaN
        # for one seed (e.g. due to vanishing/exploding gradients) the configuration receives the
        # score NaN for all seeds. This ensures that these configurations still receive the lowest rank
        # since they are unstable
        data_frame = data_frame.groupby(["Configuration", "Environment"])["Score"].mean().reset_index()

        pivot_table = data_frame.pivot(index="Configuration", columns="Environment", values="Score")
        pivot_normalized = normalization_func(pivot_table)

        X = pivot_normalized.to_numpy()
        y = pivot_normalized.mean(axis=1).to_numpy()
        environments = pivot_normalized.columns

        merged[algorithm] = (X, y, environments)

    return merged


def generate_subsets(n: int, k: int):
    for subset in itertools.combinations(range(n), k):
        yield subset


def spearman_corr(x, y):
    spcorr, p_value = scipy.stats.spearmanr(x, y)
    return spcorr


def error(target, pred):
    # define the error used to compare linear model, here we pick the model with the highest correlation with the target
    # return 1 - spearman_corr(pred, target)
    return np.mean((pred - target) ** 2)


def fit_and_compute_error(
        task_subset: list[int], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray, 
        positive_weights: bool, 
        fit_intercept: bool, 
        error_func: Callable
    ) -> tuple[float, float]:
    X_train = X_train[:, np.array(task_subset)]
    X_val = X_val[:, np.array(task_subset)]

    lin_model = LinearRegression(positive=positive_weights, fit_intercept=fit_intercept)
    lin_model.fit(X_train, y_train)

    y_train_pred = lin_model.predict(X_train)
    y_val_pred = lin_model.predict(X_val)

    return error_func(y_train, y_train_pred), error_func(y_val, y_val_pred)


def cross_validate_scores(task_subset, X, y, positive_weights, fit_intercept, error_func: Callable, k: int = 5) -> tuple[float, float]:
    kf = KFold(n_splits=k, shuffle=True)
    train_scores = []
    val_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_score, val_score = fit_and_compute_error(task_subset, X_train, y_train, X_val, y_val, positive_weights, fit_intercept, error_func)
        train_scores.append(train_score)
        val_scores.append(val_score)

    return float(np.mean(train_scores)), float(np.mean(val_scores))


def evaluate_subsets(
    X: np.ndarray, y: np.ndarray, n_subset: int, engine: str, positive_weights: bool, fit_intercept: bool, error_func: Callable 
) -> tuple:
    subsets = [[x] for x in generate_subsets(X.shape[1], n_subset)]
    print(
        f"Fit linear models for all {len(subsets)} possible task subsets with size={n_subset}."
    )
    scores = parallel_for(
        cross_validate_scores,
        inputs=subsets,
        context={"X": X, "y": y, "positive_weights": positive_weights, "fit_intercept": fit_intercept, "error_func": error_func},
        engine=engine,
    )
    train_scores, val_scores = zip(*scores)

    return subsets, train_scores, val_scores


def main():
    dataset = read_arlb_dataset()
    training_data = prepare_dataset(dataset)

    for algorithm, (X, y, envs) in training_data.items():
        task_correlation = [
            spearman_corr(X[:, task], y) for task in range(X.shape[1])
        ]

        n_print = 5
        print("\nMost correlated tasks with target:")
        for i, task_index in enumerate(reversed(np.argsort(task_correlation)[-n_print:])):
            env_name = envs[task_index]
            print(
                f"Task {i}: {env_name} ({task_correlation[task_index]:.2f})"
            )
        print("Least correlated tasks with target:")
        for i, task_index in enumerate(np.argsort(task_correlation)[:n_print]):
            env_name = envs[task_index]
            print(
                f"Task {i}: {env_name} ({task_correlation[task_index]:.2f})"
            )

        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

        subsets, train_scores, val_scores = evaluate_subsets(X_train, y_train, 4, "ray", True, False, error_func=error)

        best_subset = subsets[np.argmin(val_scores)][0]

        print(f"Tasks chosen for {algorithm}:")
        for i, task_index in enumerate(best_subset):
            env_name = envs[task_index]
            print(
                f"Task {i}: {env_name}"
            )

        data = pd.DataFrame({"subset_id": np.arange(len(subsets)), "train_scores": train_scores, "val_scores": val_scores})

        plt.figure()
        ax = sns.scatterplot(data=data, x="subset_id", y="train_scores")
        sns.scatterplot(data=data, x="subset_id", y="val_scores", ax=ax)
        plt.show()
    
if __name__ == "__main__":
    main()

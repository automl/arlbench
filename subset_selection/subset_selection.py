import pandas as pd
import os
import numpy as np
import pandas as pd
from typing import Callable, Any
from sklearn.model_selection import KFold
import itertools
from parallel_for import parallel_for
import scipy
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_palette("colorblind")

RAW_SOBOL_RESULTS = "results_combined/sobol"
SUBSET_RESULTS = "subset_selection/results"
SUBSET_PLOTS = "plots/subset_selection"
SUBSETS_ALL = "subset_selection/subsets_all"


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
    # We are ranking the performances using percentile form     
    pivot_table = pivot_table.rank(axis=0, pct=True)

    # If the score is NaN we set it to the lowest rank
    pivot_table = pivot_table.fillna(0)

    return pivot_table   


def normalize_min_max(pivot_table: pd.DataFrame) -> pd.DataFrame:        
    def min_max(col):
        # We assign the highest score, i.e., the lowest performance to NaN
        col = col.fillna(col.max())
        return (col - col.min()) / (col.max() - col.min())

    return pivot_table.apply(min_max)    


def prepare_dataset(
        dataset: dict,
        normalization_func: Callable[[pd.DataFrame], pd.DataFrame]
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

        merged[algorithm] = pivot_normalized

    return merged


def generate_subsets(n: int, k: int):
    for subset in itertools.combinations(range(n), k):
        yield subset


def spearman_corr(x, y):
    spcorr, p_value = scipy.stats.spearmanr(x, y)
    return spcorr


def mse_error(target, pred):
    return np.mean((pred - target) ** 2)


def corr_error(target, pred):
    # define the error used to compare linear model, here we pick the model with the highest correlation with the target
    return 1 - spearman_corr(pred, target)


def fit_and_compute_error(
        task_subset: list[int], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray, 
        error_func: Callable
    ) -> tuple[Any, float, float, float]:
    X_train = X_train[:, np.array(task_subset)]
    X_val = X_val[:, np.array(task_subset)]

    lin_model = LinearRegression(positive=True, fit_intercept=False)
    lin_model.fit(X_train, y_train)

    y_train_pred = lin_model.predict(X_train)
    y_val_pred = lin_model.predict(X_val)

    return lin_model, error_func(y_train, y_train_pred), error_func(y_val, y_val_pred), spearman_corr(y_val, y_val_pred)


def cross_validate_scores(task_subset, X, y, error_func: Callable, k: int = 5) -> tuple[float, float, float]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_scores = []
    val_scores = []
    val_correlations = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model, train_score, val_score, val_correlation = fit_and_compute_error(task_subset, X_train, y_train, X_val, y_val, error_func)
        train_scores.append(train_score)
        val_scores.append(val_score)
        val_correlations.append(val_correlation)


    return float(np.mean(train_scores)), float(np.mean(val_scores)), float(np.mean(val_correlations))


def evaluate_subsets(
    X: np.ndarray, y: np.ndarray, n_subset: int, engine: str, error_func: Callable 
) -> tuple:
    subsets = np.array([[x] for x in generate_subsets(X.shape[1], n_subset)])
    print(
        f"Fit linear models for all {len(subsets)} possible task subsets with size={n_subset}."
    )
    scores = parallel_for(
        cross_validate_scores,
        inputs=subsets,
        context={"X": X, "y": y, "error_func": error_func},
        engine=engine,
    )
    train_scores, val_scores, val_correlations = zip(*scores)

    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)
    val_correlations = np.array(val_correlations)

    return subsets, train_scores, val_scores, val_correlations


def subset_selection(
        n_subset: int, 
        normalization_func: Callable, 
        error_func: Callable,
        strategy: str
    ) -> tuple[dict, dict]:
    dataset = read_arlb_dataset()
    training_data = prepare_dataset(dataset, normalization_func=normalization_func)

    scores = {}
    correlations = {}
    for algorithm, pivot_table in training_data.items():
        X = pivot_table.to_numpy()
        pivot_table["target"] = pivot_table.mean(axis=1)
        y = pivot_table["target"].to_numpy()
    
        task_correlation = [
            spearman_corr(X[:, task], y) for task in range(X.shape[1])
        ]

        n_print = 5
        print("\nMost correlated tasks with target:")
        for i, task_index in enumerate(reversed(np.argsort(task_correlation)[-n_print:])):
            env_name = pivot_table.columns[task_index]
            print(
                f"Task {i}: {env_name} ({task_correlation[task_index]:.2f})"
            )
        print("Least correlated tasks with target:")
        for i, task_index in enumerate(np.argsort(task_correlation)[:n_print]):
            env_name = pivot_table.columns[task_index]
            print(
                f"Task {i}: {env_name} ({task_correlation[task_index]:.2f})"
            )

        subsets, train_scores, val_scores, val_correlations = evaluate_subsets(X, y, n_subset, "ray", error_func=error_func)

        # best_subset = subsets[np.argmin(val_scores)][0]
        best_subset_idxs = np.argsort(val_scores)
        best_subsets = subsets[best_subset_idxs][:5]
        best_subset = best_subsets[0, 0]
        best_val_score = val_scores[best_subset_idxs][0]
        best_val_corr = val_correlations[best_subset_idxs][0]

        # fit the best model
        lin_model = LinearRegression(positive=True, fit_intercept=False)
        lin_model.fit(X[:, best_subset], y)

        print(f"Tasks chosen for {algorithm}:")
        for i, task_index in enumerate(best_subset):
            env_name = pivot_table.columns[task_index]
            print(
                f"Task {i}: {env_name}"
            )

        corr_unweighted = spearman_corr(
            X[:, best_subset].mean(axis=1), y
        )

        print(f"Correlation without weights: {corr_unweighted}")
        print(f"Correlation with prediction: {best_val_corr}")
        print(f"Validation error: {best_val_score}")

        with open(os.path.join(SUBSETS_ALL, f"{algorithm}_{n_subset}_{strategy}.txt"), "w") as f:
            f.write(f"Tasks chosen for {algorithm}:\n")
            for i, task_index in enumerate(best_subset):
                env_name = pivot_table.columns[task_index]
                f.write(f"Task {i}: {env_name}\n")

            f.write(f"Correlation without weights: {corr_unweighted}\n")
            f.write(f"Correlation with prediction: {best_val_corr}\n")
            f.write(f"Validation error: {best_val_score}\n")
            f.write(f"Weights:\n")
            f.write(str(lin_model.coef_))

        scores[algorithm] = val_scores[:3]
        correlations[algorithm] = val_correlations[:3]
    return scores, correlations


def get_method_comparison(max_size: int):
    functions = zip(
        [normalize_ranks, normalize_min_max, normalize_ranks, normalize_min_max],
        [corr_error, corr_error, mse_error, mse_error],
        ["Ranks + Spearman", "Min-Max + Spearman", "Ranks + MSE", "Min-Max + MSE"]
    )
    results = []
    for normalization, error, strategy in functions:
        for cur_n_subset in range(1, max_size + 1):
            scores, correlations = subset_selection(
                n_subset=cur_n_subset,
                normalization_func=normalization,
                error_func=error,
                strategy=strategy
            )
            
            for algorithm in scores:
                for i in range(len(scores[algorithm])):
                    results += [{
                        "n_subset": cur_n_subset,
                        "algorithm": algorithm,
                        "score": 0.,
                        "correlation": 1,
                        "strategy": "Optimum",
                    }]
                    results += [{
                        "n_subset": cur_n_subset,
                        "algorithm": algorithm,
                        "score": scores[algorithm][i],
                        "correlation": correlations[algorithm][i],
                        "strategy": strategy,
                    }]
    
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(SUBSET_RESULTS, "method_comparison.csv"), index=False)
    return results

def plot_method_comparison(method_results: pd.DataFrame):
    algorithms = ["ppo", "dqn", "sac"]

    fig, axes = plt.subplots(ncols=len(algorithms), nrows=1, figsize=(9.5, 2.5))

    for i, algorithm in enumerate(algorithms):
        data = method_results[method_results["algorithm"] == algorithm]
        sns.lineplot(data=data, x="n_subset", y="score", hue="strategy", ax=axes[i], errorbar=("ci", 95))
        axes[i].set_title(algorithm.upper())
        axes[i].set_ylabel("")
        axes[i].set_xlabel("Subset Size")
        axes[i].legend().set_visible(False)
        axes[i].set_xticks(np.arange(1, data["n_subset"].max() + 1))

    axes[-1].legend().set_visible(True)
    axes[0].set_ylabel("Validation Error")


    fig.subplots_adjust(bottom=0.3, wspace=0.33)

    axes[-1].legend(loc='upper center', bbox_to_anchor=(-0.95, -0.3), ncol=5, fancybox=False, shadow=False, frameon=False)

    plt.tight_layout()
    path = os.path.join(SUBSET_PLOTS, "method_comparison.png")
    plt.savefig(path, dpi=500)
    plt.show()

def plot_method_comparison_corr(method_results: pd.DataFrame):
    algorithms = ["ppo", "dqn", "sac"]

    fig, axes = plt.subplots(ncols=len(algorithms), nrows=1, figsize=(9.5, 2.5))

    for i, algorithm in enumerate(algorithms):
        data = method_results[method_results["algorithm"] == algorithm]
        sns.lineplot(data=data, x="n_subset", y="correlation", hue="strategy", ax=axes[i], errorbar=("ci", 95))
        axes[i].set_title(algorithm.upper())
        axes[i].set_ylabel("")
        axes[i].set_xlabel("Subset Size")
        axes[i].legend().set_visible(False)
        axes[i].set_xticks(np.arange(1, data["n_subset"].max() + 1))

    axes[-1].legend().set_visible(True)
    axes[0].set_ylabel("Spearman Correlation")

    fig.subplots_adjust(bottom=0.3, wspace=0.33)

    axes[-1].legend(loc='upper center', bbox_to_anchor=(-0.95, -0.3), ncol=5, fancybox=False, shadow=False, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(SUBSET_PLOTS, "method_comparison_correlation.png"), dpi=500)
    plt.show()


def plot_method_comparison_single_strategy(method_results: pd.DataFrame, strategy: str):
    algorithms = ["ppo", "dqn", "sac"]

    method_results = method_results[method_results["strategy"].isin(["Optimum", strategy])]

    fig, axes = plt.subplots(ncols=len(algorithms), nrows=1, figsize=(7, 2), sharey="row")

    for i, algorithm in enumerate(algorithms):
        data = method_results[method_results["algorithm"] == algorithm]
        sns.lineplot(data=data, x="n_subset", y="correlation", hue="strategy", ax=axes[i], errorbar=("ci", 95))
        axes[i].set_title(algorithm.upper())
        axes[i].set_ylabel("")
        axes[i].set_xlabel("Subset Size")
        axes[i].legend().set_visible(False)
        axes[i].set_xticks(np.arange(1, data["n_subset"].max() + 1))

    axes[-1].legend().set_visible(True)
    axes[0].set_ylabel("Spearman Correlation")

    fig.subplots_adjust(bottom=0.305, wspace=0.33)

    axes[-1].legend(loc='upper center', bbox_to_anchor=(-0.95, -0.3), ncol=2, fancybox=False, shadow=False, frameon=False)

    plt.tight_layout()
    path = os.path.join(SUBSET_PLOTS, f"method_comparison_{strategy.replace(' + ', '')}.png")
    plt.savefig(path, dpi=500)
    plt.show()

if __name__ == "__main__":
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    get_method_comparison(6)

    data = pd.read_csv(os.path.join(SUBSET_RESULTS, "method_comparison.csv"))
    plot_method_comparison(data)
    plot_method_comparison_corr(data)
    plot_method_comparison_single_strategy(data, "Ranks + Spearman")

    
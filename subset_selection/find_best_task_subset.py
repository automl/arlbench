import itertools
from typing import List
import os
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression

from create_arlbench_dataset import DATA_DIR
from offline_evaluations import OfflineEvaluations
from parallel_for import parallel_for


def generate_subsets(n: int, k: int):
    for subset in itertools.combinations(range(n), k):
        yield subset


def spearman_corr(x, y):
    spcorr, p_value = scipy.stats.spearmanr(x, y)
    return spcorr


def error(target, pred):
    # define the error used to compare linear model, here we pick the model with the highest correlation with the target
    return 1 - spearman_corr(pred, target)


def fit_and_compute_error(task_subset, train_metrics, train_target):
    X = train_metrics[:, np.array(task_subset)]
    y = train_target
    lin_model = LinearRegression(positive=positive_weights, fit_intercept=fit_intercept)
    lin_model.fit(X, y)
    y_pred = lin_model.predict(X)
    return error(y, y_pred)


def load_random():
    n_models = 256
    n_datasets = 14
    n_seeds = 3
    n_iterations = 100

    metrics = np.random.rand(n_models, n_datasets, n_seeds, n_iterations)

    # Use the average of metrics over seeds and dataset for the target score
    target = metrics.mean(axis=(1, 2, 3))
    higher_is_better = True
    return OfflineEvaluations(
        metrics=metrics,
        datasets=[f"d{i}" for i in range(n_datasets)],
        models=[f"m{i}" for i in range(n_models)],
        target=target,
    ), higher_is_better


def load_arlbench(algorithm="ppo"):
    data = OfflineEvaluations.load(path=os.path.join(DATA_DIR, algorithm))
    higher_is_better = True
    return data, higher_is_better


def atari5_strategy(
    n_subset: int, n_tasks: int, metrics_flatten, target, engine
) -> List[int]:
    subsets = [[x] for x in generate_subsets(n_tasks, n_subset)]
    print(
        f"Fit linear models for all {len(subsets)} possible task subsets with size={n_subset}."
    )
    correlations = parallel_for(
        fit_and_compute_error,
        inputs=subsets,
        context={"train_metrics": metrics_flatten, "train_target": target},
        engine=engine,
    )
    best_subset = subsets[np.argmin(correlations)][0]

    return best_subset


def main(data, higher_is_better):
    # values are ranked between models, when data is missing we impute the worse value which is 0 if higher is better
    impute_rank_value = 0 if higher_is_better else 1

    n_models, n_datasets, _, _ = data.metrics.shape

    # use rank for the target
    order = data.target.argsort()
    target = order.argsort()
    target = target / len(target)

    # Use the last iteration only in case we have several
    # (n_models, n_datasets, n_seeds, n_iterations) -> (n_models, n_datasets, n_seeds)
    metrics = data.metrics[:, :, :, -1]

    # Fow now takes the average over seeds, this enforce that we select environment and not environment-seed combination
    # (n_models, n_datasets, n_seeds, 1) -> (n_models, n_datasets)
    metrics = metrics.mean(axis=2)

    # Use ranking to normalize across environments
    metrics = pd.DataFrame(metrics).rank(axis=0, pct=True)

    # Impute missing values (needed for Atari5 data where some models are partly evaluated), we fill the worse score
    metrics = metrics.fillna(impute_rank_value).values

    # Reshape to (n_models, n_datasets, 1, 1), we do this so that this code can possibly be adapted to select
    # environment-seed combinations
    metrics = metrics.reshape(n_models, n_datasets, 1, 1)
    n_models, n_datasets, n_seeds, n_iterations = metrics.shape

    metrics_flatten = metrics.reshape(n_models, -1)
    n_tasks = metrics_flatten.shape[1]

    # Mapping to recover dataset/seed/iteration from flatten task indices
    taskindex_to_tuple = {
        i: (dataset, seed, iteration)
        for i, (dataset, seed, iteration) in enumerate(
            itertools.product(range(n_datasets), range(n_seeds), range(n_iterations))
        )
    }

    task_correlation = [
        spearman_corr(metrics_flatten[:, task], target) for task in range(n_tasks)
    ]

    n_print = 5
    print("\nMost correlated tasks with target:")
    for i, task_index in enumerate(np.argsort(task_correlation)[-n_print:]):
        dataset_index, seed_index, iteration_index = taskindex_to_tuple[task_index]
        print(
            f"Task {i}: {data.datasets[dataset_index]}-seed-{seed_index}-iteration-{iteration_index} ({task_correlation[task_index]:.2f})"
        )

    print("Least correlated tasks with target:")
    for i, task_index in enumerate(np.argsort(task_correlation)[:n_print]):
        dataset_index, seed_index, iteration_index = taskindex_to_tuple[task_index]
        print(
            f"Task {i}: {data.datasets[dataset_index]}-seed-{seed_index}-iteration-{iteration_index} ({task_correlation[task_index]:.2f})"
        )

    # ~1 min on my mac in parallel with subset=5 and atari 5
    task_order = atari5_strategy(n_subset, n_tasks, metrics_flatten, target, engine)

    print("Tasks chosen with Atari5 strategy:")
    for i, task_index in enumerate(task_order):
        dataset_index, seed_index, iteration_index = taskindex_to_tuple[task_index]
        print(
            f"Task {i}: {data.datasets[dataset_index]}-seed-{seed_index}-iteration-{iteration_index}"
        )

    # correlation when taking the average of task selected
    corr_unweighted = spearman_corr(
        metrics[:, task_order, :, :].mean(axis=(1, 2, 3)), target
    )
    corr_weighted = 1 - fit_and_compute_error(task_order, metrics_flatten, target)

    print(f"Correlation without weights: {corr_unweighted}")
    print(f"Correlation with weights: {corr_weighted}")

    n_draws = 100
    corrs = []
    for _ in range(n_draws):
        random_tasks = np.random.choice(np.arange(n_tasks), n_subset)
        corrs.append(
            spearman_corr(metrics[:, random_tasks, :, :].mean(axis=(1, 2, 3)), target)
        )
    print(f"Average correlation with {n_subset} random tasks: {np.mean(corrs)}")


if __name__ == "__main__":
    n_subset = 4  # use "ray" to enable parallelism when using a large number of subsets
    # engine = "ray"
    engine = "sequential"

    # whether to use only positive weights in linear model
    # I feel skeptical about negative weights and generalization for this use-case
    positive_weights = True
    fit_intercept = False  # whether to fit an intercept in the linear model

    # Note the correlation with Atari5 are not very high but this is due to a lack of data-cleaning compared to the original paper
    # In other cases, I have obtained much larger correlations.
    data, higher_is_better = load_arlbench()

    # Example to show how to load from numpy data
    # data, higher_is_better = load_random()

    main(data, higher_is_better)

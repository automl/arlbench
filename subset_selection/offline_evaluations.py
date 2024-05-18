import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


class OfflineEvaluations:
    def __init__(
        self,
        metrics: np.array,
        datasets: List[str],
        models: List[str],
        target: np.array = None,
    ):
        """
        :param metrics: (n_models, n_datasets, n_seeds, n_iterations)
        :param datasets: names of the datasets with length n_datasets
        :param datasets: names of the models with length n_models
        :param target: value of the targets that should be approximated with length n_models
        can be the average of model performance across seeds, iterations and datasets or something else
        """
        assert metrics.ndim == 4
        (n_models, n_datasets, n_seeds, n_iterations) = metrics.shape
        if target is not None:
            assert len(target) == n_models
        assert len(datasets) == n_datasets
        assert len(models) == n_models

        # TODO check nans?
        self.metrics = metrics
        self.datasets = datasets
        self.models = models
        self.target = target if target is not None else metrics.mean(axis=(1, 2, 3))

    def __str__(self):
        stats = ["models", "datasets", "seeds", "iterations"]
        stat_str = ", ".join(
            [f"{num} {stat}" for num, stat in zip(self.metrics.shape, stats)]
        )
        descr = f"Evaluations with {stat_str}."
        sorted_metrics = sorted(self.model_metrics().items(), key=lambda x: x[1])
        best_method = sorted_metrics[-1]
        best_method_str = f"{best_method[0]} ({best_method[1]})"
        worst_method = sorted_metrics[0]
        worst_method_str = f"{worst_method[0]} ({worst_method[1]})"
        descr += f"\nBest method: {best_method_str}, worst method: {worst_method_str}."
        return descr

    def model_metrics(self) -> Dict[str, float]:
        return dict(zip(self.models, self.target))

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "metadata.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "datasets": self.datasets,
                        "models": self.models,
                    }
                )
            )

        np.save(path / "scores.npy", self.metrics)
        np.save(path / "target.npy", self.target)

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        res = cls(
            metrics=np.load(path / "scores.npy"),
            target=np.load(path / "target.npy"),
            **metadata,
        )
        print(f"Loaded {path}: {res}")
        return res

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target: Dict[str, float] = None,
        model_col: str | None = "model",
        dataset_col: str | None = "dataset",
        seed_col: str | None = "seed",
        iteration_col: str | None = "iteration",
        metric_col: str | None = "metric",
    ):
        """
        :param df: dataframe that contains evaluations to be converted
        :param target: dictionary from model to target value (if not passed use the average metric of the model on
        available datasets
        :param model_col:
        :param dataset_col:
        :param seed_col: if None, the column is created
        :param iteration_col: if None, the column is created
        :param metric_col:
        :return:
        """

        if iteration_col is None:
            print(
                "No iteration columns specified, creating iteration column with dummy value."
            )
            iteration_col = "iteration"
            df.loc[:, iteration_col] = 0
        if seed_col is None:
            print(
                "No seed columns specified, creating iteration column with dummy value."
            )
            seed_col = "seed"
            df.loc[:, seed_col] = 0
        for col in [model_col, dataset_col, seed_col, iteration_col, metric_col]:
            assert col in df.columns, f"Missing column {col}"
        df_pivot = df.pivot_table(
            index=model_col,
            columns=[dataset_col, seed_col, iteration_col],
            values=metric_col,
            dropna=False,
            fill_value=None,
        )
        models = df.loc[:, model_col].unique().tolist()
        datasets = df.loc[:, dataset_col].unique().tolist()
        n_models = len(models)
        n_datasets = len(datasets)
        n_seeds = len(df.loc[:, seed_col].unique())
        n_iterations = len(df.loc[:, iteration_col].unique())
        metric_tensor = df_pivot.values.reshape(
            n_models, n_datasets, n_seeds, n_iterations
        )
        # if target is None:
        #     # Takes mean over seeds and dataset at the last iteration
        #     target = {model: metric_tensor[i, :, :, -1].mean(axis=(0, 1)) for i, model in enumerate(models)}
        return cls(
            metrics=metric_tensor,
            datasets=datasets,
            models=models,
            target=[target[model] for model in models],
        )

    def select_models(self, models: List[str]):
        # Returns evaluations where only some models are present
        assert all(m in self.models for m in models)
        model_indices = np.array(
            [i for i, model in enumerate(self.models) if model in models]
        )
        target = self.target[model_indices]
        return OfflineEvaluations(
            metrics=self.metrics[model_indices],
            datasets=self.datasets,
            models=models,
            target=target,
        )

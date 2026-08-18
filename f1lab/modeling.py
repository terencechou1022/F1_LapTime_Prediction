"""Model training and evaluation classes.

`ModelTrainer` runs a strict three-tier holdout:
    - Train: first 80% (time-ordered) with TimeSeriesSplit CV inside
    - Valid: last 20% (within-era held-out)
`ModelEvaluator` runs the saved model on a separate test file (e.g. a
held-out 2025 race), with optional median-residual bias correction
to address year-on-year drift inside the same regulatory era.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from f1lab.models import get_model_spec
from f1lab.preprocessing import BaseLapPreprocessor


@dataclass
class Metrics:
    """Regression metrics: MAE, MSE, RMSE, R²."""
    mae: float
    mse: float
    rmse: float
    r2: float

    @classmethod
    def compute(cls, y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> "Metrics":
        mse_val = float(mean_squared_error(y_true, y_pred))
        return cls(
            mae=float(mean_absolute_error(y_true, y_pred)),
            mse=mse_val,
            rmse=float(np.sqrt(mse_val)),
            r2=float(r2_score(y_true, y_pred)),
        )

    def report(self, prefix: str = "") -> None:
        tag = f"{prefix} " if prefix else ""
        print(f"{tag}MAE:  {self.mae:.3f}")
        print(f"{tag}MSE:  {self.mse:.3f}")
        print(f"{tag}RMSE: {self.rmse:.3f}")
        print(f"{tag}R2:   {self.r2:.3f}")


class ModelTrainer:
    """Train a regression model with TimeSeriesSplit hyperparameter search.

    Model class is selected by name ('dt', 'rf', 'xgb'); see
    `f1lab.models.MODEL_SPECS` for the per-model estimator factories and
    grids. Param grid can be overridden via `param_grid=`; `quick=True`
    selects the tiny smoke-test grid instead of the full one.
    """

    DEFAULT_MODEL: ClassVar[str] = "xgb"

    def __init__(
        self,
        preprocessor: BaseLapPreprocessor,
        model_name: str = DEFAULT_MODEL,
        param_grid: dict[str, list[Any]] | None = None,
        test_size: float = 0.2,
        cv_splits: int = 5,
        random_state: int = 42,
        quick: bool = False,
    ) -> None:
        factory, default_grid = get_model_spec(model_name, quick=quick)
        self.preprocessor = preprocessor
        self.model_name = model_name
        self._factory = factory
        self.param_grid = param_grid or default_grid
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.x_train: pd.DataFrame | None = None
        self.x_valid: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_valid: pd.Series | None = None
        self.model: BaseEstimator | None = None
        self.best_params: dict[str, Any] | None = None
        self.cv_results_: dict[str, Any] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

    def fit(self) -> BaseEstimator:
        _, x, y = self.preprocessor.run()
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            x, y, test_size=self.test_size, shuffle=False
        )

        grid = GridSearchCV(
            estimator=self._factory(self.random_state),
            param_grid=self.param_grid,
            cv=TimeSeriesSplit(n_splits=self.cv_splits),
            scoring="r2",
            n_jobs=-1,
        )
        grid.fit(self.x_train, self.y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.cv_results_ = grid.cv_results_
        return self.model

    def predict_valid(self) -> np.ndarray:
        self._require_fitted()
        return self.model.predict(self.x_valid)

    def report(self) -> Metrics:
        y_pred = self.predict_valid()
        print(f"Model: {self.model_name}")
        print(f"Best params: {self.best_params}")
        metrics = Metrics.compute(self.y_valid, y_pred)
        metrics.report(prefix="[holdout]")
        return metrics

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved: {path}")


@dataclass
class EvaluationResult:
    metrics: Metrics
    metrics_corrected: Metrics | None = None
    bias_offset: float | None = None
    y_true: pd.Series = field(default_factory=pd.Series)
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_corrected: np.ndarray | None = None
    x: pd.DataFrame | None = None  # feature matrix used for prediction; exposed for residual-vs-feature plots


class ModelEvaluator:
    """Run a saved model against a separate dataset.

    Optionally apply median-residual bias correction to remove
    year-on-year drift within the same regulatory era.
    """

    def __init__(
        self,
        model_path: str | Path,
        preprocessor_cls: type[BaseLapPreprocessor],
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model_path = path
        self.preprocessor_cls = preprocessor_cls
        self.model: BaseEstimator = joblib.load(path)

    @property
    def feature_names(self) -> list[str] | None:
        return list(self.model.feature_names_in_) if hasattr(self.model, "feature_names_in_") else None

    def evaluate(self, data_path: str | Path, bias_correct: bool = False) -> EvaluationResult:
        preprocessor = self.preprocessor_cls.from_excel(data_path, features=self.feature_names)
        _, x, y = preprocessor.run()
        y_pred = self.model.predict(x)

        result = EvaluationResult(
            metrics=Metrics.compute(y, y_pred),
            y_true=y,
            y_pred=y_pred,
            x=x,
        )
        result.metrics.report(prefix="[raw]")

        if bias_correct:
            offset = float(np.median(y - y_pred))
            y_pred_adj = y_pred + offset
            result.bias_offset = offset
            result.y_pred_corrected = y_pred_adj
            result.metrics_corrected = Metrics.compute(y, y_pred_adj)
            print(f"Median residual offset: {offset:.2f} s")
            result.metrics_corrected.report(prefix="[bias-corrected]")

        return result

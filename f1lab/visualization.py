"""Plotting helpers — stateless static methods on a single class.

Every plot function accepts an optional `save_path=`. When set, the figure
is written to disk at 150 DPI and closed; otherwise it is shown interactively.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.inspection import PartialDependenceDisplay


def _to_array(y: pd.Series | np.ndarray) -> np.ndarray:
    return y.values if hasattr(y, "values") else np.asarray(y)


def _emit(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Either save the figure to `save_path` (closing it after) or show()."""
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _lowess_overlay(ax: plt.Axes, x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> None:
    """Add a LOWESS smoothing line if statsmodels is installed.

    Silently skips the overlay when statsmodels is missing — diagnostic
    plots still render without the smoothed reference curve.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        return
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="darkorange", linewidth=2, label="LOWESS")
    ax.legend(loc="best")


class Visualizer:
    """Static plotting helpers for regression diagnostics."""

    DEFAULT_Y_LABEL: str = "LapTimeDelta (Seconds)"

    @staticmethod
    def feature_importance(
        model: BaseEstimator,
        columns: list[str],
        save_path: str | Path | None = None,
    ) -> None:
        importances = pd.DataFrame({
            "Feature": columns,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x="Importance", y="Feature", data=importances,
            hue="Feature", palette="viridis", legend=False, ax=ax,
        )
        ax.set_title("Feature Importance")
        _emit(fig, save_path)

    @staticmethod
    def prediction_vs_actual(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_label: str = DEFAULT_Y_LABEL,
        title: str = "Predicted vs Actual",
        save_path: str | Path | None = None,
    ) -> None:
        y_true_arr = _to_array(y_true)
        idx = range(len(y_true_arr))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(idx, y_true_arr, label="Actual", marker="o", alpha=0.7)
        ax.plot(idx, y_pred, label="Predicted", linestyle="-", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Sample Sequence")
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        _emit(fig, save_path)

    @staticmethod
    def residual_distribution(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Distribution",
        save_path: str | Path | None = None,
    ) -> None:
        residual = _to_array(y_true) - y_pred

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(residual, kde=True, color="purple", ax=ax)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Prediction Error (Seconds)")
        _emit(fig, save_path)

    @staticmethod
    def residual_vs_predicted(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual vs Predicted",
        save_path: str | Path | None = None,
    ) -> None:
        """Scatter of residuals against predicted values, with LOWESS overlay.

        Reveals heteroscedasticity, systematic bias, and individual burst values.
        """
        y_pred_arr = _to_array(y_pred)
        residual = _to_array(y_true) - y_pred_arr

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_pred_arr, residual, alpha=0.4, s=15, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        _lowess_overlay(ax, y_pred_arr, residual)
        ax.set_xlabel("Predicted LapTimeDelta (s)")
        ax.set_ylabel("Residual (Actual − Predicted, s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        _emit(fig, save_path)

    @staticmethod
    def residual_vs_feature(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        feature_values: pd.Series | np.ndarray,
        feature_name: str,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        """Scatter of residuals against a single feature, with LOWESS overlay.

        Locates *where in feature space* the model fails — a clean PDP reading
        depends on residuals being flat across the feature range.
        """
        feature_arr = _to_array(feature_values)
        residual = _to_array(y_true) - _to_array(y_pred)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(feature_arr, residual, alpha=0.4, s=15, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        _lowess_overlay(ax, feature_arr, residual)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Residual (Actual − Predicted, s)")
        ax.set_title(title or f"Residual vs {feature_name}")
        ax.grid(True, alpha=0.3)
        _emit(fig, save_path)

    @staticmethod
    def partial_dependence(
        model: BaseEstimator,
        x: pd.DataFrame,
        feature: str,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        """Partial Dependence Plot (average marginal effect) for one feature.

        Used to extract the response function f(feature) → LapTimeDelta from
        the trained model; the strategy correction term (`f1lab.strategy`)
        reads its values off this curve.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(
            model, x, features=[feature], kind="average", ax=ax,
            line_kw={"linewidth": 2, "color": "steelblue"},
        )
        ax.set_title(title or f"PDP — {feature} → LapTimeDelta")
        ax.set_ylabel("Partial Dependence on LapTimeDelta (s)")
        ax.grid(True, alpha=0.3)
        _emit(fig, save_path)

    @staticmethod
    def ice(
        model: BaseEstimator,
        x: pd.DataFrame,
        feature: str,
        subsample: int = 200,
        random_state: int = 42,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        """ICE lines + PDP overlay — checks the average reaction is not pulled
        by a few samples (per-stint reaction curves should share the shape)."""
        x_sub = x.sample(n=subsample, random_state=random_state) if len(x) > subsample else x

        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(
            model, x_sub, features=[feature], kind="both", ax=ax,
            ice_lines_kw={"alpha": 0.25, "linewidth": 0.5, "color": "steelblue"},
            pd_line_kw={"linewidth": 2.5, "color": "darkorange", "label": "PDP (average)"},
        )
        ax.set_title(title or f"ICE + PDP overlay — {feature}")
        ax.set_ylabel("Partial Dependence on LapTimeDelta (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        _emit(fig, save_path)

"""繪圖輔助函式——集中在單一類別上的無狀態靜態方法。

每個繪圖函式都接受可選的 `save_path=`。有給值時，圖會以 150 DPI 寫到磁碟
然後關閉；沒給就以互動方式顯示。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.inspection import PartialDependenceDisplay


def _to_array(y: pd.Series | np.ndarray) -> np.ndarray:
    return y.values if hasattr(y, "values") else np.asarray(y)


def _emit(fig: plt.Figure, save_path: str | Path | None) -> None:
    """把圖存到 `save_path`（存完關閉），或者直接 show()。"""
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _lowess_overlay(ax: plt.Axes, x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> None:
    """statsmodels 有裝的話，加一條 LOWESS 平滑線。

    沒裝 statsmodels 時安靜略過這層疊圖——診斷圖少了平滑參考曲線
    仍然畫得出來。
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        return
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="darkorange", linewidth=2, label="LOWESS")
    ax.legend(loc="best")


class Visualizer:
    """迴歸診斷用的靜態繪圖輔助函式。"""

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
        """殘差對預測值的散布圖，疊上 LOWESS。

        用來看出異質變異、系統性偏差，以及個別暴衝的值。
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
        """殘差對單一特徵的散布圖，疊上 LOWESS。

        用來定位模型是在特徵空間的「哪裡」失準——要讀得出乾淨的 PDP，
        前提是殘差在整個特徵範圍上是平的。
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
        """單一特徵的部分依賴圖（平均邊際效應）。

        用來從訓練好的模型抽出反應函式 f(特徵) → LapTimeDelta；
        策略層的修正項（`f1lab.strategy`）就是從這條曲線讀值的。
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
        """ICE 曲線疊上 PDP——檢查平均反應沒有被少數樣本拉走
        （各 stint 的反應曲線應該共享同一個形狀）。"""
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

    @staticmethod
    def monte_carlo_distribution(
        result: Any,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        """繪製蒙特卡羅模擬之超車餘裕（Margin）機率分佈對比圖。

        若環境條件在訓練支撐區間內，同時疊加「未修正」與「修正後」兩條分佈曲線，
        並標出 margin = 0 的勝負分界線。
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 未修正情境分佈
        sns.histplot(
            result.margins_uncorrected,
            kde=True,
            stat="density",
            color="slategray",
            alpha=0.35,
            label=f"Base (Uncorrected) — Win Rate: {result.success_prob_uncorrected * 100:.1f}%",
            ax=ax,
        )
        ax.axvline(
            result.mean_margin_uncorrected,
            color="slategray",
            linestyle=":",
            linewidth=1.5,
            label=f"Base Mean ({result.mean_margin_uncorrected:+.2f} s)",
        )

        # 修正後情境分佈（若可用）
        if result.applicable and result.margins_corrected is not None:
            sns.histplot(
                result.margins_corrected,
                kde=True,
                stat="density",
                color="forestgreen",
                alpha=0.45,
                label=f"Corrected (PDP Adjustment) — Win Rate: {result.success_prob_corrected * 100:.1f}%",
                ax=ax,
            )
            ax.axvline(
                result.mean_margin_corrected,
                color="forestgreen",
                linestyle=":",
                linewidth=1.8,
                label=f"Corrected Mean ({result.mean_margin_corrected:+.2f} s)",
            )

        # 0 軸臨界線 (margin > 0 為 Undercut 成功)
        ax.axvline(0, color="crimson", linestyle="--", linewidth=2.0, label="Breakeven (Margin = 0)")

        # 標註成功區域
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.axvspan(0, max(xlim[1], 3.0), color="green", alpha=0.06)

        default_title = (
            f"Monte Carlo Undercut Simulation — {result.feature} (N={result.config.n_simulations:,})\n"
            f"Win Rate: {result.success_prob_uncorrected * 100:.1f}%"
        )
        if result.applicable and result.success_prob_corrected is not None:
            diff = (result.success_prob_corrected - result.success_prob_uncorrected) * 100
            default_title += f" → {result.success_prob_corrected * 100:.1f}% ({diff:+.1f}%)"
        else:
            default_title += " [OOD Withheld]"

        ax.set_title(title or default_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Undercut Margin (s)  [ > 0: Undercut Successful (OPEN) ]")
        ax.set_ylabel("Probability Density")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.9)
        _emit(fig, save_path)


"""CLI：蒙特卡羅進站決策模擬（不確定性分析與勝率分佈）。

載入兩項研究的 RF 模型與對稱進站參數，執行 10,000 次蒙特卡羅抽樣，
量化換胎時間失誤、圈速波動與環境條件擾動對提前進站（Undercut）勝率的影響：

    1. wind（沙烏地阿拉伯 2025）：當前逆風在訓練支撐區間內
       → 計算環境修正前與修正後之超車勝率 P(margin > 0) 與 95% 信賴區間
    2. temp（拉斯維加斯 2025）：當前賽道溫度為 OOD
       → 抽樣檢驗觸發 OOD 門檻，安全收回修正項並拒絕預測

輸出：
    plots/monte_carlo_undercut_wind.png
    plots/monte_carlo_undercut_temp.png
    stdout: 統計摘要表與勝率對比

使用方式：
    python scripts/monte_carlo.py
    python scripts/monte_carlo.py --n-simulations 20000 --save-plots-dir plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend

import _bootstrap  # noqa: F401
import joblib

from f1lab import (
    MonteCarloConfig,
    MonteCarloUndercut,
    TempPreprocessor,
    UndercutScenario,
    Visualizer,
    WindPreprocessor,
)

OUT_DIR = _bootstrap.PROJECT_ROOT / "plots"

# 與 mechanism.py 完全相同的對稱情境參數
SCENARIO = dict(pit_loss=20.0, gap=19.50, ours_new_outlap=95.2, rival_old_inlap=95.5, n_remaining=10)

_STUDIES = {
    "wind": {
        "preprocessor": WindPreprocessor,
        "model": "models/azerbaijan_rf.joblib",
        "data": "data/merged/2022-2024_Azerbaijan_Grand_Prix.xlsx",
        "target_data": "data/merged/2025_Saudi_Arabian_Grand_Prix.xlsx",
        "feature": "HeadWind",
        "plot_name": "monte_carlo_undercut_wind.png",
    },
    "temp": {
        "preprocessor": TempPreprocessor,
        "model": "models/singapore_rf.joblib",
        "data": "data/merged/2022-2024_Singapore_Grand_Prix.xlsx",
        "target_data": "data/merged/2025_Las_Vegas_Grand_Prix.xlsx",
        "feature": "TrackTemp",
        "plot_name": "monte_carlo_undercut_temp.png",
    },
}


def _load(study: dict) -> tuple:
    model = joblib.load(_bootstrap.PROJECT_ROOT / study["model"])
    _, x, _ = study["preprocessor"].from_excel(str(_bootstrap.PROJECT_ROOT / study["data"])).run()
    return model, x


def main() -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo Undercut Strategy Simulation")
    parser.add_argument("--n-simulations", type=int, default=10000, help="Number of Monte Carlo iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving diagnostic plots.")
    parser.add_argument("--save-plots-dir", type=Path, default=OUT_DIR, help="Directory to save plots.")
    args = parser.parse_args()

    plots_dir = args.save_plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("monte_carlo.py — Probabilistic Pit Stop Simulation under Uncertainty")
    print("=" * 70)
    print(f"  Iterations per scenario : {args.n_simulations:,}")
    print(f"  Random seed             : {args.seed}")
    print(f"  Fixed scenario params   : {SCENARIO}")

    cfg = MonteCarloConfig(n_simulations=args.n_simulations, random_state=args.seed)

    # 1. WIND STUDY — 沙烏地阿拉伯 2025 (In-Domain Support)
    print("\n" + "-" * 70)
    print("Scenario 1: Saudi Arabia 2025 (Wind / In-support)")
    print("-" * 70)
    wind_study = _STUDIES["wind"]
    wind_model, x_wind = _load(wind_study)
    wind_scenario = UndercutScenario(wind_model, x_wind, wind_study["feature"], **SCENARIO)

    _, x_saudi, _ = WindPreprocessor.from_excel(str(_bootstrap.PROJECT_ROOT / wind_study["target_data"])).run()
    current_hw = float(x_saudi["HeadWind"].max())

    mc_wind = MonteCarloUndercut(wind_scenario, cfg).simulate(current_hw)
    mc_wind.report()

    if not args.no_plots:
        save_path = plots_dir / wind_study["plot_name"]
        Visualizer.monte_carlo_distribution(
            mc_wind,
            title=(
                f"Saudi Arabia 2025 Undercut — Wind Effect (N={args.n_simulations:,})\n"
                f"Win Rate: {mc_wind.success_prob_uncorrected * 100:.1f}% → "
                f"{mc_wind.success_prob_corrected * 100:.1f}% "
                f"({(mc_wind.success_prob_corrected - mc_wind.success_prob_uncorrected) * 100:+.1f}%)"
            ),
            save_path=save_path,
        )
        print(f"\n  Saved plot: {save_path}")

    # 2. TEMP STUDY — 拉斯維加斯 2025 (OOD)
    print("\n" + "-" * 70)
    print("Scenario 2: Las Vegas 2025 (Temp / OOD)")
    print("-" * 70)
    temp_study = _STUDIES["temp"]
    temp_model, x_temp = _load(temp_study)
    temp_scenario = UndercutScenario(temp_model, x_temp, temp_study["feature"], **SCENARIO)

    _, x_vegas, _ = TempPreprocessor.from_excel(str(_bootstrap.PROJECT_ROOT / temp_study["target_data"])).run()
    current_temp = float(x_vegas["TrackTemp"].mean())

    mc_temp = MonteCarloUndercut(temp_scenario, cfg).simulate(current_temp)
    mc_temp.report()

    if not args.no_plots:
        save_path = plots_dir / temp_study["plot_name"]
        Visualizer.monte_carlo_distribution(
            mc_temp,
            title=f"Las Vegas 2025 Undercut — Temp Effect (N={args.n_simulations:,}) [OOD Withheld]",
            save_path=save_path,
        )
        print(f"\n  Saved plot: {save_path}")

    print("\n" + "=" * 70)
    print("Monte Carlo Simulation completed successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

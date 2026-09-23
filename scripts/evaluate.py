"""CLI：把訓練好的模型拿到保留的賽事檔上評估。"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from f1lab import ModelEvaluator, Visualizer, get_preprocessor
from f1lab.models import MODEL_NAMES

# 四個項目都預設 bias_correct=False，這樣沒有明確給旗標時，四項研究的行為
# 完全一致。偏差修正是使用者透過 --bias-correct 主動選用的；原始那一跑
# 才是自然的基準。
_DEFAULTS: dict[str, dict[str, object]] = {
    "wind": {
        "data": "data/merged/2025_Azerbaijan_Grand_Prix.xlsx",
        "model_template": "models/azerbaijan_{model}.joblib",
        "preprocessor": "wind",
        "bias_correct": False,
    },
    "wind-cross-domain": {
        "data": "data/merged/2025_Saudi_Arabian_Grand_Prix.xlsx",
        "model_template": "models/azerbaijan_{model}.joblib",
        "preprocessor": "wind",
        "bias_correct": False,
    },
    "temp": {
        "data": "data/merged/2025_Singapore_Grand_Prix.xlsx",
        "model_template": "models/singapore_{model}.joblib",
        "preprocessor": "temp",
        "bias_correct": False,
    },
    "temp-cross-domain": {
        "data": "data/merged/2025_Las_Vegas_Grand_Prix.xlsx",
        "model_template": "models/singapore_{model}.joblib",
        "preprocessor": "temp",
        "bias_correct": False,
    },
}

# 殘差對特徵的散布圖：每項研究要對哪些特徵欄位作圖。每份清單包含：
#   - 2 個該研究的因果軸（HeadWind/CrossWind 或 AirTemp/TrackTemp）——
#     用於機制估計的診斷
#   - 4 個輪胎／stint 進程特徵（LapNumber、LapInStint、TyreLife、
#     TyreLifeNorm）——同時涵蓋比賽與 stint 位置的絕對與相對視角，
#     方便定位模型是在 stint 的哪個階段失準
# 二元特徵（FreshTyre、Compound_*）與 Rainfall 刻意排除：
# 對著近乎二元的軸作散布圖，診斷價值很低。
_RESIDUAL_FEATURES: dict[str, list[str]] = {
    "wind": ["HeadWind", "CrossWind", "LapNumber", "LapInStint", "TyreLife", "TyreLifeNorm"],
    "temp": ["AirTemp",  "TrackTemp", "LapNumber", "LapInStint", "TyreLife", "TyreLifeNorm"],
}


def _plot_prefix(experiment: str, bias_correct: bool, model_name: str) -> str:
    """組出與 log 檔命名慣例一致的檔名前綴：
    eval_{study}_{domain}_{mode}_{model}_..."""
    study = "wind" if experiment.startswith("wind") else "temp"
    domain = "crossdomain" if "cross-domain" in experiment else "indomain"
    mode = "bias" if bias_correct else "raw"
    return f"eval_{study}_{domain}_{mode}_{model_name}"


def _evaluate_one(
    experiment: str,
    model_name: str,
    data_path: Path,
    model_path: Path,
    bias_correct: bool,
    no_plots: bool,
    save_plots_dir: Path | None,
) -> None:
    print(f"\n=== Evaluating {model_name.upper()} on '{experiment}' ===")
    defaults = _DEFAULTS[experiment]
    study = str(defaults["preprocessor"])
    preprocessor_cls = get_preprocessor(study)
    evaluator = ModelEvaluator(model_path, preprocessor_cls)
    result = evaluator.evaluate(data_path, bias_correct=bias_correct)

    if no_plots:
        return

    # 挑要畫哪一組殘差（原始 vs 偏差修正後）
    y_pred_for_plots = result.y_pred_corrected if bias_correct else result.y_pred

    if save_plots_dir is not None:
        prefix = _plot_prefix(experiment, bias_correct, model_name)
        out = save_plots_dir
        out.mkdir(parents=True, exist_ok=True)

        # 既有的診斷圖（存檔而不顯示）
        Visualizer.prediction_vs_actual(
            result.y_true, y_pred_for_plots,
            title=f"Predicted vs Actual — {prefix}",
            save_path=out / f"{prefix}_prediction_vs_actual.png",
        )
        Visualizer.residual_distribution(
            result.y_true, y_pred_for_plots,
            title=f"Residual Distribution — {prefix}",
            save_path=out / f"{prefix}_residual_distribution.png",
        )

        # 新增：殘差對預測值（#1）
        Visualizer.residual_vs_predicted(
            result.y_true, y_pred_for_plots,
            title=f"Residual vs Predicted — {prefix}",
            save_path=out / f"{prefix}_residual_vs_predicted.png",
        )

        # 新增：殘差對特徵（#3）——每項研究 4 個特徵
        for feature in _RESIDUAL_FEATURES[study]:
            if result.x is None or feature not in result.x.columns:
                print(f"  WARN: feature '{feature}' missing from X; skipping its plot")
                continue
            Visualizer.residual_vs_feature(
                result.y_true, y_pred_for_plots,
                result.x[feature], feature,
                title=f"Residual vs {feature} — {prefix}",
                save_path=out / f"{prefix}_residual_vs_{feature}.png",
            )
        print(f"  plots saved to {out}/")
        return

    # 互動模式（沒給 --save-plots-dir、也沒給 --no-plots）：直接顯示在螢幕上
    Visualizer.prediction_vs_actual(result.y_true, result.y_pred, title="Predicted vs Actual (raw)")
    Visualizer.residual_distribution(result.y_true, result.y_pred)
    Visualizer.residual_vs_predicted(result.y_true, result.y_pred)
    for feature in _RESIDUAL_FEATURES[study]:
        if result.x is not None and feature in result.x.columns:
            Visualizer.residual_vs_feature(result.y_true, result.y_pred, result.x[feature], feature)
    if result.y_pred_corrected is not None:
        Visualizer.prediction_vs_actual(
            result.y_true, result.y_pred_corrected,
            title="Predicted vs Actual (bias-corrected)",
        )
        Visualizer.residual_distribution(
            result.y_true, result.y_pred_corrected,
            title="Residual Distribution (bias-corrected)",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained model(s).")
    parser.add_argument("--experiment", required=True, choices=sorted(_DEFAULTS.keys()))
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "all"],
        default="all",
        help="Model class to evaluate. 'all' (default) evaluates DT, RF, and XGBoost sequentially.",
    )
    parser.add_argument("--data", type=Path, help="Override test data path.")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override model file path. Only valid with a single --model (not 'all').",
    )
    parser.add_argument("--bias-correct", action="store_true", help="Force bias correction on.")
    parser.add_argument("--no-bias-correct", action="store_true", help="Force bias correction off.")

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--no-plots", action="store_true",
        help="Skip all plotting (fastest; for headless metric-only runs).",
    )
    plot_group.add_argument(
        "--save-plots-dir", type=Path,
        help="Save diagnostic plots (residual-vs-predicted, residual-vs-feature, "
             "residual distribution, prediction-vs-actual) to this directory instead "
             "of showing them. Filenames follow eval_{study}_{domain}_{mode}_{model}_<type>.png.",
    )
    args = parser.parse_args()

    defaults = _DEFAULTS[args.experiment]
    data_path = args.data or _bootstrap.PROJECT_ROOT / str(defaults["data"])

    bias_correct = bool(defaults["bias_correct"])
    if args.bias_correct:
        bias_correct = True
    if args.no_bias_correct:
        bias_correct = False

    if args.model_path is not None and args.model == "all":
        parser.error("--model-path cannot be used with --model all.")

    save_plots_dir: Path | None = None
    if args.save_plots_dir is not None:
        save_plots_dir = args.save_plots_dir
        if not save_plots_dir.is_absolute():
            save_plots_dir = _bootstrap.PROJECT_ROOT / save_plots_dir

    models_to_eval = list(MODEL_NAMES) if args.model == "all" else [args.model]

    for model_name in models_to_eval:
        if args.model_path is not None:
            model_path = args.model_path
        else:
            template = str(defaults["model_template"])
            model_path = _bootstrap.PROJECT_ROOT / template.format(model=model_name)
        _evaluate_one(
            experiment=args.experiment,
            model_name=model_name,
            data_path=data_path,
            model_path=model_path,
            bias_correct=bias_correct,
            no_plots=args.no_plots,
            save_plots_dir=save_plots_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

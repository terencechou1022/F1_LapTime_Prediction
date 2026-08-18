"""CLI: evaluate a trained model on a held-out race file."""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from f1lab import ModelEvaluator, Visualizer, get_preprocessor
from f1lab.models import MODEL_NAMES

# All four entries default to bias_correct=False so that the four studies
# behave identically when run without an explicit flag. Bias correction is
# the user's opt-in via --bias-correct; the raw run is the natural baseline.
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

# Residual-vs-feature plots: which feature columns to scatter against, per
# study. Each list contains:
#   - 2 study causal axes (HeadWind/CrossWind or AirTemp/TrackTemp) — for the
#     mechanism-estimation diagnostic
#   - 4 tyre/stint progression features (LapNumber, LapInStint, TyreLife,
#     TyreLifeNorm) — cover both absolute and relative views of race + stint
#     position, useful for locating where in a stint the model fails
# Binary features (FreshTyre, Compound_*) and Rainfall are deliberately
# omitted: scatter against a near-binary axis has low diagnostic value.
_RESIDUAL_FEATURES: dict[str, list[str]] = {
    "wind": ["HeadWind", "CrossWind", "LapNumber", "LapInStint", "TyreLife", "TyreLifeNorm"],
    "temp": ["AirTemp",  "TrackTemp", "LapNumber", "LapInStint", "TyreLife", "TyreLifeNorm"],
}


def _plot_prefix(experiment: str, bias_correct: bool, model_name: str) -> str:
    """Build a filename prefix matching the log-file convention:
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

    # Pick which residual series to plot (raw vs bias-corrected)
    y_pred_for_plots = result.y_pred_corrected if bias_correct else result.y_pred

    if save_plots_dir is not None:
        prefix = _plot_prefix(experiment, bias_correct, model_name)
        out = save_plots_dir
        out.mkdir(parents=True, exist_ok=True)

        # Existing diagnostics (saved instead of shown)
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

        # NEW: residual vs predicted (#1)
        Visualizer.residual_vs_predicted(
            result.y_true, y_pred_for_plots,
            title=f"Residual vs Predicted — {prefix}",
            save_path=out / f"{prefix}_residual_vs_predicted.png",
        )

        # NEW: residual vs feature (#3) — 4 features per study
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

    # Interactive mode (no --save-plots-dir, no --no-plots): show on screen
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

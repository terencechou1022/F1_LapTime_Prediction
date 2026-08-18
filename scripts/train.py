"""CLI: train regression models (DT, RF, XGBoost) for a given experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from f1lab import ModelTrainer, Visualizer, get_preprocessor
from f1lab.models import MODEL_NAMES

_DEFAULTS: dict[str, dict[str, str]] = {
    "wind": {
        "data": "data/merged/2022-2024_Azerbaijan_Grand_Prix.xlsx",
        "model_template": "models/azerbaijan_{model}.joblib",
    },
    "temp": {
        "data": "data/merged/2022-2024_Singapore_Grand_Prix.xlsx",
        "model_template": "models/singapore_{model}.joblib",
    },
}


def _train_one(
    experiment: str,
    model_name: str,
    data_path: Path,
    model_path: Path,
    save: bool,
    no_plots: bool,
    save_plots_dir: Path | None,
    quick: bool,
) -> None:
    print(f"\n=== Training {model_name.upper()} on '{experiment}' ===")
    preprocessor_cls = get_preprocessor(experiment)
    preprocessor = preprocessor_cls.from_excel(data_path)

    trainer = ModelTrainer(preprocessor, model_name=model_name, quick=quick)
    trainer.fit()
    trainer.report()

    if not no_plots:
        y_pred = trainer.predict_valid()
        if save_plots_dir is not None:
            prefix = f"train_{experiment}_{model_name}"
            Visualizer.feature_importance(
                trainer.model, list(trainer.x_train.columns),
                save_path=save_plots_dir / f"{prefix}_feature_importance.png",
            )
            Visualizer.prediction_vs_actual(
                trainer.y_valid, y_pred,
                title=f"Predicted vs Actual (validation set) — {prefix}",
                save_path=save_plots_dir / f"{prefix}_prediction_vs_actual.png",
            )
            Visualizer.residual_distribution(
                trainer.y_valid, y_pred,
                title=f"Residual Distribution (validation set) — {prefix}",
                save_path=save_plots_dir / f"{prefix}_residual_distribution.png",
            )
            print(f"  plots saved to {save_plots_dir}/")
        else:
            # Interactive mode (no --save-plots-dir, no --no-plots): show on screen
            Visualizer.feature_importance(trainer.model, list(trainer.x_train.columns))
            Visualizer.prediction_vs_actual(trainer.y_valid, y_pred)
            Visualizer.residual_distribution(trainer.y_valid, y_pred)

    if save:
        trainer.save(model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train regression model(s) for an experiment.")
    parser.add_argument("--experiment", required=True, choices=sorted(_DEFAULTS.keys()))
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "all"],
        default="all",
        help="Model class to train. 'all' runs DT, RF, and XGBoost sequentially.",
    )
    parser.add_argument("--data", type=Path, help="Override training data path.")
    parser.add_argument(
        "--model-out",
        type=Path,
        help="Override model output path. Only valid with a single --model (not 'all').",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save", action="store_true", help="Persist trained model(s).")
    save_group.add_argument(
        "--quick", action="store_true",
        help="Tiny grid smoke run, ~2 min total; for CI and first-time visitors. "
             "Mutually exclusive with --save so a smoke run can never overwrite real models.",
    )

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--no-plots", action="store_true",
        help="Skip all plotting (fastest; for headless metric-only runs).",
    )
    plot_group.add_argument(
        "--save-plots-dir", type=Path,
        help="Save training diagnostic plots (feature importance, prediction-vs-actual, "
             "residual distribution — all on the validation set) to this directory instead "
             "of showing them. Filenames follow train_{study}_{model}_<type>.png.",
    )
    args = parser.parse_args()

    defaults = _DEFAULTS[args.experiment]
    data_path = args.data or _bootstrap.PROJECT_ROOT / defaults["data"]

    models_to_train = list(MODEL_NAMES) if args.model == "all" else [args.model]

    if args.model_out is not None and args.model == "all":
        parser.error("--model-out cannot be used with --model all (path is per-model).")

    save_plots_dir: Path | None = None
    if args.save_plots_dir is not None:
        save_plots_dir = args.save_plots_dir
        if not save_plots_dir.is_absolute():
            save_plots_dir = _bootstrap.PROJECT_ROOT / save_plots_dir
        save_plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models_to_train:
        if args.model_out is not None:
            model_path = args.model_out
        else:
            model_path = _bootstrap.PROJECT_ROOT / defaults["model_template"].format(model=model_name)
        _train_one(
            experiment=args.experiment,
            model_name=model_name,
            data_path=data_path,
            model_path=model_path,
            save=args.save,
            no_plots=args.no_plots,
            save_plots_dir=save_plots_dir,
            quick=args.quick,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

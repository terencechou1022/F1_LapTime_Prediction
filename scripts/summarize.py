"""Summarize pipeline output logs into structured CSVs.

Reads `logs/*.log` produced by `main.py` (30 files when complete) and writes:
    summary/metrics.csv      — 24 rows: eval results per (study, domain, mode, model)
    summary/best_params.csv  — 6 rows: per (study, model) best_params + holdout metrics

Missing logs produce warnings but do not crash. Idempotent: re-running overwrites
the output CSVs in place. Designed to be run AFTER `main.py` completes; can also
be re-run after partial retraining to refresh whatever metrics are present.

Usage:
    python scripts/summarize.py                          # default paths
    python scripts/summarize.py --logs-dir custom_logs   # override
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import _bootstrap  # noqa: F401

STUDIES = ("wind", "temp")
MODELS = ("dt", "rf", "xgb")
DOMAINS = ("indomain", "crossdomain")
MODES = ("raw", "bias")

# A metric line looks like e.g.:
#   [holdout] MAE:  0.452
#   [raw] R2:   -0.238
#   [bias-corrected] RMSE: 0.601
_METRIC_RE = re.compile(r"\[(raw|holdout|bias-corrected)\]\s+(MAE|MSE|RMSE|R2):\s+(-?\d+\.\d+)")
_BIAS_OFFSET_RE = re.compile(r"Median residual offset:\s+(-?\d+\.\d+)")
_BEST_PARAMS_RE = re.compile(r"Best params:\s+(\{.+\})")


def _parse_metrics(log_path: Path, tag: str) -> dict[str, float] | None:
    """Return {mae, mse, rmse, r2} for the requested tag, or None if missing."""
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, float] = {}
    for m in _METRIC_RE.finditer(txt):
        if m.group(1) == tag:
            metrics[m.group(2).lower()] = float(m.group(3))
    return metrics if len(metrics) == 4 else None


def _parse_bias_offset(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    m = _BIAS_OFFSET_RE.search(log_path.read_text(encoding="utf-8", errors="replace"))
    return float(m.group(1)) if m else None


def _parse_best_params(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    m = _BEST_PARAMS_RE.search(log_path.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else None


def _build_metrics_rows(logs_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for study in STUDIES:
        for domain in DOMAINS:
            for mode in MODES:
                for model in MODELS:
                    log_path = logs_dir / f"eval_{study}_{domain}_{mode}_{model}.log"
                    # raw-mode log → take [raw] block;  bias-mode log → take [bias-corrected]
                    tag = "raw" if mode == "raw" else "bias-corrected"
                    metrics = _parse_metrics(log_path, tag)
                    if metrics is None:
                        missing.append(log_path.name)
                        continue
                    rows.append({
                        "study": study,
                        "domain": domain,
                        "mode": mode,
                        "model": model,
                        "mae": metrics["mae"],
                        "mse": metrics["mse"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "bias_offset": _parse_bias_offset(log_path) if mode == "bias" else "",
                    })
    return rows, missing


def _build_params_rows(logs_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for study in STUDIES:
        for model in MODELS:
            log_path = logs_dir / f"train_{study}_{model}.log"
            holdout = _parse_metrics(log_path, "holdout")
            params = _parse_best_params(log_path)
            if holdout is None and params is None:
                missing.append(log_path.name)
                continue
            rows.append({
                "study": study,
                "model": model,
                "holdout_mae": holdout["mae"] if holdout else "",
                "holdout_mse": holdout["mse"] if holdout else "",
                "holdout_rmse": holdout["rmse"] if holdout else "",
                "holdout_r2": holdout["r2"] if holdout else "",
                "best_params": params if params else "",
            })
    return rows, missing


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize pipeline logs into CSVs.")
    parser.add_argument(
        "--logs-dir", type=Path,
        default=_bootstrap.PROJECT_ROOT / "logs",
        help="Directory containing the .log files (default: logs/).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_bootstrap.PROJECT_ROOT / "summary",
        help="Directory to write CSVs into (default: summary/).",
    )
    args = parser.parse_args()

    # metrics.csv — 24 expected rows
    metrics_rows, missing_eval = _build_metrics_rows(args.logs_dir)
    metrics_path = args.out_dir / "metrics.csv"
    _write_csv(
        metrics_path,
        ["study", "domain", "mode", "model", "mae", "mse", "rmse", "r2", "bias_offset"],
        metrics_rows,
    )
    print(f"Wrote {metrics_path} ({len(metrics_rows)}/24 rows)")
    if missing_eval:
        print(f"  WARN: missing eval logs ({len(missing_eval)}): {missing_eval}")

    # best_params.csv — 6 expected rows
    params_rows, missing_train = _build_params_rows(args.logs_dir)
    params_path = args.out_dir / "best_params.csv"
    _write_csv(
        params_path,
        ["study", "model", "holdout_mae", "holdout_mse", "holdout_rmse", "holdout_r2", "best_params"],
        params_rows,
    )
    print(f"Wrote {params_path} ({len(params_rows)}/6 rows)")
    if missing_train:
        print(f"  WARN: missing train logs ({len(missing_train)}): {missing_train}")

    return 0 if not (missing_eval or missing_train) else 1


if __name__ == "__main__":
    raise SystemExit(main())

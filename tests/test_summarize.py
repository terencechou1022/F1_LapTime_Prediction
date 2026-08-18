"""Tests for scripts/summarize.py log parsing (canned logs in the real format)."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import summarize

# Canned logs matching the real Metrics.report / ModelEvaluator output format.
RAW_LOG = """\
[raw] MAE:  0.457
[raw] MSE:  0.383
[raw] RMSE: 0.619
[raw] R2:   0.209
"""

BIAS_LOG = """\
[raw] MAE:  1.516
[raw] MSE:  3.211
[raw] RMSE: 1.792
[raw] R2:   -6.080
Median residual offset: -1.32 s
[bias-corrected] MAE:  0.817
[bias-corrected] MSE:  1.123
[bias-corrected] RMSE: 1.060
[bias-corrected] R2:   -1.477
"""

TRAIN_LOG = """\
Model: rf
Best params: {'max_depth': None, 'n_estimators': 1500}
[holdout] MAE:  0.562
[holdout] MSE:  0.759
[holdout] RMSE: 0.871
[holdout] R2:   0.734
"""


def test_parse_metrics_by_tag(tmp_path):
    log = tmp_path / "eval_wind_indomain_raw_rf.log"
    log.write_text(RAW_LOG, encoding="utf-8")

    assert summarize._parse_metrics(log, "raw") == {
        "mae": 0.457, "mse": 0.383, "rmse": 0.619, "r2": 0.209,
    }
    # Wrong tag / missing file → None.
    assert summarize._parse_metrics(log, "bias-corrected") is None
    assert summarize._parse_metrics(tmp_path / "nope.log", "raw") is None


def test_build_metrics_rows(tmp_path):
    (tmp_path / "eval_wind_indomain_raw_rf.log").write_text(RAW_LOG, encoding="utf-8")
    (tmp_path / "eval_temp_crossdomain_bias_rf.log").write_text(BIAS_LOG, encoding="utf-8")

    rows, missing = summarize._build_metrics_rows(tmp_path)

    assert len(rows) == 2
    assert len(missing) == 22
    assert rows[0] == {
        "study": "wind", "domain": "indomain", "mode": "raw", "model": "rf",
        "mae": 0.457, "mse": 0.383, "rmse": 0.619, "r2": 0.209,
        "bias_offset": "",
    }
    # Bias-mode row takes the [bias-corrected] block and carries the offset.
    assert rows[1] == {
        "study": "temp", "domain": "crossdomain", "mode": "bias", "model": "rf",
        "mae": 0.817, "mse": 1.123, "rmse": 1.060, "r2": -1.477,
        "bias_offset": -1.32,
    }


def test_build_params_rows(tmp_path):
    (tmp_path / "train_wind_rf.log").write_text(TRAIN_LOG, encoding="utf-8")

    rows, missing = summarize._build_params_rows(tmp_path)

    assert len(rows) == 1
    assert len(missing) == 5
    assert rows[0] == {
        "study": "wind", "model": "rf",
        "holdout_mae": 0.562, "holdout_mse": 0.759,
        "holdout_rmse": 0.871, "holdout_r2": 0.734,
        "best_params": "{'max_depth': None, 'n_estimators': 1500}",
    }

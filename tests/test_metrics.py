"""Tests for the Metrics dataclass against hand-computed values."""
from __future__ import annotations

import numpy as np
import pytest

from f1lab.modeling import Metrics


def test_compute_matches_manual_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5])
    # errors = [0.5, 0.5, -0.5, 0.5] → MAE 0.5, MSE 0.25, RMSE 0.5
    # SST = 5.0 (mean 2.5) → R² = 1 − 1.0/5.0 = 0.8
    m = Metrics.compute(y_true, y_pred)

    assert m.mae == pytest.approx(0.5)
    assert m.mse == pytest.approx(0.25)
    assert m.rmse == pytest.approx(0.5)
    assert m.r2 == pytest.approx(0.8)


def test_rmse_is_sqrt_of_mse():
    y_true = np.array([0.0, 1.0, 5.0, 9.0, 10.0])
    y_pred = np.array([0.5, 0.0, 6.0, 9.2, 12.0])
    m = Metrics.compute(y_true, y_pred)
    assert m.rmse == pytest.approx(np.sqrt(m.mse))


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = Metrics.compute(y, y.copy())
    assert (m.mae, m.mse, m.rmse, m.r2) == (0.0, 0.0, 0.0, 1.0)

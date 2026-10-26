"""MonteCarloUndercut 的單元測試（隨機抽樣、勝率分佈、確定性退化、OOD 拒絕）。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from f1lab.strategy import MonteCarloConfig, MonteCarloUndercut, UndercutScenario


@pytest.fixture(scope="module")
def scenario() -> UndercutScenario:
    # 特徵 F 的支撐區間是 [0, 10]；y 隨 F 單調遞增
    x = pd.DataFrame({
        "F": np.linspace(0.0, 10.0, 60),
        "G": np.tile([0.0, 1.0, 2.0], 20),
    })
    y = 2.0 * x["F"] + 0.5 * x["G"]
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(x, y)
    return UndercutScenario(
        model, x, "F",
        pit_loss=20.0,
        gap=19.50,
        ours_new_outlap=95.20,
        rival_old_inlap=95.50,
        n_remaining=10,
    )


def test_monte_carlo_deterministic_degeneration(scenario):
    """當所有 std 設為 0 且無換胎失誤時，蒙特卡羅結果應退化為決定論點估計。"""
    cfg = MonteCarloConfig(
        n_simulations=500,
        pit_loss_std=0.0,
        pit_failure_prob=0.0,
        ours_new_outlap_std=0.0,
        rival_old_inlap_std=0.0,
        gap_std=0.0,
        current_value_std=0.0,
    )

    det_res = scenario.evaluate(8.0)
    mc_res = scenario.simulate_monte_carlo(8.0, config=cfg)

    assert mc_res.in_support and mc_res.applicable
    assert mc_res.ood_rate == 0.0

    # 未修正指標比對
    assert mc_res.mean_margin_uncorrected == pytest.approx(det_res.margin_uncorrected)
    assert mc_res.ci_95_uncorrected[0] == pytest.approx(det_res.margin_uncorrected)
    assert mc_res.ci_95_uncorrected[1] == pytest.approx(det_res.margin_uncorrected)
    assert mc_res.success_prob_uncorrected == (1.0 if det_res.margin_uncorrected > 0 else 0.0)

    # 修正後指標比對
    assert mc_res.mean_margin_corrected == pytest.approx(det_res.margin_corrected)
    assert mc_res.success_prob_corrected == (1.0 if det_res.margin_corrected > 0 else 0.0)


def test_monte_carlo_probabilistic_shift(scenario):
    """在隨機擾動下，環境修正應使平均餘裕增加、勝率上升。"""
    cfg = MonteCarloConfig(n_simulations=5000, random_state=42)
    mc_res = scenario.simulate_monte_carlo(8.0, config=cfg)

    assert mc_res.in_support and mc_res.applicable
    assert 0.0 <= mc_res.success_prob_uncorrected <= 1.0
    assert 0.0 <= mc_res.success_prob_corrected <= 1.0

    # 8.0 高於均值 5.0，故修正後 margin 平均值應顯著大於未修正
    assert mc_res.mean_margin_corrected > mc_res.mean_margin_uncorrected
    assert mc_res.success_prob_corrected > mc_res.success_prob_uncorrected
    assert mc_res.prob_flip is not None and mc_res.prob_flip > 0.0


def test_monte_carlo_ood_rejection(scenario):
    """當前環境數值遠超出訓練支撐區間時，OOD 率應觸發拒絕機制。"""
    cfg = MonteCarloConfig(n_simulations=1000, random_state=42)
    mc_res = scenario.simulate_monte_carlo(-5.0, config=cfg)

    assert mc_res.ood_rate == 1.0
    assert not mc_res.in_support
    assert not mc_res.applicable

    # 修正後項目應安全收回 (None)
    assert mc_res.margins_corrected is None
    assert mc_res.success_prob_corrected is None
    assert mc_res.mean_margin_corrected is None
    assert mc_res.ci_95_corrected is None

    # 未修正的算術仍正常輸出
    assert mc_res.margins_uncorrected is not None
    assert 0.0 <= mc_res.success_prob_uncorrected <= 1.0


def test_monte_carlo_pit_failure_tail_risk(scenario):
    """當加入換胎失誤機率時，平均超車餘裕應向負側漂移（勝率下降）。"""
    cfg_clean = MonteCarloConfig(n_simulations=5000, pit_failure_prob=0.0, random_state=42)
    cfg_risky = MonteCarloConfig(
        n_simulations=5000,
        pit_failure_prob=0.20,
        pit_failure_delay_mean=3.0,
        random_state=42,
    )

    res_clean = scenario.simulate_monte_carlo(8.0, config=cfg_clean)
    res_risky = scenario.simulate_monte_carlo(8.0, config=cfg_risky)

    assert res_risky.mean_margin_uncorrected < res_clean.mean_margin_uncorrected
    assert res_risky.success_prob_uncorrected < res_clean.success_prob_uncorrected
    assert res_risky.mean_margin_corrected < res_clean.mean_margin_corrected


def test_monte_carlo_cache_reproducibility(scenario):
    """從 to_cache / from_cache 重建的情境，蒙特卡羅模擬結果應完全一致。"""
    cache_data = json.loads(json.dumps(scenario.to_cache()))
    rebuilt = UndercutScenario.from_cache(
        cache_data,
        scenario.feature,
        pit_loss=scenario.pit_loss,
        gap=scenario.gap,
        ours_new_outlap=scenario.ours_new_outlap,
        rival_old_inlap=scenario.rival_old_inlap,
        n_remaining=scenario.n_remaining,
    )

    cfg = MonteCarloConfig(n_simulations=2000, random_state=123)
    orig_res = scenario.simulate_monte_carlo(7.5, config=cfg)
    rebuilt_res = rebuilt.simulate_monte_carlo(7.5, config=cfg)

    assert orig_res.success_prob_uncorrected == pytest.approx(rebuilt_res.success_prob_uncorrected)
    assert orig_res.success_prob_corrected == pytest.approx(rebuilt_res.success_prob_corrected)
    assert orig_res.mean_margin_corrected == pytest.approx(rebuilt_res.mean_margin_corrected)
    assert np.allclose(orig_res.margins_uncorrected, rebuilt_res.margins_uncorrected)

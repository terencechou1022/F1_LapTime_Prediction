"""UndercutScenario 的測試：在支撐區間內套用修正，OOD 時收回。"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from f1lab.strategy import UndercutScenario


@pytest.fixture(scope="module")
def scenario() -> UndercutScenario:
    # 特徵 F 的支撐區間是 [0, 10]；y 依賴 F，所以 PDP 不會是平的。
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


def test_support_and_mean(scenario):
    assert scenario.support == (0.0, 10.0)
    assert scenario.training_mean == pytest.approx(5.0)


def test_in_support_correction_applied(scenario):
    res = scenario.evaluate(8.0)

    assert res.in_support and res.applicable
    # 固定參數的算術：net = 20.0 + 95.20 − 95.50 = 19.70。
    assert res.net == pytest.approx(19.70)
    assert res.margin_uncorrected == pytest.approx(-0.20)
    assert res.decision_uncorrected == "CLOSE"
    # 修正相關的欄位都有填值，而且彼此一致。
    assert res.delta_per_lap is not None
    assert res.delta_total == pytest.approx(res.n_remaining * res.delta_per_lap)
    assert res.gap_corrected == pytest.approx(res.gap + res.delta_total)
    assert res.margin_corrected == pytest.approx(res.gap_corrected - res.net)
    assert res.decision_corrected in ("OPEN", "CLOSE")
    assert res.flipped == (res.decision_corrected != res.decision_uncorrected)
    # 8.0 遠高於一條上升反應曲線的平均值 → 修正量為正。
    assert res.delta_per_lap > 0


def test_out_of_support_correction_withheld(scenario):
    res = scenario.evaluate(-5.0)

    assert not res.in_support and not res.applicable
    assert res.delta_per_lap is None
    assert res.delta_total is None
    assert res.gap_corrected is None
    assert res.margin_corrected is None
    assert res.decision_corrected is None
    assert res.flipped is False
    # net 與 gap 是純算術——即使在 OOD 也一定會算出來。
    assert res.net == pytest.approx(19.70)
    assert res.gap == pytest.approx(19.50)
    assert res.margin_uncorrected == pytest.approx(-0.20)
    assert res.decision_uncorrected == "CLOSE"


def test_from_cache_reproduces_the_model_path(scenario):
    """經過 to_cache/from_cache 的 JSON 來回轉換，結果必須完全不變。

    正是這個等價關係，讓只需要修正量的場合可以只帶 PDP 曲線，
    而不必帶著擬合好的森林。
    """
    rebuilt = UndercutScenario.from_cache(
        json.loads(json.dumps(scenario.to_cache())),
        scenario.feature,
        pit_loss=scenario.pit_loss,
        gap=scenario.gap,
        ours_new_outlap=scenario.ours_new_outlap,
        rival_old_inlap=scenario.rival_old_inlap,
        n_remaining=scenario.n_remaining,
    )

    assert rebuilt.support == scenario.support
    assert rebuilt.training_mean == scenario.training_mean
    # 涵蓋支撐區間之下、兩個邊界、區間之內，以及區間之上。
    for value in (-5.0, 0.0, 5.0, 8.0, 10.0, 12.0):
        assert asdict(rebuilt.evaluate(value)) == asdict(scenario.evaluate(value))

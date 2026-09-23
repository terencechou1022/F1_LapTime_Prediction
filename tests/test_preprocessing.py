"""BaseLapPreprocessor 管線的測試（過濾、stint 特徵、預測目標、one-hot 對齊）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import LAPS_PER_STINT

from f1lab.experiments import TempPreprocessor, WindPreprocessor


def _laps(df: pd.DataFrame, driver: str) -> set[int]:
    return set(df.loc[df["Driver"] == driver, "LapNumber"])


def test_invalid_lap_filter(laps_df):
    df, _, _ = TempPreprocessor(laps_df).run()

    ver = _laps(df, "VER")
    ham = _laps(df, "HAM")

    # 進站圈（15）、出站圈（16），以及出站圈的下一圈（17）都不見了。
    assert not {15, 16, 17} & ver
    assert not {15, 16, 17} & ham
    # 非全綠旗的圈（VER 第 5 圈）與緊接其後那一圈（6）都不見了。
    assert not {5, 6} & ver
    # 無效圈之後第二圈存活；HAM 全綠旗的第 5 圈也存活。
    assert 7 in ver
    assert {5, 6} <= ham


def test_lap_in_stint_computed_pre_filter(laps_df):
    df, _, _ = TempPreprocessor(laps_df).run()

    # 存活下來的圈保住它原本在 stint 內的位置（容許中間有缺號）。
    expected = df["LapNumber"] - (df["Stint"] - 1) * LAPS_PER_STINT
    assert (df["LapInStint"] == expected).all()

    ver_s1 = set(df.loc[(df["Driver"] == "VER") & (df["Stint"] == 1), "LapInStint"])
    assert not {5, 6} & ver_s1  # dropped laps leave gaps
    assert 7 in ver_s1          # ...but positions after the gap are unchanged


def test_target_is_delta_to_stint_best(laps_df):
    df, _, y = TempPreprocessor(laps_df).run()

    assert (y >= 0).all()
    seconds = df["LapTime"].dt.total_seconds()
    for _, group in df.groupby(["Driver", "Stint"]):
        grp_sec = seconds.loc[group.index]
        expected = grp_sec - grp_sec.min()
        assert np.allclose(group["LapTimeDelta"], expected)
        assert (group["LapTimeDelta"] == 0).any()  # stint-best lap has delta 0

    # 已知值：VER 的 stint 2 存活圈從第 18 圈開始（16、17 被過濾掉），
    # 所以該 stint 的最小值就是第 18 圈本身，而第 19 圈比它高 0.4 秒。
    ver_s2 = df[(df["Driver"] == "VER") & (df["Stint"] == 2)].set_index("LapNumber")
    assert ver_s2.loc[18, "LapTimeDelta"] == 0.0
    assert np.isclose(ver_s2.loc[19, "LapTimeDelta"], 0.4)


def test_one_hot_alignment_fills_missing_compound(laps_df):
    _, x_ref, _ = WindPreprocessor(laps_df).run()
    assert "Compound_INTERMEDIATE" not in x_ref.columns

    features = [*x_ref.columns, "Compound_INTERMEDIATE"]
    _, x, _ = WindPreprocessor(laps_df, features=features).run()

    assert list(x.columns) == features
    assert (x["Compound_INTERMEDIATE"] == 0.0).all()

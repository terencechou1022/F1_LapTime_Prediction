"""Tests for the BaseLapPreprocessor pipeline (filter, stint features, target, one-hot alignment)."""
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

    # Pit-in lap (15), pit-out lap (16), and the lap after the pit-out (17) are gone.
    assert not {15, 16, 17} & ver
    assert not {15, 16, 17} & ham
    # Non-green lap (VER 5) and the lap immediately after it (6) are gone.
    assert not {5, 6} & ver
    # The second lap after an invalid lap survives; HAM's green lap 5 survives.
    assert 7 in ver
    assert {5, 6} <= ham


def test_lap_in_stint_computed_pre_filter(laps_df):
    df, _, _ = TempPreprocessor(laps_df).run()

    # Surviving laps keep their original in-stint position (gaps allowed).
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

    # Known value: VER stint 2 survivors start at lap 18 (16/17 filtered), so
    # the stint minimum is lap 18 itself and lap 19 sits 0.4 s above it.
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

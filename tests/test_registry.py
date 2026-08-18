"""Tests for the preprocessor self-registry and the symmetric feature design."""
from __future__ import annotations

import pandas as pd
import pytest

from f1lab.experiments import TempPreprocessor, WindPreprocessor
from f1lab.preprocessing import BaseLapPreprocessor


def test_get_returns_registered_subclasses():
    assert BaseLapPreprocessor.get("wind") is WindPreprocessor
    assert BaseLapPreprocessor.get("temp") is TempPreprocessor


def test_get_unknown_name_raises_with_valid_options():
    with pytest.raises(ValueError, match=r"Unknown experiment 'nope'.*temp.*wind"):
        BaseLapPreprocessor.get("nope")


def test_both_studies_expose_11_features():
    empty = pd.DataFrame()
    wind_cols = WindPreprocessor(empty).feature_columns
    temp_cols = TempPreprocessor(empty).feature_columns

    assert len(wind_cols) == 11
    assert len(temp_cols) == 11
    assert set(wind_cols) - set(BaseLapPreprocessor.BASE_FEATURES) == {"HeadWind", "CrossWind"}
    assert set(temp_cols) - set(BaseLapPreprocessor.BASE_FEATURES) == {"AirTemp", "TrackTemp"}

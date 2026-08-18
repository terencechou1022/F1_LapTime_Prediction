"""Experiment-specific preprocessor subclasses.

Each subclass declares its `experiment_name` (which auto-registers it on
`BaseLapPreprocessor`) and extends `BASE_FEATURES` with the study's own
explanatory variables:

    Temp study  ← AirTemp, TrackTemp  (variables under investigation)
    Wind study  ← HeadWind, CrossWind (variables under investigation)

The shared `BASE_FEATURES` (LapNumber, LapInStint, Compound, TyreLife,
TyreLifeNorm, FreshTyre, FuelLoad, Humidity, Rainfall) act as control
variables in both studies.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from f1lab.preprocessing import BaseLapPreprocessor


class TempPreprocessor(BaseLapPreprocessor):
    """Temp study (temperature-driven tyre degradation): AirTemp / TrackTemp as explanatory."""

    experiment_name: ClassVar[str] = "temp"

    @property
    def feature_columns(self) -> list[str]:
        return [*self.BASE_FEATURES, "AirTemp", "TrackTemp"]


class WindPreprocessor(BaseLapPreprocessor):
    """Wind study (wind-driven performance loss): HeadWind / CrossWind as explanatory."""

    experiment_name: ClassVar[str] = "wind"

    @property
    def feature_columns(self) -> list[str]:
        return [*self.BASE_FEATURES, "HeadWind", "CrossWind"]

    def _engineer_specific_features(self) -> None:
        df = self.df
        radians = np.radians(df["WindDirection"])
        df["HeadWind"] = df["WindSpeed"] * np.cos(radians)
        df["CrossWind"] = df["WindSpeed"] * np.sin(radians)
        self.df = df


def get_preprocessor(name: str) -> type[BaseLapPreprocessor]:
    """Backwards-compatible facade for `BaseLapPreprocessor.get`."""
    return BaseLapPreprocessor.get(name)

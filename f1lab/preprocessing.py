"""Lap-data preprocessing pipeline (Template Method pattern).

`BaseLapPreprocessor` defines the shared sequence:
    sort → engineer common features → filter invalid laps
        → engineer experiment-specific features → compute target
        → encode compound → align feature matrix

Stint-position features (LapInStint, TyreLifeNorm) are computed BEFORE
the invalid-lap filter, so their values reflect each lap's true position
in the original stint regardless of how many SC/pit laps were dropped.

Subclasses override `feature_columns` and (optionally) `_engineer_specific_features`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd


class BaseLapPreprocessor(ABC):
    """Abstract base preprocessor for race lap regression.

    Subclasses self-register via the `experiment_name` class attribute and
    can be looked up by `BaseLapPreprocessor.get(name)`.
    """

    GROUP_KEYS: ClassVar[tuple[str, ...]] = ("RaceYear", "GPName", "Driver", "Stint")
    DRIVER_KEYS: ClassVar[tuple[str, ...]] = ("RaceYear", "GPName", "Driver")
    SORT_KEYS: ClassVar[tuple[str, ...]] = ("RaceYear", "GPName", "Driver", "LapNumber")
    TARGET_COLUMN: ClassVar[str] = "LapTimeDelta"

    # Baseline features shared by every study. Subclasses extend this list with
    # study-specific explanatory variables (e.g. AirTemp/TrackTemp for the temp
    # study, HeadWind/CrossWind for the wind study). Anything in this list is a
    # control variable — not the phenomenon under investigation.
    #
    # Tyre-related axes: TyreLife (raw absolute use), TyreLifeNorm (stint-relative
    # position 0~1), LapInStint (in-stint count). The three carry complementary
    # information: raw age captures absolute wear (and carries across stints for
    # inherited tyres), the normalised forms capture within-stint progression.
    # The earlier `TyreLifeTemp = TyreLifeNorm × TrackTemp` interaction was
    # removed: tree-based models can learn the interaction implicitly from main
    # effects, and including it would back-door TrackTemp into the wind study.
    BASE_FEATURES: ClassVar[list[str]] = [
        "LapNumber",
        "LapInStint",
        "Compound",
        "TyreLife",
        "TyreLifeNorm",
        "FreshTyre",
        "FuelLoad",
        "Humidity",
        "Rainfall",
    ]

    # Fuel mass burns at ~1.7 kg/lap from a 110 kg start (Cappello, 2025)
    FUEL_START_KG: ClassVar[float] = 110.0
    FUEL_BURN_KG_PER_LAP: ClassVar[float] = 1.7

    # Subclass registry (auto-populated via __init_subclass__).
    experiment_name: ClassVar[str] = ""
    _registry: ClassVar[dict[str, type["BaseLapPreprocessor"]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.experiment_name:
            BaseLapPreprocessor._registry[cls.experiment_name] = cls

    def __init__(self, df: pd.DataFrame, features: Sequence[str] | None = None) -> None:
        self.df: pd.DataFrame = df.copy()
        self.features: list[str] | None = list(features) if features is not None else None

    # ---- factories --------------------------------------------------------

    @classmethod
    def from_excel(cls, path: str | Path, features: Sequence[str] | None = None) -> "BaseLapPreprocessor":
        return cls(pd.read_excel(path), features=features)

    @classmethod
    def get(cls, name: str) -> type["BaseLapPreprocessor"]:
        """Look up a registered preprocessor subclass by experiment name."""
        try:
            return cls._registry[name]
        except KeyError as exc:
            valid = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown experiment {name!r}. Valid options: {valid}"
            ) from exc

    # ---- public pipeline (template method) --------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Run the full preprocessing pipeline and return (df, X, y).

        Order rationale: stint-position features (LapInStint, TyreLifeNorm) are
        computed BEFORE invalid-lap filtering so that they reflect the lap's
        true position in the original stint, not the post-filter visible
        position. This keeps `LapInStint` consistent across runs regardless
        of how many SC/pit laps were removed in a given stint.
        """
        self._sort()
        self._engineer_common_features()
        self._filter_invalid_laps()
        self._engineer_specific_features()
        self._compute_target()

        x = self._build_feature_matrix()
        y = self.df[self.TARGET_COLUMN]
        return self.df, x, y

    # ---- abstract / hook methods ------------------------------------------

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Columns selected from the dataframe before one-hot encoding."""

    def _engineer_specific_features(self) -> None:
        """Override to add experiment-specific features (no-op by default)."""

    # ---- shared steps -----------------------------------------------------

    def _sort(self) -> None:
        self.df = self.df.sort_values(list(self.SORT_KEYS))

    def _filter_invalid_laps(self) -> None:
        # Scoped to this method to avoid the global side effect of
        # `pd.set_option` at module import. Both `.replace("", np.nan)` and
        # `.fillna(False)` below would otherwise emit a FutureWarning about
        # silent downcasting of object dtypes.
        with pd.option_context("future.no_silent_downcasting", True):
            df = self.df
            df[["PitInTime", "PitOutTime"]] = df[["PitInTime", "PitOutTime"]].replace("", np.nan)
            is_pit = df["PitInTime"].notna() | df["PitOutTime"].notna()
            not_green = df["TrackStatus"].astype(str) != "1"
            invalid = is_pit | not_green
            prev_invalid = (
                invalid.groupby([df[k] for k in self.DRIVER_KEYS])
                .shift(1)
                .fillna(False)
                .astype(bool)
            )
            self.df = df.loc[~(invalid | prev_invalid)].copy()

    def _engineer_common_features(self) -> None:
        df = self.df
        groups = df.groupby(list(self.GROUP_KEYS))

        df["LapInStint"] = groups.cumcount() + 1
        df["TyreLifeNorm"] = df["TyreLife"] / groups["TyreLife"].transform("max")
        df = df[df["TyreLifeNorm"] > 0].copy()  # explicit copy so subsequent assignment doesn't trigger SettingWithCopyWarning
        df["FuelLoad"] = self.FUEL_START_KG - (df["LapNumber"] * self.FUEL_BURN_KG_PER_LAP)
        self.df = df

    def _compute_target(self) -> None:
        df = self.df
        df["LapTime"] = pd.to_timedelta(df["LapTime"], errors="coerce")
        stint_min = df.groupby(list(self.GROUP_KEYS))["LapTime"].transform("min")
        df[self.TARGET_COLUMN] = (df["LapTime"] - stint_min).dt.total_seconds()
        self.df = df.dropna(subset=[self.TARGET_COLUMN])

    def _build_feature_matrix(self) -> pd.DataFrame:
        x = self.df[self.feature_columns].copy()
        if "Compound" in x.columns:
            x = pd.get_dummies(x, columns=["Compound"])

        if self.features is not None:
            for col in self.features:
                if col not in x.columns:
                    x[col] = 0.0
            x = x[self.features]

        return x.astype(float)

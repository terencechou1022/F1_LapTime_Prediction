"""各實驗專屬的前處理器子類別。

每個子類別宣告自己的 `experiment_name`（會自動註冊到 `BaseLapPreprocessor`），
並在 `BASE_FEATURES` 之外補上該研究自己的解釋變數：

    Temp 研究  ← AirTemp、TrackTemp  （受檢驗的變數）
    Wind 研究  ← HeadWind、CrossWind （受檢驗的變數）

共用的 `BASE_FEATURES`（LapNumber、LapInStint、Compound、TyreLife、
TyreLifeNorm、FreshTyre、FuelLoad、Humidity、Rainfall）在兩項研究裡
都扮演控制變數。
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from f1lab.preprocessing import BaseLapPreprocessor


class TempPreprocessor(BaseLapPreprocessor):
    """Temp 研究（溫度驅動的輪胎衰退）：以 AirTemp / TrackTemp 為解釋變數。"""

    experiment_name: ClassVar[str] = "temp"

    @property
    def feature_columns(self) -> list[str]:
        return [*self.BASE_FEATURES, "AirTemp", "TrackTemp"]


class WindPreprocessor(BaseLapPreprocessor):
    """Wind 研究（風造成的表現損失）：以 HeadWind / CrossWind 為解釋變數。"""

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
    """`BaseLapPreprocessor.get` 的相容性門面。"""
    return BaseLapPreprocessor.get(name)

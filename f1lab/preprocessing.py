"""圈速資料的前處理管線（Template Method 模式）。

`BaseLapPreprocessor` 定義共用的順序：
    排序 → 共用特徵工程 → 過濾無效圈
        → 各實驗專屬的特徵工程 → 計算目標
        → 編碼 compound → 對齊特徵矩陣

stint 位置相關的特徵（LapInStint、TyreLifeNorm）在無效圈過濾「之前」就算好，
所以它們的值反映的是每一圈在原始 stint 裡的真實位置，
與該 stint 被丟掉多少安全車圈或進站圈無關。

子類別覆寫 `feature_columns`，並可選擇覆寫 `_engineer_specific_features`。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd


class BaseLapPreprocessor(ABC):
    """賽事圈速迴歸的抽象前處理基底類別。

    子類別透過 `experiment_name` 類別屬性自動註冊，
    之後可用 `BaseLapPreprocessor.get(name)` 取回。
    """

    GROUP_KEYS: ClassVar[tuple[str, ...]] = ("RaceYear", "GPName", "Driver", "Stint")
    DRIVER_KEYS: ClassVar[tuple[str, ...]] = ("RaceYear", "GPName", "Driver")
    SORT_KEYS: ClassVar[tuple[str, ...]] = ("RaceYear", "GPName", "Driver", "LapNumber")
    TARGET_COLUMN: ClassVar[str] = "LapTimeDelta"

    # 每項研究共用的基線特徵。子類別在這份清單之外補上各自的解釋變數
    # （例如 temp 研究的 AirTemp/TrackTemp、wind 研究的 HeadWind/CrossWind）。
    # 在這份清單裡的一律是控制變數，不是被檢驗的那個現象。
    #
    # 輪胎相關的三個軸：TyreLife(原始絕對使用量)、TyreLifeNorm(stint 內的
    # 相對位置 0~1)、LapInStint(stint 內的計數)。三者帶的是互補資訊：
    # 原始圈齡捕捉絕對磨耗(沿用輪胎時會跨 stint 累加),正規化的兩個
    # 則捕捉 stint 內的進程。
    # 早期的 `TyreLifeTemp = TyreLifeNorm × TrackTemp` 交互作用已移除：
    # 樹模型可以從主效應隱式學到這個交互作用，而把它放進來會從後門
    # 讓 TrackTemp 進到 wind 研究裡。
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

    # 油量從 110 kg 起跑，以每圈約 1.7 kg 消耗(Cappello, 2025)
    FUEL_START_KG: ClassVar[float] = 110.0
    FUEL_BURN_KG_PER_LAP: ClassVar[float] = 1.7

    # 子類別註冊表(由 __init_subclass__ 自動填入)。
    experiment_name: ClassVar[str] = ""
    _registry: ClassVar[dict[str, type["BaseLapPreprocessor"]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.experiment_name:
            BaseLapPreprocessor._registry[cls.experiment_name] = cls

    def __init__(self, df: pd.DataFrame, features: Sequence[str] | None = None) -> None:
        self.df: pd.DataFrame = df.copy()
        self.features: list[str] | None = list(features) if features is not None else None

    # ---- 工廠方法 ----------------------------------------------------------

    @classmethod
    def from_excel(cls, path: str | Path, features: Sequence[str] | None = None) -> "BaseLapPreprocessor":
        return cls(pd.read_excel(path), features=features)

    @classmethod
    def get(cls, name: str) -> type["BaseLapPreprocessor"]:
        """以實驗名稱查出已註冊的前處理器子類別。"""
        try:
            return cls._registry[name]
        except KeyError as exc:
            valid = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown experiment {name!r}. Valid options: {valid}"
            ) from exc

    # ---- 公開管線(template method)-----------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """跑完整條前處理管線，回傳 (df, X, y)。

        順序的理由：stint 位置相關的特徵(LapInStint、TyreLifeNorm)在無效圈
        過濾「之前」就算好，這樣它們反映的是該圈在原始 stint 裡的真實位置，
        而不是過濾後看得到的位置。如此一來，不論某個 stint 被移掉多少
        安全車圈或進站圈，`LapInStint` 在不同次執行之間都保持一致。
        """
        self._sort()
        self._engineer_common_features()
        self._filter_invalid_laps()
        self._engineer_specific_features()
        self._compute_target()

        x = self._build_feature_matrix()
        y = self.df[self.TARGET_COLUMN]
        return self.df, x, y

    # ---- 抽象方法與 hook ---------------------------------------------------

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """one-hot 編碼之前，要從 dataframe 選出來的欄位。"""

    def _engineer_specific_features(self) -> None:
        """覆寫它來加入各實驗專屬的特徵(預設不做任何事)。"""

    # ---- 共用步驟 ----------------------------------------------------------

    def _sort(self) -> None:
        self.df = self.df.sort_values(list(self.SORT_KEYS))

    def _filter_invalid_laps(self) -> None:
        # 限定在這個方法內，避免在模組匯入時呼叫 `pd.set_option` 造成全域副作用。
        # 否則底下的 `.replace("", np.nan)` 與 `.fillna(False)` 都會噴出
        # 關於 object dtype 被靜默降型的 FutureWarning。
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
        df = df[df["TyreLifeNorm"] > 0].copy()  # 明確複製一份，後續賦值才不會觸發 SettingWithCopyWarning
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

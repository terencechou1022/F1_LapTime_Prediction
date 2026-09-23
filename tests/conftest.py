"""f1lab 測試套件的共用 fixture。

把專案根目錄加進 sys.path（與 scripts/_bootstrap.py 一致），
讓 `f1lab` 不論 pytest 以哪種方式啟動都找得到。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

LAPS_PER_STINT = 15


def _driver_laps(driver: str, compounds: tuple[str, str]) -> list[dict]:
    """為單一車手造出兩個各 LAPS_PER_STINT 圈的 stint。

    圈速從 90.0 秒起跳，每在 stint 內多跑一圈就加 0.4 秒。第 15 圈是進站圈
    （有設 PitInTime），第 16 圈是出站圈（有設 PitOutTime）。
    """
    rows = []
    for stint, compound in enumerate(compounds, start=1):
        for lap_in_stint in range(1, LAPS_PER_STINT + 1):
            lap_number = (stint - 1) * LAPS_PER_STINT + lap_in_stint
            sec = 90.0 + 0.4 * (lap_in_stint - 1)
            rows.append({
                "RaceYear": 2023,
                "GPName": "Testville Grand Prix",
                "Driver": driver,
                "LapNumber": lap_number,
                "Stint": stint,
                "LapTime": f"0 days 00:01:{sec - 60:09.6f}",
                "PitInTime": "0 days 00:25:00" if lap_number == LAPS_PER_STINT else "",
                "PitOutTime": "0 days 00:25:20" if lap_number == LAPS_PER_STINT + 1 else "",
                "TrackStatus": "1",
                "Compound": compound,
                "TyreLife": lap_in_stint,
                "FreshTyre": True,
                "WindSpeed": 2.0,
                "WindDirection": 45.0,
                "AirTemp": 28.0,
                "TrackTemp": 35.0,
                "Humidity": 55.0,
                "Rainfall": False,
            })
    return rows


@pytest.fixture
def laps_df() -> pd.DataFrame:
    """合成的一場比賽：2 位車手 × 2 個 stint × 15 圈 = 60 列。

    無效圈：兩位車手都在第 15 圈進站、第 16 圈出站；VER 的第 5 圈跑在
    TrackStatus '4'（安全車）之下。過濾（無效圈本身 + 緊接其後那一圈）之後
    預期存活的情況：VER 的 stint 1 丟掉第 5、6、15 圈；stint 2 丟掉第 16、17 圈；
    HAM 的 stint 1 只丟掉第 15 圈。
    """
    rows = _driver_laps("VER", ("SOFT", "HARD")) + _driver_laps("HAM", ("MEDIUM", "HARD"))
    df = pd.DataFrame(rows)
    df.loc[(df["Driver"] == "VER") & (df["LapNumber"] == 5), "TrackStatus"] = "4"
    return df

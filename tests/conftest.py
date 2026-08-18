"""Shared fixtures for the f1lab test suite.

Adds the project root to sys.path (mirrors scripts/_bootstrap.py) so `f1lab`
resolves regardless of how pytest is invoked.
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
    """Build two stints of LAPS_PER_STINT laps for one driver.

    Lap times rise 0.4 s per in-stint lap from a 90.0 s base. Lap 15 is the
    pit-in lap (PitInTime set), lap 16 the pit-out lap (PitOutTime set).
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
    """Synthetic race: 2 drivers x 2 stints x 15 laps = 60 rows.

    Invalid laps: both drivers pit on lap 15 (in) / 16 (out); VER lap 5 runs
    under TrackStatus '4' (safety car). Expected survivors after the filter
    (invalid laps + the lap immediately following each): VER stint 1 drops
    laps 5, 6, 15; stint 2 drops 16, 17; HAM stint 1 drops lap 15 only.
    """
    rows = _driver_laps("VER", ("SOFT", "HARD")) + _driver_laps("HAM", ("MEDIUM", "HARD"))
    df = pd.DataFrame(rows)
    df.loc[(df["Driver"] == "VER") & (df["LapNumber"] == 5), "TrackStatus"] = "4"
    return df

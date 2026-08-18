"""Data acquisition and merging.

`FastF1Downloader` wraps the fastf1 API and writes raw Excel files.
`RaceDataMerger` joins per-year laps + weather into one tidy file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import fastf1
import pandas as pd


@dataclass
class FastF1Downloader:
    """Download laps + weather + telemetry for a year range via fastf1."""

    output_root: Path
    api_sleep: float = 3.0
    telemetry_drivers: tuple[str, ...] = ("VER",)
    failures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)

    def download_range(self, start_year: int, end_year: int) -> list[str]:
        """Download every race in [start_year, end_year]. Returns failures."""
        for year in range(start_year, end_year + 1):
            print(f"\n=== Season {year} ===")
            try:
                schedule = fastf1.get_event_schedule(year)
                time.sleep(self.api_sleep)
                races = schedule[schedule["Session5"] == "Race"]
            except Exception as exc:  # noqa: BLE001
                self.failures.append(f"{year} (schedule)")
                print(f"  schedule error: {exc}")
                continue

            for _, row in races.iterrows():
                self._download_event(year, int(row["RoundNumber"]), row["EventName"])

        return self.failures

    def _download_event(self, year: int, round_number: int, event_name: str) -> None:
        gp_name = event_name.replace(" ", "_")
        print(f"  {year} R{round_number}: {gp_name}")
        try:
            session = fastf1.get_session(year, round_number, "R")
            time.sleep(self.api_sleep)
            session.load()
            time.sleep(self.api_sleep)

            self._write_laps(session, year, gp_name)
            self._write_weather(session, year, gp_name)
            for driver in self.telemetry_drivers:
                self._write_telemetry(session, year, gp_name, driver)

            print("    OK")
        except Exception as exc:  # noqa: BLE001
            self.failures.append(f"{year}_{gp_name}")
            print(f"    FAIL: {exc}")
        finally:
            time.sleep(self.api_sleep)

    def _write_laps(self, session, year: int, gp_name: str) -> None:
        laps = session.laps.copy()
        laps["RaceYear"] = year
        laps["GPName"] = gp_name
        out = self.output_root / str(year) / "laps" / f"{year}_{gp_name}_laps.xlsx"
        out.parent.mkdir(parents=True, exist_ok=True)
        laps.to_excel(out, index=False)

    def _write_weather(self, session, year: int, gp_name: str) -> None:
        weather = session.weather_data.copy()
        weather["RaceYear"] = year
        weather["GPName"] = gp_name
        out = self.output_root / str(year) / "weather" / f"{year}_{gp_name}_weather.xlsx"
        out.parent.mkdir(parents=True, exist_ok=True)
        weather.to_excel(out, index=False)

    def _write_telemetry(self, session, year: int, gp_name: str, driver: str) -> None:
        telemetry = session.laps.pick_drivers([driver]).get_telemetry().copy()
        telemetry["RaceYear"] = year
        telemetry["GPName"] = gp_name
        out = self.output_root / str(year) / "telemetry" / driver / f"{year}_{gp_name}_{driver}_telemetry.xlsx"
        out.parent.mkdir(parents=True, exist_ok=True)
        telemetry.to_excel(out, index=False)


class RaceDataMerger:
    """Merge per-year raw laps + weather files into a single tidy Excel."""

    TIME_COLUMNS: tuple[str, ...] = (
        "Time", "LapTime", "PitOutTime", "PitInTime",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    )

    def __init__(self, raw_root: Path, merged_root: Path) -> None:
        self.raw_root = Path(raw_root)
        self.merged_root = Path(merged_root)

    def merge_laps_weather(self, gp_name: str, years: list[int], output_name: str | None = None) -> Path:
        """Merge laps + weather for a single GP across multiple years."""
        self.merged_root.mkdir(parents=True, exist_ok=True)

        laps = self._concat_yearly(years, "laps", gp_name, "laps")
        weather = self._concat_yearly(years, "weather", gp_name, "weather")

        for df in (laps, weather):
            if pd.api.types.is_float_dtype(df["Time"]):
                df["Time"] = pd.to_timedelta(df["Time"], unit="D")
            df.sort_values(["RaceYear", "GPName", "Time"], inplace=True)

        merged = self._asof_join(laps, weather)
        merged = self._stringify_time_columns(merged)

        out_name = output_name or self._default_name(years, gp_name)
        output_path = self.merged_root / out_name
        merged.to_excel(output_path, index=False)
        return output_path

    def merge_telemetry(self, gp_name: str, driver: str, years: list[int]) -> Path:
        self.merged_root.mkdir(parents=True, exist_ok=True)

        telemetry = self._concat_yearly(
            years, f"telemetry/{driver}", gp_name, f"{driver}_telemetry"
        )

        for col in ("SessionTime", "Time"):
            telemetry[col] = pd.to_timedelta(telemetry[col], unit="D")
            telemetry[col] = pd.to_datetime(telemetry[col].dt.total_seconds(), unit="s")
            telemetry[col] = telemetry[col].dt.strftime("%H:%M:%S.%f").str[:-3]

        output_path = self.merged_root / f"{gp_name}_{driver}_telemetry.xlsx"
        telemetry.to_excel(output_path, index=False)
        return output_path

    # ---- helpers ----------------------------------------------------------

    def _concat_yearly(self, years: list[int], subdir: str, gp_name: str, suffix: str) -> pd.DataFrame:
        frames = []
        for year in years:
            path = self.raw_root / str(year) / subdir / f"{year}_{gp_name}_{suffix}.xlsx"
            try:
                frames.append(pd.read_excel(path))
            except FileNotFoundError:
                print(f"  missing: {path}")
        if not frames:
            raise FileNotFoundError(f"No {suffix} files found for {gp_name} in {years}")
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _asof_join(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        groups: list[pd.DataFrame] = []
        for key, lap_group in laps.groupby(["RaceYear", "GPName"]):
            weather_group = weather[(weather["RaceYear"] == key[0]) & (weather["GPName"] == key[1])]
            merged = pd.merge_asof(
                lap_group.sort_values("Time"),
                weather_group.sort_values("Time"),
                on="Time",
                by=["RaceYear", "GPName"],
                direction="backward",
            )
            groups.append(merged)
        return pd.concat(groups, ignore_index=True)

    @classmethod
    def _stringify_time_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        for col in cls.TIME_COLUMNS:
            if col not in df.columns:
                continue
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_timedelta(df[col], unit="D")
            df[col] = df[col].astype(str).str.extract(r"(\d{2}:\d{2}:\d{2}\.\d{3})")[0]
        return df

    @staticmethod
    def _default_name(years: list[int], gp_name: str) -> str:
        if len(years) == 1:
            return f"{years[0]}_{gp_name}.xlsx"
        return f"{min(years)}-{max(years)}_{gp_name}.xlsx"

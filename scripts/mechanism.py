"""CLI: PDP/ICE figures + symmetric undercut-scenario tables.

Loads the two study RF models, writes the 8 PDP/ICE figures + feature
ranges, then runs the two **symmetric** undercut scenarios — identical fixed
parameters (pit_loss / gap / out-lap / in-lap / N), differing only in whether
the queried condition is in-support or OOD:

    wind (Saudi)  : current HeadWind in training support  → correction applies → decision flips
    temp (Las Vegas): current TrackTemp OOD (< training min) → correction withheld → cannot evaluate

Outputs:
    plots/figure_6_{1,2}_pdp_{HeadWind,CrossWind}.png   plots/figure_6_{5,6}_pdp_{AirTemp,TrackTemp}.png
    plots/figure_6_{3,4}_ice_{HeadWind,CrossWind}.png   plots/figure_6_{7,8}_ice_{AirTemp,TrackTemp}.png
    stdout: feature ranges + the two scenario tables

Usage:
    python scripts/mechanism.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — no display, avoids tkinter cleanup errors on exit

import _bootstrap  # noqa: F401
import joblib

from f1lab import TempPreprocessor, UndercutScenario, Visualizer, WindPreprocessor

OUT_DIR = _bootstrap.PROJECT_ROOT / "plots"

# Two studies, fully parallel. Each: model file, training data, and the four
# (feature, pdp_fig_num, ice_fig_num) figure specs.
_STUDIES = {
    "wind": {
        "preprocessor": WindPreprocessor,
        "model": "models/azerbaijan_rf.joblib",
        "data": "data/merged/2022-2024_Azerbaijan_Grand_Prix.xlsx",
        "figures": [("HeadWind", 1, 3), ("CrossWind", 2, 4)],
    },
    "temp": {
        "preprocessor": TempPreprocessor,
        "model": "models/singapore_rf.joblib",
        "data": "data/merged/2022-2024_Singapore_Grand_Prix.xlsx",
        "figures": [("AirTemp", 5, 7), ("TrackTemp", 6, 8)],
    },
}

# Shared undercut parameters — IDENTICAL across both studies (the symmetry).
# pit_loss/gap/out-lap/in-lap are strategy-desk parameters (not telemetry-measurable),
# so they are illustrative; the ENVIRONMENTAL conditions below are REAL measured values.
SCENARIO = dict(pit_loss=20.0, gap=19.50, ours_new_outlap=95.2, rival_old_inlap=95.5, n_remaining=10)

# "Current" environmental conditions read from the REAL cross-domain 2025 test races:
#   wind: Saudi 2025 strongest measured headwind (an undercut is a single-lap decision →
#         use the wind at that moment; the race-mean −0.31 sits in the flat PDP region)
#   temp: Las Vegas 2025 mean track temp (the venue's cold regime, ~17 °C → OOD)
SAUDI_DATA = "data/merged/2025_Saudi_Arabian_Grand_Prix.xlsx"
VEGAS_DATA = "data/merged/2025_Las_Vegas_Grand_Prix.xlsx"


def _load(study: dict) -> tuple:
    model = joblib.load(_bootstrap.PROJECT_ROOT / study["model"])
    _, x, _ = study["preprocessor"].from_excel(str(_bootstrap.PROJECT_ROOT / study["data"])).run()
    return model, x


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("mechanism.py — PDP/ICE figures + symmetric undercut scenarios")
    print("=" * 60)

    models, frames = {}, {}
    for name, study in _STUDIES.items():
        print(f"\n--- {name} study ({study['data']}) ---")
        model, x = _load(study)
        models[name], frames[name] = model, x
        print(f"  data shape: X={x.shape}")
        for feature, pdp_num, ice_num in study["figures"]:
            Visualizer.partial_dependence(
                model, x, feature,
                title=f"PDP — {feature} → LapTimeDelta",
                save_path=OUT_DIR / f"figure_6_{pdp_num}_pdp_{feature}.png",
            )
            Visualizer.ice(
                model, x, feature,
                title=f"ICE + PDP overlay — {feature}",
                save_path=OUT_DIR / f"figure_6_{ice_num}_ice_{feature}.png",
            )
            print(f"  saved figure_6_{pdp_num}_pdp_{feature}.png, figure_6_{ice_num}_ice_{feature}.png")

    # ---- feature ranges ----
    print("\n" + "=" * 60)
    print("Feature ranges (training support)")
    print("=" * 60)
    x_wind, x_temp = frames["wind"], frames["temp"]
    for feat in ("HeadWind", "CrossWind"):
        lo, hi, mean = x_wind[feat].min(), x_wind[feat].max(), x_wind[feat].mean()
        print(f"  Wind  {feat:10s}: [{lo:+6.2f}, {hi:+6.2f}] m/s   mean={mean:+5.2f}   range={hi - lo:.2f}")
    for feat in ("AirTemp", "TrackTemp"):
        lo, hi, mean = x_temp[feat].min(), x_temp[feat].max(), x_temp[feat].mean()
        print(f"  Temp  {feat:10s}: [{lo:6.2f}, {hi:6.2f}] °C    mean={mean:5.2f}   range={hi - lo:.2f}")

    # ---- symmetric undercut scenarios ----
    print("\n" + "=" * 60)
    print("Two symmetric undercut scenarios (identical fixed params)")
    print("=" * 60)
    print(f"  shared: {SCENARIO}")

    # WIND (Saudi cross-domain, physics-compatible): current HeadWind in-support → apply
    print("\n" + "-" * 60)
    print("(1) Saudi Arabia 2025 — wind correction (cross-domain, in-support)")
    print("-" * 60)
    wind = UndercutScenario(models["wind"], x_wind, "HeadWind", **SCENARIO)
    _, x_saudi, _ = WindPreprocessor.from_excel(str(_bootstrap.PROJECT_ROOT / SAUDI_DATA)).run()
    current_hw = float(x_saudi["HeadWind"].max())  # Saudi 2025 strongest measured headwind (real)
    print(f"  current HeadWind = Saudi 2025 measured max = {current_hw:+.2f} m/s "
          f"(race mean {x_saudi['HeadWind'].mean():+.2f} sits in the flat PDP region)")
    wind_res = wind.evaluate(current_hw)
    wind_res.report()

    # TEMP (Las Vegas cross-domain, OOD): current TrackTemp out-of-support → withhold
    print("\n" + "-" * 60)
    print("(2) Las Vegas 2025 — temperature correction (cross-domain, OOD)")
    print("-" * 60)
    temp = UndercutScenario(models["temp"], x_temp, "TrackTemp", **SCENARIO)
    _, x_vegas, _ = TempPreprocessor.from_excel(str(_bootstrap.PROJECT_ROOT / VEGAS_DATA)).run()
    current_temp = float(x_vegas["TrackTemp"].mean())  # Las Vegas 2025 measured mean track temp (real)
    print(f"  current TrackTemp = Vegas 2025 measured mean = {current_temp:.2f} °C (real)")
    temp_res = temp.evaluate(current_temp)
    temp_res.report()

    print("\n" + "=" * 60)
    print("Done — 8 figures in plots/, two scenario tables above.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

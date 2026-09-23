"""CLI：PDP/ICE 圖 + 對稱的 undercut 情境表。

載入兩項研究的 RF 模型，輸出 8 張 PDP/ICE 圖與特徵範圍，接著跑兩個**對稱**的
undercut 情境——固定參數完全相同（pit_loss / gap / 出站圈 / 進站圈 / N），
唯一的差別是被查詢的條件落在支撐區間內還是 OOD：

    wind（沙烏地）   ：當前 HeadWind 在訓練支撐區間內 → 修正生效 → 決策翻轉
    temp（Las Vegas）：當前 TrackTemp 為 OOD（低於訓練最小值）→ 修正收回 → 無法評估

輸出：
    plots/figure_6_{1,2}_pdp_{HeadWind,CrossWind}.png   plots/figure_6_{5,6}_pdp_{AirTemp,TrackTemp}.png
    plots/figure_6_{3,4}_ice_{HeadWind,CrossWind}.png   plots/figure_6_{7,8}_ice_{AirTemp,TrackTemp}.png
    stdout: feature ranges + the two scenario tables

使用方式：
    python scripts/mechanism.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — no display, avoids tkinter cleanup errors on exit

import _bootstrap  # noqa: F401
import joblib

from f1lab import TempPreprocessor, UndercutScenario, Visualizer, WindPreprocessor

OUT_DIR = _bootstrap.PROJECT_ROOT / "plots"

# 兩項研究完全對稱。每一項包含：模型檔、訓練資料，以及四組
# (特徵， pdp 圖號， ice 圖號) 的圖表規格。
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

# 共用的 undercut 參數——兩項研究完全相同（這就是那個對稱）。
# pit_loss/gap/出站圈/進站圈 屬於策略桌參數（遙測量不到），所以是示意用的；
# 底下的「環境條件」則是真實量測到的值。
SCENARIO = dict(pit_loss=20.0, gap=19.50, ours_new_outlap=95.2, rival_old_inlap=95.5, n_remaining=10)

# 「當前」環境條件讀自真實的 2025 跨域測試賽事：
#   wind：沙烏地 2025 實測最強逆風（undercut 是單圈決策 → 取當下那一刻的風；
#         全場平均 −0.31 落在 PDP 的平坦區）
#   temp：Las Vegas 2025 平均賽道溫度（該場地的低溫區間，約 17 °C → OOD）
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

    # ---- 特徵範圍 ----
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

    # ---- 對稱的 undercut 情境 ----
    print("\n" + "=" * 60)
    print("Two symmetric undercut scenarios (identical fixed params)")
    print("=" * 60)
    print(f"  shared: {SCENARIO}")

    # WIND（沙烏地跨域，物理條件相容）：當前 HeadWind 在支撐區間內 → 套用修正
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

    # TEMP（Las Vegas 跨域，OOD）：當前 TrackTemp 超出支撐區間 → 收回修正
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

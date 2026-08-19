"""Streamlit demo: PDP-derived undercut correction with the OOD-refusal UX.

Loads the two study RF models, builds the symmetric `UndercutScenario` pair
(identical fixed params), and lets the user sweep the current condition beyond
the training support to see the correction applied (in-support) or withheld
(OOD) — refusal instead of silent extrapolation.

Usage:
    streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend — Streamlit renders figures itself

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from f1lab import TempPreprocessor, UndercutScenario, WindPreprocessor  # noqa: E402

# Shared undercut parameters — IDENTICAL across both studies (the symmetry).
SCENARIO = dict(pit_loss=20.0, gap=19.50, ours_new_outlap=95.2, rival_old_inlap=95.5, n_remaining=10)

STUDIES = {
    "Wind — HeadWind [m/s]": {
        "feature": "HeadWind",
        "unit": "m/s",
        "model": "models/azerbaijan_rf.joblib",
        "data": "data/merged/2022-2024_Azerbaijan_Grand_Prix.xlsx",
        "preprocessor": WindPreprocessor,
        "slider": dict(min_value=-5.0, max_value=6.0, step=0.05, value=1.38),
    },
    "Temperature — TrackTemp [°C]": {
        "feature": "TrackTemp",
        "unit": "°C",
        "model": "models/singapore_rf.joblib",
        "data": "data/merged/2022-2024_Singapore_Grand_Prix.xlsx",
        "preprocessor": TempPreprocessor,
        "slider": dict(min_value=10.0, max_value=45.0, step=0.1, value=17.3),
    },
}


@st.cache_resource(show_spinner="Loading models + precomputing PDP grids (first run only)...")
def build_scenarios() -> dict[str, UndercutScenario]:
    missing = [s["model"] for s in STUDIES.values() if not (PROJECT_ROOT / s["model"]).exists()]
    if missing:
        st.error(
            f"Missing model files: {', '.join(missing)}. Download the trained models from the "
            "GitHub Release assets and place them in `models/`, or retrain via `python run.py`."
        )
        st.stop()
    scenarios: dict[str, UndercutScenario] = {}
    for label, s in STUDIES.items():
        model = joblib.load(PROJECT_ROOT / s["model"])
        _, x, _ = s["preprocessor"].from_excel(str(PROJECT_ROOT / s["data"])).run()
        scenarios[label] = UndercutScenario(model, x, s["feature"], **SCENARIO)
    return scenarios


st.set_page_config(page_title="F1 Undercut Correction — OOD-aware", layout="wide")
st.title("F1 Undercut Correction — OOD-aware")

scenarios = build_scenarios()

# ---- sidebar: study + current condition ----
study_label = st.sidebar.radio("Study", list(STUDIES))
study = STUDIES[study_label]
scenario = scenarios[study_label]
feature, unit = study["feature"], study["unit"]
lo, hi = scenario.support

current = st.sidebar.slider(f"Current {feature} [{unit}]", **study["slider"])
st.sidebar.caption(f"Training support: [{lo:.2f}, {hi:.2f}] {unit} — slider deliberately extends beyond it.")

res = scenario.evaluate(current)

# ---- (1) fixed strategy parameters ----
st.subheader("Fixed strategy parameters (identical across studies)")
params = pd.DataFrame(
    {
        "Parameter": [
            "pit_loss [s]", "gap uncorrected [s]", "our new-tyre out-lap [s]",
            "rival old-tyre in-lap [s]", "rival remaining laps", "net = pit_loss + out - in [s]",
        ],
        "Value": [
            f"{res.pit_loss:.1f}", f"{res.gap:.2f}", f"{res.ours_new_outlap:.2f}",
            f"{res.rival_old_inlap:.2f}", f"{res.n_remaining}", f"{res.net:.2f}",
        ],
    }
).set_index("Parameter")
st.table(params)

# ---- (2) applicability banner ----
if res.in_support:
    st.success(f"IN SUPPORT — correction applied ({feature} = {current:.2f} {unit} lies inside [{lo:.2f}, {hi:.2f}])")
else:
    st.error(f"OUT OF DISTRIBUTION — correction withheld ({feature} = {current:.2f} {unit} is outside [{lo:.2f}, {hi:.2f}])")

# ---- (3) correction metrics ----
c1, c2, c3 = st.columns(3)
c1.metric("Δ per lap [s]", f"{res.delta_per_lap:+.3f}" if res.applicable else "—")
c2.metric(f"Δ × {res.n_remaining} laps [s]", f"{res.delta_total:+.3f}" if res.applicable else "—")
c3.metric("Corrected gap [s]", f"{res.gap_corrected:.2f}" if res.applicable else "—")

# ---- (4) verdict ----
st.subheader("Verdict")
v1, v2 = st.columns(2)
v1.metric("Uncorrected decision", res.decision_uncorrected)
v2.metric("Corrected decision", res.decision_corrected if res.applicable else "—")
if res.flipped:
    st.warning(f"DECISION FLIP: {res.decision_uncorrected} → {res.decision_corrected} — the correction changes the call.")
elif not res.applicable:
    st.info("PDP undefined outside the training support → corrected decision cannot be evaluated.")

# ---- (5) PDP curve with support band ----
st.subheader(f"PDP — {feature} → LapTimeDelta")
fig, ax = plt.subplots(figsize=(9, 4))
# _grid/_pdp: the scenario's precomputed PDP grid (recomputing it here would double the slow RF PDP)
ax.plot(scenario._grid, scenario._pdp, color="tab:blue", lw=2, label="PDP f(feature)")
ax.axvspan(lo, hi, color="tab:green", alpha=0.12, label=f"training support [{lo:.2f}, {hi:.2f}]")
ax.axvline(scenario.training_mean, color="gray", ls=":", lw=1.5,
           label=f"training mean {scenario.training_mean:.2f}")
if res.in_support:
    ax.axvline(current, color="tab:green", lw=2, label=f"current {current:.2f} (in support)")
else:
    ax.axvline(current, color="red", ls="--", lw=2, label=f"current {current:.2f} (OOD)")
pad = 0.05 * (max(hi, current) - min(lo, current))
ax.set_xlim(min(lo, current) - pad, max(hi, current) + pad)
ax.set_xlabel(f"{feature} [{unit}]")
ax.set_ylabel("PDP of LapTimeDelta [s]")
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# ---- design position ----
st.divider()
st.markdown(
    "**Design position:** outside the training support the PDP has no statistical basis, so the "
    "correction is *withheld* rather than extrapolated — the model refuses instead of silently emitting a wrong number.  \n"
    "Empirical justification: the temperature model's raw cross-domain R² = **−6.08** (Las Vegas ~17 °C vs "
    "Singapore's 27.6–37.4 °C training range). Full rationale: `docs/methodology.md`."
)

"""Streamlit demo: PDP-derived undercut correction with the OOD-refusal UX.

Rebuilds `UndercutScenario` from the exported PDP cache (identical fixed
params across studies) and lets the user sweep the current condition beyond
the training support to see the correction applied (in-support) or withheld
(OOD) — refusal instead of silent extrapolation.

All four causal features are exposed, two per study axis (wind: HeadWind /
CrossWind, temp: TrackTemp / AirTemp). HeadWind and TrackTemp default to the
thesis §5.4 scenario values (Saudi 2025 measured max / Las Vegas 2025 measured
mean); the CrossWind and AirTemp defaults are derived with the same method and
exist for the same-axis structure contrast (threshold-like vs near-linear wind
response; smooth TrackTemp decrease vs AirTemp's sparse-region drop), not as
thesis scenarios.

The cache holds the four PDP curves plus each study's training support and
baseline — the only model-derived values `evaluate()` ever reads — so this app
loads no .joblib and no race data. Regenerate it with
`python scripts/export_pdp_cache.py` after retraining.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — Streamlit renders figures itself

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from f1lab import UndercutScenario  # noqa: E402

CACHE_PATH = PROJECT_ROOT / "demo" / "pdp_cache.json"

# Shared undercut parameters — IDENTICAL across all studies (the symmetry).
SCENARIO = dict(pit_loss=20.0, gap=19.50, ours_new_outlap=95.2, rival_old_inlap=95.5, n_remaining=10)

# Defaults marked "thesis §5.4 scenario" are the canonical values from the thesis;
# the other two are derived with the same method (scripts/mechanism.py: Saudi 2025
# measured max for wind, Las Vegas 2025 measured mean for temperature).
STUDIES = {
    "Wind — HeadWind [m/s]": {
        "feature": "HeadWind",
        "unit": "m/s",
        "axis": "wind",
        "slider": dict(min_value=-5.0, max_value=6.0, step=0.05, value=1.38),
        "default_note": "Default +1.38 m/s = Saudi Arabia 2025 measured max — the thesis §5.4 wind scenario.",
        "caveat": "The step above ≈ +1 m/s sits in a sparse-sample region of the training data — the thesis "
                  "(§5.5) treats magnitudes read there as directional, not precise.",
    },
    "Wind — CrossWind [m/s]": {
        "feature": "CrossWind",
        "unit": "m/s",
        "axis": "wind",
        "slider": dict(min_value=-5.0, max_value=6.0, step=0.05, value=1.37),
        "default_note": "Default +1.37 m/s = Saudi Arabia 2025 measured max, derived exactly like HeadWind's "
                        "canonical value — here for the axis contrast, not a thesis scenario.",
    },
    "Temperature — TrackTemp [°C]": {
        "feature": "TrackTemp",
        "unit": "°C",
        "axis": "temp",
        "slider": dict(min_value=10.0, max_value=45.0, step=0.1, value=17.3),
        "default_note": "Default 17.3 °C = Las Vegas 2025 measured mean — the thesis §5.4 temperature scenario "
                        "(deliberately OOD).",
    },
    "Temperature — AirTemp [°C]": {
        "feature": "AirTemp",
        "unit": "°C",
        "axis": "temp",
        "slider": dict(min_value=10.0, max_value=45.0, step=0.1, value=16.5),
        "default_note": "Default 16.5 °C = Las Vegas 2025 measured mean, derived exactly like TrackTemp's "
                        "canonical value — also OOD (thesis §5.4: both temperature variables are refused).",
        "caveat": "AirTemp's steep drop at 27.3–27.5 °C sits in a sparse-sample region — the thesis (§5.5) "
                  "flags it as possibly a statistical artifact rather than physics; corrections read off "
                  "this curve are directional only.",
    },
}

# Same-axis structure readings, from thesis §5.3 (fig. 5.1–5.8, tables 5.1–5.2).
CONTRAST_NOTES = {
    "wind": "Thesis §5.3(1): HeadWind's PDP is threshold-like — near-flat through the mid-range, stepping up "
            "above ≈ +1 m/s — while CrossWind rises near-linearly across its whole range with roughly twice "
            "the span. §5.4 still demos the correction on HeadWind: its physical story (drag on the straights) "
            "is the most direct on a pit wall, and if the smaller-span feature already flips a boundary "
            "decision, CrossWind bounds the effect from below.",
    "temp": "Thesis §5.3(2): TrackTemp falls smoothly at ≈ −0.06 s/°C — hotter track, tyres closer to their "
            "working window, smaller delta. AirTemp's headline structure is a steep drop at 27.3–27.5 °C in a "
            "sparse-sample region (§5.5: possibly a statistical artifact — directional only). At Las Vegas "
            "both variables are OOD and the support check refuses both (§5.4).",
}


@st.cache_resource(show_spinner="Loading the exported PDP curves...")
def load_cache() -> dict:
    if not CACHE_PATH.exists():
        st.error(
            f"Missing `demo/{CACHE_PATH.name}`. Regenerate it with "
            "`python scripts/export_pdp_cache.py` (needs the trained models in `models/`)."
        )
        st.stop()
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


st.set_page_config(page_title="F1 Undercut Correction — OOD-aware", layout="wide")
st.title("F1 Undercut Correction — OOD-aware")

cache = load_cache()

# ---- sidebar: study + current condition ----
study_label = st.sidebar.radio("Study", list(STUDIES))
study = STUDIES[study_label]
feature, unit = study["feature"], study["unit"]
curve = cache[feature]
scenario = UndercutScenario.from_cache(curve, feature, **SCENARIO)
lo, hi = scenario.support

current = st.sidebar.slider(f"Current {feature} [{unit}]", **study["slider"])
st.sidebar.caption(f"Training support: [{lo:.2f}, {hi:.2f}] {unit} — slider deliberately extends beyond it.")
st.sidebar.caption(study["default_note"])

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
if "caveat" in study:
    st.caption("⚠️ " + study["caveat"])

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
ax.plot(curve["grid"], curve["pdp"], color="tab:blue", lw=2, label="PDP f(feature)")
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

# ---- (6) same-axis structure contrast ----
axis_features = [s["feature"] for s in STUDIES.values() if s["axis"] == study["axis"]]
st.subheader(f"Same-axis contrast — {' vs '.join(axis_features)}")
fig, ax = plt.subplots(figsize=(9, 4))
for feat, color in zip(axis_features, ("tab:blue", "tab:orange")):
    c = cache[feat]
    span = max(c["pdp"]) - min(c["pdp"])
    style = dict(lw=2) if feat == feature else dict(lw=2, ls="--", alpha=0.8)
    ax.plot(c["grid"], c["pdp"], color=color, label=f"{feat} (PDP span ≈ {span:.2f} s)", **style)
ax.set_xlabel(f"[{unit}] — curves cover the 5–95% training quantile range, as in thesis figs 5.1–5.8")
ax.set_ylabel("PDP of LapTimeDelta [s]")
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)
st.caption(CONTRAST_NOTES[study["axis"]])

# ---- design position ----
st.divider()
st.markdown(
    "**Design position:** outside the training support the PDP has no statistical basis, so the "
    "correction is *withheld* rather than extrapolated — the model refuses instead of silently emitting a wrong number.  \n"
    "Empirical justification: the temperature model's raw cross-domain R² = **−6.08** (Las Vegas ~17 °C vs "
    "Singapore's 27.6–37.4 °C training range). Full rationale: `docs/methodology.md`.  \n"
    "The thesis §5.4 scenarios use one canonical feature per axis — HeadWind (the conservative lower bound: "
    "the weaker wind signal already flips the call) and TrackTemp (the diagnostic axis of the cross-domain "
    "failure). CrossWind and AirTemp are exposed for the same-axis structure contrast."
)

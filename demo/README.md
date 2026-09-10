# Demo — OOD-aware undercut correction

Interactive Streamlit front-end for the PDP-derived undercut correction: pick one of the four
causal features — two per study axis (wind: HeadWind / CrossWind; temperature: TrackTemp /
AirTemp) — sweep the current condition — the slider deliberately extends beyond the training
support — and watch the correction get applied (in-support, decision may flip) or withheld (OOD
refusal), with the PDP curve, support band, and current value plotted. A same-axis contrast plot
overlays both PDPs of the selected axis (threshold-like HeadWind vs near-linear CrossWind; smooth
TrackTemp decrease vs AirTemp's sparse-region drop).

## The four studies

Defaults follow one method (`scripts/mechanism.py`): wind values are the Saudi Arabia 2025
measured max (an undercut is a single-lap decision), temperatures the Las Vegas 2025 measured
mean.

| Feature | Default | Status at default | Role |
|---|---|---|---|
| HeadWind | +1.38 m/s | in support → decision flips | thesis §5.4 wind scenario (canonical) |
| CrossWind | +1.37 m/s | in support | axis contrast — not a thesis scenario |
| TrackTemp | 17.3 °C | OOD → correction withheld | thesis §5.4 temperature scenario (canonical) |
| AirTemp | 16.5 °C | OOD → correction withheld | axis contrast — both temp variables are refused (§5.4) |

AirTemp carries an in-app caveat: its steep PDP drop at 27.3–27.5 °C sits in a sparse-sample
region and the thesis (§5.5) flags it as possibly a statistical artifact — corrections read off
that curve are directional only. HeadWind's step above ≈ +1 m/s gets the same qualifier.

Run: `pip install -e .[demo]` (or `pip install streamlit`), then `streamlit run streamlit_app.py` from the
project root.

## Why no model files

`UndercutScenario` touches its model exactly once, in `__init__`, to build the PDP grid;
`evaluate()` afterwards reads only that grid, the training support and the training mean. Those
few hundred numbers live in `pdp_cache.json` (12.6 KB), so the app needs **no `.joblib` and no race
data** — it runs on a fresh clone and boots in seconds instead of recomputing a ~1 min RF PDP.
`UndercutScenario.from_cache()` rebuilds a scenario that is identical field-for-field to the one
built from the model.

Regenerate the cache after retraining — otherwise the demo keeps showing the previous model's PDP:

```
python scripts/export_pdp_cache.py
```

That step is the only one that needs `models/azerbaijan_rf.joblib` + `models/singapore_rf.joblib`
(download from the GitHub Release assets or retrain via `python main.py`) and the two 2022-2024
merged xlsx in `data/merged/`.

## Known quirk

On a cold start the sidebar can occasionally fail to render if the browser connects mid-startup;
refresh once and it comes back.

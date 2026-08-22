# Demo — OOD-aware undercut correction

Interactive Streamlit front-end for the PDP-derived undercut correction: pick a study
(wind/HeadWind or temperature/TrackTemp), sweep the current condition — the slider deliberately
extends beyond the training support — and watch the correction get applied (in-support, decision
may flip) or withheld (OOD refusal), with the PDP curve, support band, and current value plotted.

Run: `pip install -e .[demo]` (or `pip install streamlit`), then `streamlit run app.py` from the
project root.

## Why no model files

`UndercutScenario` touches its model exactly once, in `__init__`, to build the PDP grid;
`evaluate()` afterwards reads only that grid, the training support and the training mean. Those
few hundred numbers live in `pdp_cache.json` (6.7 KB), so the app needs **no `.joblib` and no race
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

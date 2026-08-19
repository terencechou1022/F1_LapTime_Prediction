# Demo — OOD-aware undercut correction

Interactive Streamlit front-end for the PDP-derived undercut correction: pick a study
(wind/HeadWind or temperature/TrackTemp), sweep the current condition — the slider deliberately
extends beyond the training support — and watch the correction get applied (in-support, decision
may flip) or withheld (OOD refusal), with the PDP curve, support band, and current value plotted.

Run: `pip install -e .[demo]` (or `pip install streamlit`), then `streamlit run demo/app.py`.

Requires `models/azerbaijan_rf.joblib` + `models/singapore_rf.joblib` (download from the GitHub
Release assets or retrain via `python run.py`) and the two 2022-2024 merged xlsx in `data/merged/`.

First load takes ~1 min (the RF PDP grids are precomputed once and cached). If you open the page
before that finishes and the sidebar is missing, refresh once.

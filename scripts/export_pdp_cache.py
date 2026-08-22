"""CLI: export the two study PDP curves so the demo needs no model files.

`UndercutScenario` touches its model exactly once, in `__init__`, to build the
PDP grid; `evaluate()` afterwards reads only that grid, the training support and
the training mean. Precomputing those few hundred numbers lets the deployed
Streamlit app ship no .joblib and no race data — `UndercutScenario.from_cache()`
rebuilds an identical scenario from the JSON.

Re-run after retraining, or the demo keeps showing the previous model's PDP.

Outputs:
    demo/pdp_cache.json

Usage:
    python scripts/export_pdp_cache.py
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import joblib

from f1lab import TempPreprocessor, UndercutScenario, WindPreprocessor

OUT_PATH = _bootstrap.PROJECT_ROOT / "demo" / "pdp_cache.json"

# The two features the demo exposes — same models and training data as scripts/mechanism.py.
_STUDIES = {
    "HeadWind": {
        "preprocessor": WindPreprocessor,
        "model": "models/azerbaijan_rf.joblib",
        "data": "data/merged/2022-2024_Azerbaijan_Grand_Prix.xlsx",
    },
    "TrackTemp": {
        "preprocessor": TempPreprocessor,
        "model": "models/singapore_rf.joblib",
        "data": "data/merged/2022-2024_Singapore_Grand_Prix.xlsx",
    },
}

# to_cache() exports only the model-derived values, so the fixed strategy
# parameters are irrelevant here — the app supplies its own.
_UNUSED_PARAMS = dict(pit_loss=0.0, gap=0.0, ours_new_outlap=0.0, rival_old_inlap=0.0, n_remaining=0)


def main() -> None:
    cache = {}
    for feature, study in _STUDIES.items():
        model = joblib.load(_bootstrap.PROJECT_ROOT / study["model"])
        _, x, _ = study["preprocessor"].from_excel(
            str(_bootstrap.PROJECT_ROOT / study["data"])
        ).run()
        scenario = UndercutScenario(model, x, feature, **_UNUSED_PARAMS)
        entry = scenario.to_cache()
        entry["model"] = study["model"]
        cache[feature] = entry
        lo, hi = entry["support"]
        print(f"  {feature:10s} {len(entry['grid']):3d} grid points   "
              f"support [{lo:.2f}, {hi:.2f}]   mean {entry['training_mean']:.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT_PATH.relative_to(_bootstrap.PROJECT_ROOT)} "
          f"({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

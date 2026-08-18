"""Model specifications for the comparative study.

Each entry pairs a scikit-learn-compatible estimator factory with a
GridSearchCV parameter grid. Grids are designed with boundary safety
margins so the search optimum should not land at the explored extremes
without a defensible reason.

Public API:
    MODEL_SPECS         - dict[name -> (factory, param_grid)]
    get_model_spec(name, quick=False) - lookup helper with explicit error;
                          quick=True swaps in a tiny smoke-test grid
    MODEL_NAMES         - tuple of supported names in canonical order
"""
from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# --- Decision Tree -----------------------------------------------------------
# Boundary safety margins (vs. typical optima at this sample size ~1.7k):
#   - max_depth: None (unlimited) and 30 (very deep) at top; 3 (shallow) at
#     bottom. Both ends of the bias-variance spectrum representable.
#   - min_samples_leaf: 1 (sklearn default, no leaf reg) at bottom; 150 at top
#     well above the meaningful "strong regularization" region.
#   - min_samples_split: 2 (sklearn default) at bottom; 200 at top.
#   - max_features: None (all features, DT default) at one end; "sqrt" (~3
#     features for 11-dim input) at the other; 0.5 in the middle. For a
#     single tree this controls early-split variability.
_DT_GRID: dict[str, list[Any]] = {
    "max_depth": [None, 3, 5, 10, 20, 30],
    "min_samples_leaf": [1, 5, 10, 30, 90, 150],
    "min_samples_split": [2, 10, 30, 50, 100, 200],
    "max_features": [None, "sqrt", 0.5],
}


def _dt_factory(random_state: int) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(random_state=random_state)


# --- Random Forest -----------------------------------------------------------
#   - n_estimators: 200/500 verify the 1000-tree plateau claim; 1500/2000
#     extend the upper range so the search doesn't pin at a hard boundary.
#     If 2000 is selected, the cv_results_ gap vs 1500 quantifies whether
#     even more trees would meaningfully help.
#   - max_depth: None (unlimited) and 3 (very shallow) cover both ends.
#   - min_samples_leaf: 1 (sklearn default) reachable; previous grid started
#     at 10 which masked the no-regularization region.
#   - min_samples_split: 2 (sklearn default); both prior models pinned at
#     the previous grid's lower bound 50, a boundary-pinning symptom now
#     resolved.
#   - max_features: the core decorrelation knob for RF; omitting it forces
#     every tree to use every feature, defeating variance-reduction.
_RF_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 500, 1000, 1500, 2000],
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 30, 90],
    "min_samples_split": [2, 10, 30, 50, 100],
    "max_features": ["sqrt", 0.5, 1.0],
}


def _rf_factory(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=random_state, n_jobs=-1)


# --- XGBoost -----------------------------------------------------------------
# Boundary safety margins (XGBoost defaults marked DEFAULT):
#   - n_estimators: 1500/2000 extend the upper range so the search doesn't
#     pin at a hard boundary; same logic as RF. If 2000 is selected, the
#     cv_results_ gap vs 1500 quantifies whether even more trees would help.
#   - max_depth: 6 is XGBoost DEFAULT; 3 shallow; 10 moderate; 15 safety
#     margin (XGBoost rarely benefits beyond depth 10).
#   - learning_rate: 0.3 is XGBoost DEFAULT; 0.05/0.1/0.2 cover the
#     slow-learning regime typical of careful research use.
#   - subsample: 1.0 (no row subsampling) is DEFAULT; 0.7 strong; 0.85 mid.
#   - colsample_bytree: 1.0 is DEFAULT; 0.7 strong; 0.85 mid.
#   - min_child_weight: 1 is DEFAULT; 10 strong leaf regularization.
# All defaults are reachable so a "default-pin" outcome is defensible as a
# finding, not a grid limitation.
_XGB_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 500, 1000, 1500, 2000],
    "max_depth": [3, 6, 10, 15],
    "learning_rate": [0.05, 0.1, 0.2, 0.3],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 10],
}


def _xgb_factory(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
        verbosity=0,
    )


# --- Quick smoke-test grids --------------------------------------------------
# Tiny grids (~2 combos each) for pipeline sanity checks, not for research
# results. Selected via get_model_spec(name, quick=True); the estimator
# factories (and random_state=42) are shared with the full grids.
_DT_QUICK_GRID: dict[str, list[Any]] = {
    "max_depth": [5, 10],
    "min_samples_leaf": [10],
    "min_samples_split": [30],
    "max_features": [None],
}

_RF_QUICK_GRID: dict[str, list[Any]] = {
    "n_estimators": [50],
    "max_depth": [3, 5],
    "min_samples_leaf": [10],
    "min_samples_split": [30],
    "max_features": ["sqrt"],
}

_XGB_QUICK_GRID: dict[str, list[Any]] = {
    "n_estimators": [50],
    "max_depth": [3, 6],
    "learning_rate": [0.1],
    "subsample": [1.0],
    "colsample_bytree": [1.0],
    "min_child_weight": [1],
}

_QUICK_GRIDS: dict[str, dict[str, list[Any]]] = {
    "dt": _DT_QUICK_GRID,
    "rf": _RF_QUICK_GRID,
    "xgb": _XGB_QUICK_GRID,
}


# --- Public registry ---------------------------------------------------------
MODEL_NAMES: tuple[str, ...] = ("dt", "rf", "xgb")

MODEL_SPECS: dict[str, tuple[Callable[[int], Any], dict[str, list[Any]]]] = {
    "dt": (_dt_factory, _DT_GRID),
    "rf": (_rf_factory, _RF_GRID),
    "xgb": (_xgb_factory, _XGB_GRID),
}


def get_model_spec(name: str, quick: bool = False) -> tuple[Callable[[int], Any], dict[str, list[Any]]]:
    """Return (factory, param_grid) for the named model.

    With quick=True the full grid is replaced by a tiny smoke-test grid
    (~2 combos) for fast pipeline verification.

    Raises ValueError on unknown name to fail loudly rather than silently
    fall back to a default.
    """
    if name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{name}'. Choices: {sorted(MODEL_SPECS)}"
        )
    factory, grid = MODEL_SPECS[name]
    if quick:
        grid = _QUICK_GRIDS[name]
    return factory, grid

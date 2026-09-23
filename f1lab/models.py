"""比較研究用的模型規格。

每一項把一個相容於 scikit-learn 的估計器工廠，與一份 GridSearchCV 的參數網格
配成一對。網格都留有邊界餘裕，所以搜尋的最佳解不應該在沒有充分理由的情況下
落在被探索範圍的兩端。

公開 API：
    MODEL_SPECS         - dict[名稱 -> (工廠， 參數網格)]
    get_model_spec(name, quick=False) - 查詢輔助函式，查不到會明確報錯；
                          quick=True 會換成極小的冒煙測試網格
    MODEL_NAMES         - 支援的名稱，依正規順序排列的 tuple
"""
from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# --- Decision Tree -----------------------------------------------------------
# 邊界餘裕（相對於這個樣本數約 1.7k 時的典型最佳解）：
#   - max_depth：上端放 None（無限制）與 30（很深），下端放 3（很淺）。
#     偏差與變異這個光譜的兩端都表示得出來。
#   - min_samples_leaf：下端放 1（sklearn 預設，葉節點不做正則化），
#     上端放 150，遠高於「強正則化」真正有意義的區間。
#   - min_samples_split：下端 2（sklearn 預設），上端 200。
#   - max_features：一端放 None（全部特徵，DT 預設），另一端放 "sqrt"
#     （11 維輸入約取 3 個特徵），中間放 0.5。對單一棵樹而言，
#     這個參數控制的是早期分裂的變異程度。
_DT_GRID: dict[str, list[Any]] = {
    "max_depth": [None, 3, 5, 10, 20, 30],
    "min_samples_leaf": [1, 5, 10, 30, 90, 150],
    "min_samples_split": [2, 10, 30, 50, 100, 200],
    "max_features": [None, "sqrt", 0.5],
}


def _dt_factory(random_state: int) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(random_state=random_state)


# --- Random Forest -----------------------------------------------------------
#   - n_estimators：200/500 用來驗證「1000 棵就進入平台期」這個說法；
#     1500/2000 把上界往外延，讓搜尋不會貼死在硬邊界上。
#     如果選中 2000，cv_results_ 中它與 1500 的差距可以量出更多樹
#     到底有沒有實質幫助。
#   - max_depth：None（無限制）與 3（很淺）涵蓋兩端。
#   - min_samples_leaf：1（sklearn 預設）搆得到；先前的網格從 10 起跳，
#     把「不做正則化」那個區間遮掉了。
#   - min_samples_split：2（sklearn 預設）；先前兩個模型都貼在舊網格的
#     下界 50，那正是貼邊界的徵兆，現在已經解決。
#   - max_features：RF 去相關性的核心旋鈕；不放它等於強迫每棵樹都用
#     全部特徵，變異縮減的效果就沒了。
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
# 邊界餘裕（XGBoost 的預設值標為「預設」）：
#   - n_estimators：1500/2000 把上界往外延，讓搜尋不會貼死在硬邊界上，
#     邏輯與 RF 相同。如果選中 2000，cv_results_ 中它與 1500 的差距
#     可以量出更多樹到底有沒有幫助。
#   - max_depth：6 是 XGBoost 預設；3 很淺；10 中等；15 是安全餘裕
#     （XGBoost 超過深度 10 之後很少再有好處）。
#   - learning_rate：0.3 是 XGBoost 預設；0.05/0.1/0.2 涵蓋研究上
#     謹慎使用時典型的慢速學習區間。
#   - subsample：1.0（不做列抽樣）是預設；0.7 強；0.85 中等。
#   - colsample_bytree：1.0 是預設；0.7 強；0.85 中等。
#   - min_child_weight：1 是預設；10 是強度較高的葉節點正則化。
# 所有預設值都搆得到，所以「最後選中預設值」這個結果站得住腳，
# 是一個發現，而不是網格的限制。
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


# --- 快速冒煙測試網格 ---------------------------------------------------------
# 極小的網格（各約 2 種組合），用來檢查管線是否正常，不是用來產出研究結果。
# 由 get_model_spec(name, quick=True) 選用；估計器工廠（以及 random_state=42）
# 與完整網格共用。
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


# --- 公開註冊表 ---------------------------------------------------------------
MODEL_NAMES: tuple[str, ...] = ("dt", "rf", "xgb")

MODEL_SPECS: dict[str, tuple[Callable[[int], Any], dict[str, list[Any]]]] = {
    "dt": (_dt_factory, _DT_GRID),
    "rf": (_rf_factory, _RF_GRID),
    "xgb": (_xgb_factory, _XGB_GRID),
}


def get_model_spec(name: str, quick: bool = False) -> tuple[Callable[[int], Any], dict[str, list[Any]]]:
    """回傳指定模型名稱的 (工廠， 參數網格)。

    quick=True 時，完整網格會換成極小的冒煙測試網格（約 2 種組合），
    用來快速驗證管線。

    名稱不認得時拋出 ValueError，寧可大聲失敗，也不要默默退回某個預設值。
    """
    if name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{name}'. Choices: {sorted(MODEL_SPECS)}"
        )
    factory, grid = MODEL_SPECS[name]
    if quick:
        grid = _QUICK_GRIDS[name]
    return factory, grid

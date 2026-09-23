"""由 PDP 導出之修正項的策略應用層。

訓練好的研究模型（wind 或 temp）透過 PDP 得到一條反應函式。每圈的修正項是

    Δ = f(當前值) − f(訓練平均)

而這個修正項「只在」當前條件落在模型的訓練支撐區間（該特徵觀測到的
[min, max]）之內，才會套用到 undercut 決策上。落在支撐區間之外時，
該條件就是分佈外（OOD）：RF 的樹從沒在那裡分裂過，所以 PDP 沒有定義，
修正項會被「收回」——模型選擇拒絕預測，而不是默默外推。

兩項研究共用同一個 `UndercutScenario` 與同一組固定參數；它們唯一的差別
在於被查詢的條件是落在支撐區間內（wind／沙烏地 → 修正項生效，決策可能翻轉）
還是 OOD（temp／Las Vegas → 修正項收回，決策無法評估）。這個對稱正是重點。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence


@dataclass
class ScenarioResult:
    """一次 undercut 情境評估的結果。

    被查詢的條件為 OOD 時 `applicable` 為 False，此時 corrected-* 這幾個欄位
    都是 None（決策無法評估）。
    """
    feature: str
    # --- 固定的 undercut 參數（兩項研究完全相同）---
    pit_loss: float
    gap: float
    ours_new_outlap: float
    rival_old_inlap: float
    n_remaining: int
    net: float
    margin_uncorrected: float
    decision_uncorrected: str            # "OPEN" | "CLOSE"
    # --- 修正項與適用性 ---
    training_mean: float
    current_value: float
    support_lo: float
    support_hi: float
    in_support: bool
    delta_per_lap: float | None
    delta_total: float | None
    gap_corrected: float | None
    margin_corrected: float | None
    decision_corrected: str | None       # "OPEN" | "CLOSE" | None (OOD)
    flipped: bool

    @property
    def applicable(self) -> bool:
        return self.in_support

    def report(self) -> None:
        print(f"\n  Feature under study : {self.feature}")
        print("  Scenario fixed parameters:")
        print(f"    pit_loss              = {self.pit_loss:.1f} s")
        print(f"    gap (uncorrected)     = {self.gap:.2f} s")
        print(f"    our new-tyre out-lap  = {self.ours_new_outlap:.2f} s")
        print(f"    rival old-tyre in-lap = {self.rival_old_inlap:.2f} s")
        print(f"    rival remaining laps  = {self.n_remaining}")
        print(f"    net = pit_loss + ours_new - rival_old = {self.net:.2f} s")
        print(f"    uncorrected: net {self.net:.2f} vs gap {self.gap:.2f} "
              f"(margin {self.margin_uncorrected:+.2f}) → {self.decision_uncorrected}")

        print(f"\n  Applicability check ({self.feature}):")
        print(f"    training support      = [{self.support_lo:+.2f}, {self.support_hi:+.2f}]")
        print(f"    training mean         = {self.training_mean:+.2f}")
        print(f"    current value         = {self.current_value:+.2f}")
        print(f"    in support?           = {self.in_support}")

        if self.applicable:
            print("\n  Correction (in-support → applied):")
            print(f"    Δ per lap             = {self.delta_per_lap:+.3f} s")
            print(f"    Δ over {self.n_remaining} laps        = {self.delta_total:+.3f} s")
            print(f"    gap (corrected)       = {self.gap_corrected:.2f} s")
            print(f"    corrected: net {self.net:.2f} vs gap {self.gap_corrected:.2f} "
                  f"(margin {self.margin_corrected:+.2f}) → {self.decision_corrected}")
            if self.flipped:
                print(f"    [FLIP] {self.decision_uncorrected} → {self.decision_corrected}")
            else:
                print("    no flip")
        else:
            print("\n  Correction (OOD → WITHHELD):")
            print(f"    current value {self.current_value:+.2f} is outside training "
                  f"support [{self.support_lo:+.2f}, {self.support_hi:+.2f}]")
            print("    PDP undefined here → correction NOT applicable → decision CANNOT be evaluated")


class UndercutScenario:
    """一個進站 undercut 決策，外加一項由 PDP 導出的修正。

    PDP 網格由該研究的訓練特徵矩陣預先算好一次；
    `evaluate(current_value)` 依觀測到的訓練支撐區間判定適用與否，
    在區間內時才從 PDP 讀出修正量。
    """

    def __init__(
        self,
        model: BaseEstimator,
        x: pd.DataFrame,
        feature: str,
        *,
        pit_loss: float,
        gap: float,
        ours_new_outlap: float,
        rival_old_inlap: float,
        n_remaining: int,
        grid_resolution: int = 100,
    ) -> None:
        self.model = model
        self.x = x
        self.feature = feature
        self.pit_loss = pit_loss
        self.gap = gap
        self.ours_new_outlap = ours_new_outlap
        self.rival_old_inlap = rival_old_inlap
        self.n_remaining = n_remaining

        # PDP 曲線（用來讀出修正量）
        result = partial_dependence(model, x, features=[feature], grid_resolution=grid_resolution)
        self._grid = np.asarray(result["grid_values"][0])
        self._pdp = np.asarray(result["average"][0])
        self._support = (float(x[feature].min()), float(x[feature].max()))
        self._training_mean = float(x[feature].mean())

    @classmethod
    def from_cache(
        cls,
        cache: dict,
        feature: str,
        *,
        pit_loss: float,
        gap: float,
        ours_new_outlap: float,
        rival_old_inlap: float,
        n_remaining: int,
    ) -> UndercutScenario:
        """不帶模型，從 `to_cache()` 匯出的內容重建一個情境。

        模型只在 `__init__` 被碰過一次，用來算出 PDP 網格；之後 `evaluate()`
        只讀那個網格、支撐區間與訓練平均。因此只需要修正量的部署場合，
        可以只帶幾百個浮點數，而不必帶著擬合好的森林與它的訓練資料。
        """
        obj = cls.__new__(cls)
        obj.model = None
        obj.x = None
        obj.feature = feature
        obj.pit_loss = pit_loss
        obj.gap = gap
        obj.ours_new_outlap = ours_new_outlap
        obj.rival_old_inlap = rival_old_inlap
        obj.n_remaining = n_remaining
        obj._grid = np.asarray(cache["grid"], dtype=float)
        obj._pdp = np.asarray(cache["pdp"], dtype=float)
        obj._support = (float(cache["support"][0]), float(cache["support"][1]))
        obj._training_mean = float(cache["training_mean"])
        return obj

    def to_cache(self) -> dict:
        """`evaluate()` 會讀到的那些模型衍生值，以可直接 JSON 化的型別表示。

        `from_cache()` 的反向操作；固定的策略參數刻意不放進來，
        由使用端自己提供。
        """
        lo, hi = self.support
        return {
            "grid": self._grid.tolist(),
            "pdp": self._pdp.tolist(),
            "support": [lo, hi],
            "training_mean": self.training_mean,
        }

    @property
    def support(self) -> tuple[float, float]:
        """訓練支撐區間 = 該特徵觀測到的 [min, max]
        （也就是 RF 真正看過、也真的在上面分裂過的範圍）。"""
        return self._support

    @property
    def training_mean(self) -> float:
        return self._training_mean

    def _pdp_at(self, value: float) -> float:
        """在網格上以線性內插取得任意一點的 PDP 值。"""
        return float(np.interp(value, self._grid, self._pdp))

    def evaluate(self, current_value: float) -> ScenarioResult:
        lo, hi = self.support
        in_support = lo <= current_value <= hi

        net = self.pit_loss + self.ours_new_outlap - self.rival_old_inlap
        margin_unc = self.gap - net
        decision_unc = "OPEN" if net < self.gap else "CLOSE"

        delta = delta_total = gap_corr = margin_corr = decision_corr = None
        flipped = False
        if in_support:
            delta = self._pdp_at(current_value) - self._pdp_at(self.training_mean)
            delta_total = self.n_remaining * delta
            gap_corr = self.gap + delta_total
            margin_corr = gap_corr - net
            decision_corr = "OPEN" if net < gap_corr else "CLOSE"
            flipped = decision_corr != decision_unc

        return ScenarioResult(
            feature=self.feature,
            pit_loss=self.pit_loss,
            gap=self.gap,
            ours_new_outlap=self.ours_new_outlap,
            rival_old_inlap=self.rival_old_inlap,
            n_remaining=self.n_remaining,
            net=net,
            margin_uncorrected=margin_unc,
            decision_uncorrected=decision_unc,
            training_mean=self.training_mean,
            current_value=current_value,
            support_lo=lo,
            support_hi=hi,
            in_support=in_support,
            delta_per_lap=delta,
            delta_total=delta_total,
            gap_corrected=gap_corr,
            margin_corrected=margin_corr,
            decision_corrected=decision_corr,
            flipped=flipped,
        )

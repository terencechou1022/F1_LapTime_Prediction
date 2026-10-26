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

    def simulate_monte_carlo(
        self,
        current_value: float,
        config: MonteCarloConfig | None = None,
    ) -> MonteCarloResult:
        """執行蒙特卡羅模擬，評估進站超車在不確定性下的勝率分佈。"""
        return MonteCarloUndercut(self, config).simulate(current_value)


@dataclass
class MonteCarloConfig:
    """蒙特卡羅模擬的隨機變數參數設定。

    所有標準差（std）預設為合理的 F1 實務量級；
    設為 0.0 時將退化為完全確定性的點估計。
    """
    n_simulations: int = 10000
    random_state: int = 42

    # 進站損失 (秒)：常態分佈 + 尾部失誤機率
    pit_loss_std: float = 0.25
    pit_failure_prob: float = 0.03        # 換胎失誤機率 (如卡螺栓、釋放延遲)
    pit_failure_delay_mean: float = 1.5   # 失誤時額外增加之秒數 (指數分佈)

    # 圈速波動 (秒)
    ours_new_outlap_std: float = 0.20     # 我方出站圈配速波動 (暖胎、車流)
    rival_old_inlap_std: float = 0.25     # 對手進站圈配速波動 (舊胎抓地、進站前推進)
    gap_std: float = 0.05                 # 計時差距雜訊

    # 環境因果變數波動 (與 feature 單位相同，如 m/s 或 °C)
    current_value_std: float = 0.25       # 環境條件擾動 (如陣風/局部氣候波動)
    ood_threshold: float = 0.05           # 允許的最大 OOD 樣本比例，超過則判定不可信


@dataclass
class MonteCarloResult:
    """蒙特卡羅模擬之統計與決策結果。"""
    feature: str
    config: MonteCarloConfig
    current_value: float

    # 樣本陣列 (長度 N)
    margins_uncorrected: np.ndarray
    margins_corrected: np.ndarray | None   # 若 OOD 則為 None

    # 未修正之機率指標
    success_prob_uncorrected: float        # P(margin > 0)
    mean_margin_uncorrected: float
    ci_95_uncorrected: tuple[float, float]

    # 修正後之機率指標 (在支撐域內時計算)
    success_prob_corrected: float | None
    mean_margin_corrected: float | None
    ci_95_corrected: tuple[float, float] | None
    prob_flip: float | None                # 決策受環境修正而翻轉之機率

    # 適用性與 OOD 檢核
    ood_rate: float                        # 落在訓練支撐區間外的抽樣比例
    in_support: bool                       # ood_rate <= config.ood_threshold

    @property
    def applicable(self) -> bool:
        return self.in_support

    def report(self) -> None:
        print(f"\n  Monte Carlo Simulation ({self.feature}, N={self.config.n_simulations:,}):")
        print(f"    Current value (mean)  = {self.current_value:+.2f} (std={self.config.current_value_std:.2f})")
        print(f"    OOD sample rate       = {self.ood_rate * 100:.1f}% (threshold={self.config.ood_threshold * 100:.1f}%)")
        print(f"    In-support / Valid?   = {self.in_support}")

        print("\n  Uncorrected (Base Scenario):")
        print(f"    Win Rate P(margin > 0) = {self.success_prob_uncorrected * 100:.1f}%")
        print(f"    Mean Margin            = {self.mean_margin_uncorrected:+.2f} s")
        print(f"    95% Confidence Interval= [{self.ci_95_uncorrected[0]:+.2f}, {self.ci_95_uncorrected[1]:+.2f}] s")

        if self.applicable and self.success_prob_corrected is not None:
            print("\n  Corrected (with PDP environment adjustment):")
            print(f"    Win Rate P(margin > 0) = {self.success_prob_corrected * 100:.1f}%")
            print(f"    Mean Margin            = {self.mean_margin_corrected:+.2f} s")
            print(f"    95% Confidence Interval= [{self.ci_95_corrected[0]:+.2f}, {self.ci_95_corrected[1]:+.2f}] s")
            diff_prob = (self.success_prob_corrected - self.success_prob_uncorrected) * 100
            print(f"    Win Rate Shift (Δ)     = {diff_prob:+.1f}%")
            print(f"    Decision Flip Prob     = {self.prob_flip * 100:.1f}%")
        else:
            print("\n  Corrected (OOD → WITHHELD):")
            print("    Environment conditions out of training support.")
            print("    Correction NOT applicable → Probabilistic prediction safely withheld.")


class MonteCarloUndercut:
    """UndercutScenario 的蒙特卡羅機率化評估器。

    將決定論式的進站差值計算，擴展為考慮換胎失誤、圈速擾動與風速/溫度波動的
    機率分佈模擬。
    """

    def __init__(self, scenario: UndercutScenario, config: MonteCarloConfig | None = None) -> None:
        self.scenario = scenario
        self.config = config or MonteCarloConfig()

    def simulate(self, current_value: float) -> MonteCarloResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        n = cfg.n_simulations

        # 1. 抽樣進站損失 (常態分佈 + 泊松/指數失誤延遲)
        pit_loss = rng.normal(self.scenario.pit_loss, cfg.pit_loss_std, size=n)
        if cfg.pit_failure_prob > 0 and cfg.pit_failure_delay_mean > 0:
            failures = rng.random(size=n) < cfg.pit_failure_prob
            additional_delay = rng.exponential(cfg.pit_failure_delay_mean, size=n)
            pit_loss = pit_loss + failures * additional_delay

        # 2. 抽樣出站圈、進站圈與差距
        outlap = rng.normal(self.scenario.ours_new_outlap, cfg.ours_new_outlap_std, size=n)
        inlap = rng.normal(self.scenario.rival_old_inlap, cfg.rival_old_inlap_std, size=n)
        gap = rng.normal(self.scenario.gap, cfg.gap_std, size=n)

        # 3. 未修正之淨損失與餘裕
        net = pit_loss + outlap - inlap
        margin_unc = gap - net

        success_prob_unc = float(np.mean(margin_unc > 0))
        mean_margin_unc = float(np.mean(margin_unc))
        ci_95_unc = (
            float(np.percentile(margin_unc, 2.5)),
            float(np.percentile(margin_unc, 97.5)),
        )

        # 4. 抽樣環境因果變數
        env_vals = rng.normal(current_value, cfg.current_value_std, size=n)
        lo, hi = self.scenario.support
        in_support_mask = (env_vals >= lo) & (env_vals <= hi)
        ood_rate = float(np.mean(~in_support_mask))
        in_support = ood_rate <= cfg.ood_threshold

        margin_corr = None
        success_prob_corr = None
        mean_margin_corr = None
        ci_95_corr = None
        prob_flip = None

        if in_support:
            # 限制在訓練支撐區間內做線性內插
            clipped_env = np.clip(env_vals, lo, hi)
            pdp_vals = np.interp(clipped_env, self.scenario._grid, self.scenario._pdp)
            pdp_mean = float(np.interp(self.scenario.training_mean, self.scenario._grid, self.scenario._pdp))

            delta_per_lap = pdp_vals - pdp_mean
            delta_total = self.scenario.n_remaining * delta_per_lap
            gap_corr = gap + delta_total
            margin_corr = gap_corr - net

            success_prob_corr = float(np.mean(margin_corr > 0))
            mean_margin_corr = float(np.mean(margin_corr))
            ci_95_corr = (
                float(np.percentile(margin_corr, 2.5)),
                float(np.percentile(margin_corr, 97.5)),
            )
            prob_flip = float(np.mean((margin_corr > 0) != (margin_unc > 0)))

        return MonteCarloResult(
            feature=self.scenario.feature,
            config=cfg,
            current_value=current_value,
            margins_uncorrected=margin_unc,
            margins_corrected=margin_corr,
            success_prob_uncorrected=success_prob_unc,
            mean_margin_uncorrected=mean_margin_unc,
            ci_95_uncorrected=ci_95_unc,
            success_prob_corrected=success_prob_corr,
            mean_margin_corrected=mean_margin_corr,
            ci_95_corrected=ci_95_corr,
            prob_flip=prob_flip,
            ood_rate=ood_rate,
            in_support=in_support,
        )


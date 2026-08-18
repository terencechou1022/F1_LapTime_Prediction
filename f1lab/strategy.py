"""Strategy-application layer for the PDP-derived correction term.

A trained study model (wind or temp) yields a response function via PDP. The
per-lap correction term is

    Δ = f(current_value) − f(training_mean)

applied to an undercut decision **iff** the current condition lies inside the
model's training support (observed [min, max] of the feature). Outside that
support the condition is out-of-distribution (OOD): the RF's trees never split
there, so the PDP is undefined and the correction is *withheld* — the model
refuses to predict rather than silently extrapolating.

The two studies share an identical `UndercutScenario` and identical fixed
parameters; they differ only in whether the queried condition is in-support
(wind/Saudi → correction applies, decision may flip) or OOD (temp/Las Vegas →
correction withheld, decision cannot be evaluated). That symmetry is the point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence


@dataclass
class ScenarioResult:
    """Outcome of one undercut scenario evaluation.

    `applicable` is False when the queried condition is OOD; the corrected-*
    fields are then None (decision cannot be evaluated).
    """
    feature: str
    # --- fixed undercut parameters (identical across studies) ---
    pit_loss: float
    gap: float
    ours_new_outlap: float
    rival_old_inlap: float
    n_remaining: int
    net: float
    margin_uncorrected: float
    decision_uncorrected: str            # "OPEN" | "CLOSE"
    # --- correction / applicability ---
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
    """A pit-stop undercut decision augmented by a PDP-derived correction term.

    The PDP grid is precomputed once from the study's training feature matrix;
    `evaluate(current_value)` decides applicability by the observed training
    support and (when in-support) reads the correction off the PDP.
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

        # PDP curve (for reading off correction values)
        result = partial_dependence(model, x, features=[feature], grid_resolution=grid_resolution)
        self._grid = np.asarray(result["grid_values"][0])
        self._pdp = np.asarray(result["average"][0])

    @property
    def support(self) -> tuple[float, float]:
        """Training support = observed [min, max] of the feature (the range the
        RF actually saw and split on)."""
        return float(self.x[self.feature].min()), float(self.x[self.feature].max())

    @property
    def training_mean(self) -> float:
        return float(self.x[self.feature].mean())

    def _pdp_at(self, value: float) -> float:
        """PDP value at an arbitrary point via linear interpolation on the grid."""
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

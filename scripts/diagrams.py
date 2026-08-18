"""Generate English portfolio diagrams: system architecture + research flow (matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import _bootstrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = _bootstrap.PROJECT_ROOT / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOD_FC, MOD_EC = "#dce6f1", "#2f5496"   # f1lab module boxes
DAT_FC, DAT_EC = "#f2f2f2", "#7f7f7f"   # CLI / orchestration layer
SHR_FC, SHR_EC = "#e2efda", "#548235"   # shared design boxes
AX1_FC, AX1_EC = "#fbe5d6", "#c55a11"   # temp study
AX2_FC, AX2_EC = "#deebf7", "#1f77b4"   # wind study


def box(ax, cx, cy, w, h, lines, fc, ec, fs=11, lw=1.4, dashed=False, bold_first=True):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, linestyle="--" if dashed else "-")
    ax.add_patch(p)
    n = len(lines)
    for i, ln in enumerate(lines):
        dy = (n - 1) / 2 - i
        weight = "bold" if (i == 0 and bold_first and n > 1) else "normal"
        size = fs if (i == 0 and bold_first) else fs - 1.5
        ax.text(cx, cy + dy * (h / (n + 0.6)), ln, ha="center", va="center",
                fontsize=size, weight=weight, color="#1a1a1a")


def arrow(ax, x1, y1, x2, y2, ec="#404040", lw=1.6, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=16, color=ec, lw=lw, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ================= System architecture =================
fig, ax = plt.subplots(figsize=(11, 7.3), dpi=150)
ax.set_xlim(0, 22)
ax.set_ylim(0, 14.6)
ax.axis("off")
ax.set_title("f1lab system architecture", fontsize=14, weight="bold", pad=12)

# CLI layer on top, Facade below it, then the f1lab package modules
box(ax, 11, 13, 20, 1.6, ["scripts/ (CLI entry points)",
                          "download.py · merge.py · train.py · evaluate.py · summarize.py · mechanism.py"],
    DAT_FC, DAT_EC)
box(ax, 11, 10.9, 14, 1.4, ["f1lab/__init__.py (Facade)",
                            "single import surface: re-exports the public classes"], SHR_FC, SHR_EC)

Y1, Y2, Y3 = 8.4, 5.4, 2.4
box(ax, 3.9, Y1, 6.6, 2.2, ["data.py", "FastF1Downloader · RaceDataMerger",
                            "raw download + per-GP merge"], MOD_FC, MOD_EC)
box(ax, 11, Y1, 6.4, 2.2, ["preprocessing.py", "BaseLapPreprocessor (ABC)",
                           "Template Method + self-registry"], MOD_FC, MOD_EC)
box(ax, 18.1, Y1, 6.6, 2.2, ["experiments.py", "WindPreprocessor · TempPreprocessor",
                             "auto-register via experiment_name"], MOD_FC, MOD_EC)
box(ax, 5.5, Y2, 6.4, 2.2, ["models.py", "get_model_spec()",
                            "dt / rf / xgb factories + search grids"], MOD_FC, MOD_EC)
box(ax, 14, Y2, 7.4, 2.2, ["modeling.py", "ModelTrainer · ModelEvaluator",
                           "Metrics · EvaluationResult"], MOD_FC, MOD_EC)
box(ax, 7, Y3, 7, 2.2, ["visualization.py", "Visualizer",
                        "residual diagnostics + PDP / ICE"], MOD_FC, MOD_EC)
box(ax, 15.5, Y3, 7.4, 2.2, ["strategy.py", "UndercutScenario · ScenarioResult",
                             "PDP correction term + OOD guard"], MOD_FC, MOD_EC)

arrow(ax, 11, 12.2, 11, 11.6)            # scripts -> facade
arrow(ax, 11, 10.2, 11, 9.5)             # facade -> package
arrow(ax, 7.2, Y1, 7.8, Y1)              # data -> preprocessing
arrow(ax, 14.2, Y1, 14.8, Y1)            # preprocessing -> experiments (subclass)
arrow(ax, 18.1, Y1 - 1.1, 16.5, Y2 + 1.1)  # experiments -> modeling (Strategy injection)
arrow(ax, 8.7, Y2, 10.3, Y2)             # models -> modeling (estimator + grid)
arrow(ax, 12, Y2 - 1.1, 8, Y3 + 1.1)     # modeling -> visualization
arrow(ax, 15, Y2 - 1.1, 15.5, Y3 + 1.1)  # modeling -> strategy

fig.savefig(OUT_DIR / "system_architecture.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================= Research flow =================
fig, ax = plt.subplots(figsize=(10, 9.9), dpi=150)
ax.set_xlim(0, 20)
ax.set_ylim(0, 19.8)
ax.axis("off")
ax.set_title("Research flow: two symmetric weather studies", fontsize=14, weight="bold", pad=12)

CXL, CXR, CXM = 5, 15, 10

box(ax, CXM, 18.4, 13, 1.6, ["download (scripts/download.py)",
                             "FastF1 laps · weather · telemetry, seasons 2022–2025"], DAT_FC, DAT_EC, fs=11.5)
box(ax, CXM, 16.3, 13, 1.6, ["merge (scripts/merge.py)",
                             "multi-year per-GP merge, lap × weather timestamp alignment"], DAT_FC, DAT_EC, fs=11.5)

box(ax, CXL, 13.7, 8.6, 2.2, ["Wind study", "train: Azerbaijan 2022–2024 (Baku)",
                              "causal axis: HeadWind · CrossWind"], AX2_FC, AX2_EC, fs=11.5)
box(ax, CXR, 13.7, 8.6, 2.2, ["Temperature study", "train: Singapore 2022–2024 (night race)",
                              "causal axis: AirTemp · TrackTemp"], AX1_FC, AX1_EC, fs=11.5)

box(ax, CXM, 10.9, 15, 2.4, ["train (scripts/train.py): 3 models × 2 studies",
                             "DT · RF · XGB, GridSearchCV × TimeSeriesSplit(5)",
                             "2022–2024: first 80% train / last 20% validation; 2025 held out"],
    SHR_FC, SHR_EC, fs=11.5)

box(ax, CXL, 7.8, 8.6, 3.0, ["evaluate: wind study", "in-domain: 2025 Azerbaijan",
                             "cross-domain: 2025 Saudi Arabia (Jeddah)",
                             "each with raw + bias-corrected metrics",
                             "physics-compatible: extrapolation holds"], AX2_FC, AX2_EC, fs=11.5)
box(ax, CXR, 7.8, 8.6, 3.0, ["evaluate: temp study", "in-domain: 2025 Singapore",
                             "cross-domain: 2025 Las Vegas",
                             "each with raw + bias-corrected metrics",
                             "out-of-distribution: cross-domain fails"], AX1_FC, AX1_EC, fs=11.5)

box(ax, CXM, 4.9, 15, 1.6, ["summarize (scripts/summarize.py)",
                            "parse 30 logs into summary/metrics.csv + best_params.csv"], SHR_FC, SHR_EC, fs=11.5)

box(ax, CXM, 2.1, 15, 2.4, ["mechanism (scripts/mechanism.py)",
                            "PDP / ICE response functions + symmetric undercut scenarios",
                            "in-support: apply correction / OOD: withhold, refuse to extrapolate"],
    SHR_FC, SHR_EC, fs=11.5)

arrow(ax, CXM, 17.6, CXM, 17.1)   # download -> merge
arrow(ax, CXL, 15.5, CXL, 14.8)   # merge -> wind study
arrow(ax, CXR, 15.5, CXR, 14.8)   # merge -> temp study
arrow(ax, CXL, 12.6, CXL, 12.1)   # studies -> train
arrow(ax, CXR, 12.6, CXR, 12.1)
arrow(ax, CXL, 9.7, CXL, 9.3)     # train -> evaluate
arrow(ax, CXR, 9.7, CXR, 9.3)
arrow(ax, CXL, 6.3, CXL, 5.7)     # evaluate -> summarize
arrow(ax, CXR, 6.3, CXR, 5.7)
arrow(ax, CXM, 4.1, CXM, 3.3)     # summarize -> mechanism

fig.savefig(OUT_DIR / "research_flow.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("diagrams generated")

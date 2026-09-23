"""一道指令跑完整條管線：6 次訓練 + 24 次評估 + 彙整 + 機制抽取。

每個步驟依序執行，各自在 `logs/` 底下留一份 log，跑完之後 `summary/` 與
`plots/` 就都有東西了。在 8 核 CPU 上預算約 75 至 80 小時。

步驟是由底下的研究／模型／條件清單產生的，所以某一步的 log 檔名與它的旗標
永遠來自同一組參數。

使用方式：
    python main.py              # 完整執行（用啟動它的那個直譯器）
    python main.py --dry-run    # 只印出指令序列，不實際執行
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = "plots"

STUDIES = ("wind", "temp")
# 每項研究都先跑 DT：在投入 16 小時以上給 RF 之前，先快速確認管線沒問題。
MODELS = ("dt", "rf", "xgb")
DOMAINS = (("indomain", "{study}"), ("crossdomain", "{study}-cross-domain"))
MODES = (("raw", []), ("bias", ["--bias-correct"]))


def build_steps() -> list[tuple[str, list[str], str]]:
    """依執行順序回傳 (label, argv, log_name) 這組三元組。"""
    steps: list[tuple[str, list[str], str]] = []

    for study in STUDIES:
        for model in MODELS:
            steps.append((
                f"train {study} {model}",
                ["train.py", "--experiment", study, "--model", model,
                 "--save", "--save-plots-dir", PLOTS_DIR],
                f"train_{study}_{model}.log",
            ))

    for study in STUDIES:
        for domain, experiment_tpl in DOMAINS:
            for mode, mode_flags in MODES:
                for model in MODELS:
                    experiment = experiment_tpl.format(study=study)
                    steps.append((
                        f"evaluate {study} {domain} {mode} {model}",
                        ["evaluate.py", "--experiment", experiment, "--model", model,
                         *mode_flags, "--save-plots-dir", PLOTS_DIR],
                        f"eval_{study}_{domain}_{mode}_{model}.log",
                    ))

    steps.append(("summarize", ["summarize.py"], "summarize.log"))
    steps.append(("mechanism (PDP/ICE + undercut scenarios)", ["mechanism.py"], "mechanism.log"))
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full training/evaluation pipeline.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command sequence and exit without running anything.",
    )
    args = parser.parse_args()

    steps = build_steps()

    if args.dry_run:
        for _, argv, log_name in steps:
            print(f"python scripts/{argv[0]} {' '.join(argv[1:])} > logs/{log_name} 2>&1")
        print(f"\n{len(steps)} commands.")
        return 0

    LOGS_DIR.mkdir(exist_ok=True)
    failures: list[str] = []

    for index, (label, argv, log_name) in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {label}  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        started = time.time()
        # 每個步驟把 stdout 與 stderr 寫進自己的 log；summarize.py 之後會解析這些檔。
        with (LOGS_DIR / log_name).open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / argv[0]), *argv[1:]],
                stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT,
            )
        elapsed = time.time() - started
        # 不提早結束：某一步失敗就留下它的 log，其餘步驟照跑。
        status = "ok" if completed.returncode == 0 else f"FAILED (exit {completed.returncode})"
        print(f"      {status} in {elapsed / 60:.1f} min -> logs/{log_name}", flush=True)
        if completed.returncode != 0:
            failures.append(f"{label} (logs/{log_name})")

    print(f"\n=== DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    if failures:
        print(f"{len(failures)} step(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

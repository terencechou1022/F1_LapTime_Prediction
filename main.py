"""One-command full pipeline: 6 trainings + 24 evaluations + summary + mechanism.

Runs every step sequentially with its own log file in `logs/`, then leaves
`summary/` and `plots/` populated. Budget ~75-80 h on an 8-core CPU.

Steps are generated from the study/model/condition lists below, so the log
filename and the flags of a step always come from the same parameters.

Usage:
    python main.py              # full run (uses the interpreter it is launched with)
    python main.py --dry-run    # print the command sequence without running it
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
# DT first per study: a fast pipeline sanity check before committing 16+ hours to RF.
MODELS = ("dt", "rf", "xgb")
DOMAINS = (("indomain", "{study}"), ("crossdomain", "{study}-cross-domain"))
MODES = (("raw", []), ("bias", ["--bias-correct"]))


def build_steps() -> list[tuple[str, list[str], str]]:
    """Return the (label, argv, log_name) triples in execution order."""
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
        # Each step writes stdout+stderr to its own log; summarize.py parses these.
        with (LOGS_DIR / log_name).open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / argv[0]), *argv[1:]],
                stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT,
            )
        elapsed = time.time() - started
        # No early exit: a failed step leaves its log behind and the rest still runs.
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

"""CLI: merge per-year raw laps + weather into a single tidy file."""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from f1lab import RaceDataMerger


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge laps + weather files for a GP.")
    parser.add_argument("gp_name", help="GP name with underscores, e.g. Azerbaijan_Grand_Prix.")
    parser.add_argument("--years", type=int, nargs="+", required=True, help="Years to merge.")
    parser.add_argument("--driver", help="If set, merge telemetry for this driver instead.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=_bootstrap.PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=_bootstrap.PROJECT_ROOT / "data" / "merged",
    )
    args = parser.parse_args()

    merger = RaceDataMerger(args.raw_root, args.merged_root)

    if args.driver:
        out = merger.merge_telemetry(args.gp_name, args.driver, args.years)
    else:
        out = merger.merge_laps_weather(args.gp_name, args.years)

    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

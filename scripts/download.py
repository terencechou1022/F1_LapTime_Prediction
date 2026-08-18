"""CLI: download raw race data via fastf1."""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from f1lab import FastF1Downloader


def main() -> int:
    parser = argparse.ArgumentParser(description="Download F1 race data via fastf1.")
    parser.add_argument("--start", type=int, required=True, help="Start year (inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End year (inclusive).")
    parser.add_argument(
        "--output",
        type=Path,
        default=_bootstrap.PROJECT_ROOT / "data" / "raw",
        help="Output directory for raw files.",
    )
    args = parser.parse_args()

    downloader = FastF1Downloader(output_root=args.output)
    failures = downloader.download_range(args.start, args.end)

    if failures:
        print("\nFailed events:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll events downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

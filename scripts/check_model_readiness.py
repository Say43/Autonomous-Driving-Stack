#!/usr/bin/env python3
"""Inspect whether the current machine can run Alpamayo's M0 reference path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.model.readiness import assess_runtime, discover_runtime, format_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.cwd(),
        help="filesystem that would hold the approximately 22 GiB model download",
    )
    parser.add_argument(
        "--attention-backend",
        default="sdpa",
        choices=("sdpa", "flash_attention_2", "eager"),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    report = assess_runtime(
        discover_runtime(args.cache_dir), attention_backend=args.attention_backend
    )
    if args.json:
        print(json.dumps(report.to_json_dict(), indent=2))
    else:
        print(format_report(report))
    return 0 if report.reference_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

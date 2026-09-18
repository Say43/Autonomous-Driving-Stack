#!/usr/bin/env python3
"""Attach Kaggle batch ``PlanResult`` objects to a recorded CARLA trace.

The source trace is never modified in place. Plans are attached only to the
matching inference frame; the viewer keeps the latest plan visible between
replans while anchoring it to the source ego pose.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.record.reader import read_trace  # noqa: E402
from acarla.types import PlanResult  # noqa: E402


def _array_data(value: object) -> object:
    """Accept both Kaggle's nested lists and acarla's ndarray JSON form."""
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return value


def _load_plans(path: Path) -> dict[int, PlanResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list")

    plans: dict[int, PlanResult] = {}
    hashes: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: every plan must be a JSON object")
        if item.get("finite") is False:
            raise ValueError(f"{path}: frame {item.get('frame_id')} is marked non-finite")
        plan = PlanResult(
            frame_id=int(item["frame_id"]),
            waypoints_xyz=np.asarray(_array_data(item["waypoints_xyz"]), dtype=np.float32),
            waypoints_rot=np.asarray(_array_data(item["waypoints_rot"]), dtype=np.float32),
            reasoning=item.get("reasoning"),
            inference_ms=float(item["inference_ms"]),
            model_config_hash=str(item["model_config_hash"]),
        )
        if plan.frame_id in plans:
            raise ValueError(f"{path}: duplicate plan for frame {plan.frame_id}")
        plans[plan.frame_id] = plan
        hashes.add(plan.model_config_hash)
    if len(hashes) != 1:
        raise ValueError(f"{path}: plans use multiple model_config_hash values: {hashes}")
    return plans


def inject_plans(trace_path: Path, plans_path: Path, out_path: Path) -> tuple[int, int]:
    if trace_path.resolve() == out_path.resolve():
        raise ValueError("refusing to overwrite the source trace in place")

    plans = _load_plans(plans_path)
    header, frame_iter = read_trace(trace_path)
    frames = list(frame_iter)
    if frame_iter.skipped_lines:
        raise ValueError(
            f"{trace_path}: reader skipped {frame_iter.skipped_lines} malformed frame(s)"
        )

    frame_ids = {frame.frame_id for frame in frames}
    missing = sorted(set(plans) - frame_ids)
    if missing:
        raise ValueError(f"{plans_path}: plan frame IDs missing from trace: {missing}")

    model_hash = next(iter(plans.values())).model_config_hash
    output_header = replace(
        header,
        model_config_name="alpamayo-1.5-carla-batch",
        model_config_hash=model_hash,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(output_header.to_json_dict(), sort_keys=True) + "\n")
        attached = 0
        for frame in frames:
            plan = plans.get(frame.frame_id)
            if plan is not None:
                attached += 1
            # The output represents exactly this batch result. Clear any
            # pre-existing baseline/synthetic plan on non-matching frames.
            frame = replace(frame, plan=plan)
            output.write(json.dumps(frame.to_json_dict(), sort_keys=True) + "\n")
    return attached, len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    attached, n_frames = inject_plans(args.trace, args.plans, args.out)
    print(f"wrote {args.out}: attached {attached} plan(s) to {n_frames} frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

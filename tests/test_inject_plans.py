"""Tests for scripts/inject_plans.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from inject_plans import inject_plans  # noqa: E402
from make_synthetic_trace import write_trace  # noqa: E402

from acarla.record.reader import read_trace  # noqa: E402


def _plan_json_from_trace(trace_path: Path, frame_id: int) -> dict:
    _, frame_iter = read_trace(trace_path)
    frame = next(frame for frame in frame_iter if frame.frame_id == frame_id)
    assert frame.plan is not None
    plan = frame.plan.to_json_dict()
    plan["finite"] = True
    return plan


def test_injects_plan_and_updates_header_without_touching_source(tmp_path: Path) -> None:
    source = write_trace(tmp_path / "source", ticks=20, seed=3)
    source_before = source.read_bytes()
    plan_dict = _plan_json_from_trace(source, frame_id=10)
    plans_path = tmp_path / "plans.json"
    plans_path.write_text(json.dumps([plan_dict]), encoding="utf-8")
    output = tmp_path / "trace_with_plans.jsonl"

    attached, n_frames = inject_plans(source, plans_path, output)

    assert attached == 1
    assert n_frames == 20
    assert source.read_bytes() == source_before
    header, frame_iter = read_trace(output)
    frames = list(frame_iter)
    assert header.model_config_name == "alpamayo-1.5-carla-batch"
    assert header.model_config_hash == plan_dict["model_config_hash"]
    assert [frame.frame_id for frame in frames if frame.plan is not None] == [10]


def test_refuses_non_finite_plan_and_in_place_overwrite(tmp_path: Path) -> None:
    source = write_trace(tmp_path / "source", ticks=5, seed=2)
    plan_dict = _plan_json_from_trace(source, frame_id=0)
    plan_dict["finite"] = False
    plans_path = tmp_path / "plans.json"
    plans_path.write_text(json.dumps([plan_dict]), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        inject_plans(source, plans_path, tmp_path / "output.jsonl")
    with pytest.raises(ValueError, match="overwrite"):
        inject_plans(source, plans_path, source)

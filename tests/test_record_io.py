"""Tests for acarla.record.writer / acarla.record.reader.

Runs in the carla-less `.venv` (Python 3.13) -- neither module under test
may import `carla`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from make_synthetic_trace import build_frames  # noqa: E402

from acarla.record.reader import read_trace  # noqa: E402
from acarla.record.writer import TraceWriter  # noqa: E402
from acarla.types import ControlCommand, Pose, RunHeader, TraceFrame  # noqa: E402


def test_no_carla_import() -> None:
    """The writer/reader modules must stay importable without carla."""
    assert "carla" not in sys.modules


def _make_header(cameras: list[str] | None = None) -> RunHeader:
    return RunHeader(
        carla_version="0.9.16",
        map_name="Town01",
        seed_world=1,
        seed_traffic_manager=1,
        model_config_name="none-m4-open-loop",
        model_config_hash="none",
        git_sha="0" * 40,
        run_timestamp="1970-01-01T00:00:00Z",
        cameras=cameras or ["front_wide", "cross_left"],
        fixed_delta_seconds=0.05,
    )


def _make_frame(frame_id: int, with_images: bool = True) -> TraceFrame:
    return TraceFrame(
        frame_id=frame_id,
        sim_time=frame_id * 0.05,
        ego_pose_world=Pose(
            translation=np.array([frame_id * 1.0, 0.0, 0.0], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        ),
        actors=[],
        lanes=[],
        traffic_lights=[],
        plan=None,
        control=ControlCommand(steer=0.0, throttle=0.3, brake=0.0),
        image_paths=(
            {"front_wide": [f"images/front_wide/{frame_id:06d}.png"]}
            if with_images
            else {}
        ),
    )


def test_writer_roundtrip(tmp_path: Path) -> None:
    header = _make_header()
    out_dir = tmp_path / "run"
    with TraceWriter(out_dir, header) as writer:
        for i in range(5):
            writer.write_frame(_make_frame(i))
        assert writer.n_frames_written == 5

    read_header, frames = read_trace(out_dir / "trace.jsonl")
    assert read_header.map_name == "Town01"
    assert read_header.cameras == ["front_wide", "cross_left"]

    frame_list = list(frames)
    assert len(frame_list) == 5
    for i, frame in enumerate(frame_list):
        assert frame.frame_id == i
        assert np.array_equal(
            frame.ego_pose_world.translation, np.array([i, 0.0, 0.0], dtype=np.float32)
        )
    assert frames.skipped_lines == 0


def test_writer_streams_without_holding_frames_in_memory(tmp_path: Path) -> None:
    """The writer must not accumulate written frames -- it holds only the
    open file handle plus per-call state."""
    header = _make_header()
    writer = TraceWriter(tmp_path / "run", header)
    for i in range(50):
        writer.write_frame(_make_frame(i))
    writer.close()
    # No internal frame buffer beyond the running count.
    assert not hasattr(writer, "_frames")
    assert not hasattr(writer, "frames")


def test_image_path_helpers(tmp_path: Path) -> None:
    header = _make_header()
    writer = TraceWriter(tmp_path / "run", header)
    try:
        abs_path = writer.image_path("front_wide", 3)
        rel_path = writer.relative_image_path("front_wide", 3)
        assert abs_path.parent.is_dir()
        assert abs_path.name == "000003.png"
        assert rel_path == "images/front_wide/000003.png"
        assert abs_path == writer.images_dir / "front_wide" / "000003.png"
    finally:
        writer.close()


def test_writing_after_close_raises(tmp_path: Path) -> None:
    header = _make_header()
    writer = TraceWriter(tmp_path / "run", header)
    writer.close()
    with pytest.raises(RuntimeError):
        writer.write_frame(_make_frame(0))


def test_reader_skips_and_counts_malformed_lines(tmp_path: Path) -> None:
    header = _make_header()
    out_dir = tmp_path / "run"
    writer = TraceWriter(out_dir, header)
    writer.write_frame(_make_frame(0))
    writer.write_frame(_make_frame(1))
    writer.close()

    trace_path = out_dir / "trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write('{"frame_id": 99}\n')  # valid JSON, missing required fields
        f.write("\n")  # blank line: not corruption, just skipped silently

    _, frames = read_trace(trace_path)
    frame_list = list(frames)

    assert [f.frame_id for f in frame_list] == [0, 1]
    assert frames.skipped_lines == 2
    assert len(frames.skipped_details) == 2


def test_reader_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        read_trace(empty)


def test_reader_bad_header_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad_header.jsonl"
    bad.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_trace(bad)


def test_roundtrip_against_synthetic_trace_frames(tmp_path: Path) -> None:
    """Cross-check against the shared synthetic-trace generator so the
    writer/reader are validated against realistically-shaped frames (with
    actors, lanes, traffic lights, and a PlanResult), not just the minimal
    stub frames built above."""
    header, frames = build_frames(ticks=15, seed=2)
    out_dir = tmp_path / "synthetic_run"
    with TraceWriter(out_dir, header) as writer:
        for frame in frames:
            writer.write_frame(frame)

    read_header, read_frames = read_trace(out_dir / "trace.jsonl")
    assert read_header.cameras == header.cameras
    read_frame_list = list(read_frames)
    assert len(read_frame_list) == len(frames)
    for original, roundtripped in zip(frames, read_frame_list, strict=True):
        assert original.frame_id == roundtripped.frame_id
        assert np.array_equal(
            original.ego_pose_world.translation, roundtripped.ego_pose_world.translation
        )
        assert len(original.actors) == len(roundtripped.actors)
    assert read_frames.skipped_lines == 0

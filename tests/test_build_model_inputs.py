"""Tests for scripts/build_model_inputs.py.

Runs in the carla-less `.venv` (Python 3.13). Builds a tiny synthetic run
(a scaled-down copy of `configs/rig_alpamayo.yaml` at 96x54, a short
trace.jsonl, and matching PNGs) and checks the shapes/dtypes of the
resulting `.npz`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_model_inputs import build_model_inputs  # noqa: E402

from acarla.record.writer import TraceWriter  # noqa: E402
from acarla.types import (  # noqa: E402
    N_EGO_WAYPOINTS,
    N_HISTORY_FRAMES,
    ControlCommand,
    Pose,
    RunHeader,
    TraceFrame,
)

REAL_RIG = ROOT / "configs" / "rig_alpamayo.yaml"

# Calibration data extracted from the PhysicalAI-AV dataset is licensed for internal
# use only (NVIDIA AV Dataset License, section 4.6) and is therefore not part of the
# public repository; regenerate it with the notebooks/scripts if you hold a licence.
pytestmark = pytest.mark.skipif(
    not REAL_RIG.exists(), reason="private calibrated rig (configs/rig_alpamayo.yaml) not present"
)
MINI_WIDTH, MINI_HEIGHT = 96, 54
SCALE = MINI_WIDTH / 1920.0


def _make_mini_rig_yaml(tmp_path: Path) -> Path:
    """Scale `configs/rig_alpamayo.yaml` down to a 96x54 render size,
    rescaling the F-Theta intrinsics polynomials consistently."""
    with REAL_RIG.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    for cam_dict in doc["cameras"].values():
        intr = cam_dict["intrinsics"]
        intr["width"] = MINI_WIDTH
        intr["height"] = MINI_HEIGHT
        intr["cx"] = intr["cx"] * SCALE
        intr["cy"] = intr["cy"] * SCALE
        intr["fw_poly"] = [c * SCALE for c in intr["fw_poly"]]
        intr["bw_poly"] = [c / (SCALE**i) for i, c in enumerate(intr["bw_poly"])]

        cam_dict["carla"]["render_size"] = [MINI_WIDTH, MINI_HEIGHT]

    mini_path = tmp_path / "mini_rig.yaml"
    with mini_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f)
    return mini_path


def _write_mini_run(run_dir: Path, camera_names: list[str], n_ticks: int) -> None:
    header = RunHeader(
        carla_version="synthetic",
        map_name="test",
        seed_world=0,
        seed_traffic_manager=0,
        model_config_name="test",
        model_config_hash="test",
        git_sha="0" * 40,
        run_timestamp="1970-01-01T00:00:00Z",
        cameras=camera_names,
        fixed_delta_seconds=0.05,
    )
    identity_pose = Pose(
        translation=np.zeros(3, dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
    )
    rng = np.random.default_rng(0)

    with TraceWriter(run_dir, header) as writer:
        for frame_id in range(n_ticks):
            image_paths: dict[str, list[str]] = {}
            # sensor_tick is 0.1s while the trace runs at fixed_delta_seconds
            # = 0.05s, so images only land on every 2nd trace tick.
            if frame_id % 2 == 0:
                for cam in camera_names:
                    path = writer.image_path(cam, frame_id)
                    pixels = rng.integers(
                        0, 255, size=(MINI_HEIGHT, MINI_WIDTH, 3), dtype=np.uint8
                    )
                    Image.fromarray(pixels, mode="RGB").save(path)
                    image_paths[cam] = [writer.relative_image_path(cam, frame_id)]

            frame = TraceFrame(
                frame_id=frame_id,
                sim_time=round(frame_id * 0.05, 6),
                ego_pose_world=identity_pose,
                actors=[],
                lanes=[],
                traffic_lights=[],
                plan=None,
                control=ControlCommand(steer=0.0, throttle=0.0, brake=0.0),
                image_paths=image_paths,
            )
            writer.write_frame(frame)


def test_build_model_inputs_shapes(tmp_path: Path) -> None:
    mini_rig = _make_mini_rig_yaml(tmp_path)
    camera_names = ["cross_left", "front_wide", "cross_right", "front_tele"]
    run_dir = tmp_path / "run"

    n_ticks = 35
    _write_mini_run(run_dir, camera_names, n_ticks)

    out_arrays, meta = build_model_inputs(
        run_dir, mini_rig, every=1, max_packets=3
    )

    n_cam = len(camera_names)
    p_expected = 3

    assert out_arrays["image_frames"].shape == (
        p_expected,
        n_cam,
        N_HISTORY_FRAMES,
        3,
        MINI_HEIGHT,
        MINI_WIDTH,
    )
    assert out_arrays["image_frames"].dtype == np.uint8

    assert out_arrays["camera_indices"].tolist() == [0, 1, 2, 6]

    assert out_arrays["ego_history_xyz"].shape == (
        p_expected,
        1,
        1,
        N_EGO_WAYPOINTS,
        3,
    )
    np.testing.assert_allclose(out_arrays["ego_history_xyz"][:, 0, 0, -1, :], 0.0)

    assert out_arrays["ego_history_rot"].shape == (
        p_expected,
        1,
        1,
        N_EGO_WAYPOINTS,
        3,
        3,
    )
    for p in range(p_expected):
        for k in range(N_EGO_WAYPOINTS):
            r = out_arrays["ego_history_rot"][p, 0, 0, k]
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-4)
            assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-4)

    assert out_arrays["frame_ids"].shape == (p_expected,)
    assert out_arrays["frame_ids"].tolist() == [30, 32, 34]
    assert out_arrays["sim_times"].shape == (p_expected,)

    assert meta["n_packets"] == p_expected
    assert meta["camera_names"] == camera_names
    assert "run_header" in meta
    assert meta["total_bytes"] > 0

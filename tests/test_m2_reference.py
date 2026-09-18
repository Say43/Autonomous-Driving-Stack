"""M2 reference test: adapter output must be tensor-identical to the upstream
loader's output on a real recorded scene.

Requires `golden/m2_reference/raw_reference.npz` and
`golden/m2_reference/loader_reference.npz`, produced by a Kaggle run that
captures both the raw inputs to and the output of
`load_physical_aiavdataset`. If they are not present yet, this test is
skipped -- it is not this module's job to fabricate them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acarla.adapter.packet import build_sensor_packet
from acarla.model.inputs import sensor_packet_to_model_inputs
from acarla.types import SAMPLE_DT, SAMPLE_DT_TOLERANCE

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden" / "m2_reference"
RAW_PATH = GOLDEN_DIR / "raw_reference.npz"
LOADER_PATH = GOLDEN_DIR / "loader_reference.npz"

CAMERA_NAMES = ("cross_left", "front_wide", "cross_right", "front_tele")


def _require_uniform_grid(values_us: np.ndarray, label: str) -> None:
    """Fail loudly (rather than silently loosening the contract) if the real
    recorded timestamps do not sit on the model's assumed 10 Hz grid."""
    deltas_ms = np.diff(values_us.astype(np.float64)) / 1000.0
    expected_ms = SAMPLE_DT * 1000.0
    tolerance_ms = SAMPLE_DT_TOLERANCE * 1000.0
    bad = np.abs(deltas_ms - expected_ms) > tolerance_ms
    if np.any(bad):
        worst = np.max(np.abs(deltas_ms - expected_ms))
        pytest.fail(
            f"{label}: real recorded frame deltas violate SAMPLE_DT_TOLERANCE "
            f"-- worst deviation is {worst:.4f} ms from the expected "
            f"{expected_ms:.1f} ms step (tolerance {tolerance_ms:.1f} ms). "
            f"deltas_ms={deltas_ms.tolist()}. This is a finding for the "
            "orchestrator, not something to relax in types.py."
        )


@pytest.mark.skipif(
    not (RAW_PATH.exists() and LOADER_PATH.exists()),
    reason=(
        "golden/m2_reference/{raw_reference.npz,loader_reference.npz} are "
        "missing -- they are produced by a Kaggle run that is not part of "
        "this checkout yet. Skipping the tensor-identity reference test."
    ),
)
def test_adapter_matches_upstream_loader_exactly() -> None:
    raw = np.load(RAW_PATH)
    ref = np.load(LOADER_PATH)

    world_xyz = raw["world_xyz_history"]
    world_quat_xyzw = raw["world_quat_xyzw_history"]
    history_timestamps_us = raw["history_timestamps_us"]

    _require_uniform_grid(history_timestamps_us, "history_timestamps_us")

    frames_hwc: dict[str, np.ndarray] = {}
    image_timestamps: dict[str, list[float]] = {}
    for cam in CAMERA_NAMES:
        frames_hwc[cam] = raw[f"frames_hwc_{cam}"]
        frame_ts_us = raw[f"frame_timestamps_us_{cam}"]
        _require_uniform_grid(frame_ts_us, f"frame_timestamps_us_{cam}")
        t0 = int(frame_ts_us[0])
        image_timestamps[cam] = [(int(t) - t0) / 1_000_000.0 for t in frame_ts_us]

    t0_ego = int(history_timestamps_us[0])
    ego_timestamps = [(int(t) - t0_ego) / 1_000_000.0 for t in history_timestamps_us]

    packet = build_sensor_packet(
        frame_id=0,
        sim_time=0.0,
        frames_hwc=frames_hwc,
        image_timestamps=image_timestamps,
        world_xyz=world_xyz,
        world_quat_xyzw=world_quat_xyzw,
        ego_timestamps=ego_timestamps,
    )

    inputs = sensor_packet_to_model_inputs(packet)

    np.testing.assert_array_equal(inputs["camera_indices"], np.array([0, 1, 2, 6]))
    np.testing.assert_array_equal(inputs["image_frames"], ref["image_frames"])

    xyz_diff = np.abs(inputs["ego_history_xyz"] - ref["ego_history_xyz"])
    rot_diff = np.abs(inputs["ego_history_rot"] - ref["ego_history_rot"])
    print(f"max |ego_history_xyz diff| = {xyz_diff.max():.3e}")
    print(f"max |ego_history_rot diff| = {rot_diff.max():.3e}")

    np.testing.assert_allclose(
        inputs["ego_history_xyz"], ref["ego_history_xyz"], atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        inputs["ego_history_rot"], ref["ego_history_rot"], atol=1e-6, rtol=0
    )

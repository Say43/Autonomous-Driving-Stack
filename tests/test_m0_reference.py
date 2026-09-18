from __future__ import annotations

import numpy as np
import pytest

from acarla.model.m0_reference import (
    json_safe,
    max_repeat_delta_meters,
    min_ade_meters,
    validate_model_outputs,
    validate_upstream_sample,
)


def _sample():
    history_rot = np.broadcast_to(np.eye(3, dtype=np.float32), (1, 1, 16, 3, 3)).copy()
    return {
        "image_frames": np.zeros((4, 4, 3, 32, 48), dtype=np.uint8),
        "camera_indices": np.array([0, 1, 2, 6]),
        "ego_history_xyz": np.zeros((1, 1, 16, 3), dtype=np.float32),
        "ego_history_rot": history_rot,
        "ego_future_xyz": np.zeros((1, 1, 64, 3), dtype=np.float32),
    }


def _outputs(num_samples=1):
    xyz = np.zeros((1, 1, num_samples, 64, 3), dtype=np.float32)
    rot = np.broadcast_to(
        np.eye(3, dtype=np.float32), (1, 1, num_samples, 64, 3, 3)
    ).copy()
    return xyz, rot


def test_validates_exact_upstream_sample_contract():
    summary = validate_upstream_sample(_sample())

    assert summary["camera_indices"] == [0, 1, 2, 6]
    assert summary["image_frames_shape"] == [4, 4, 3, 32, 48]


def test_rejects_wrong_camera_order():
    sample = _sample()
    sample["camera_indices"] = np.array([1, 6, 0, 2])

    with pytest.raises(ValueError, match="camera_indices"):
        validate_upstream_sample(sample)


def test_rejects_non_finite_trajectory():
    xyz, rot = _outputs()
    xyz[..., 3, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_model_outputs(xyz, rot)


def test_min_ade_uses_best_sample_and_xy_only():
    xyz, _ = _outputs(num_samples=2)
    xyz[0, 0, 0, :, 0] = 2.0
    xyz[0, 0, 1, :, 0] = 0.5
    xyz[0, 0, 1, :, 2] = 100.0

    assert min_ade_meters(xyz, _sample()["ego_future_xyz"]) == pytest.approx(0.5)


def test_repeat_delta_and_json_conversion():
    xyz, _ = _outputs()
    changed = xyz.copy()
    changed[..., 0] = 0.25

    assert max_repeat_delta_meters([xyz, changed]) == pytest.approx(0.25)
    assert json_safe({"value": np.float32(1.5), "array": np.array([1, 2])}) == {
        "value": 1.5,
        "array": [1, 2],
    }

"""Wire format of the slow-motion closed loop (acarla.loop.codec)."""

import numpy as np
import pytest

from acarla.loop.codec import decode_job, decode_plan, encode_job, encode_plan
from acarla.types import N_PLAN_WAYPOINTS


def _model_inputs(h=24, w=40, seed=0):
    rng = np.random.default_rng(seed)
    # smooth-ish content so JPEG stays close; PNG must be exact regardless
    base = rng.integers(0, 255, size=(4, 4, 3, h, w), dtype=np.uint8)
    rot = np.tile(np.eye(3, dtype=np.float32), (1, 1, 16, 1, 1))
    return {
        "image_frames": base,
        "camera_indices": np.array([0, 1, 2, 6], dtype=np.int64),
        "ego_history_xyz": rng.standard_normal((1, 1, 16, 3)).astype(np.float32),
        "ego_history_rot": rot,
        "camera_prompt_names": ["a", "b", "c", "d"],
    }


def test_png_job_roundtrip_is_bit_exact():
    mi = _model_inputs()
    payload = encode_job(mi, run_id="r1", seq=3, frame_id=77, sim_time=3.85, codec="png")
    out = decode_job(payload)
    assert out["image_frames"].shape == mi["image_frames"].shape
    assert out["image_frames"].dtype == np.uint8
    np.testing.assert_array_equal(out["image_frames"], mi["image_frames"])
    np.testing.assert_array_equal(out["camera_indices"], mi["camera_indices"])
    np.testing.assert_array_equal(out["ego_history_xyz"], mi["ego_history_xyz"])
    np.testing.assert_array_equal(out["ego_history_rot"], mi["ego_history_rot"])
    assert out["meta"]["seq"] == 3 and out["meta"]["frame_id"] == 77
    assert out["meta"]["run_id"] == "r1" and out["meta"]["codec"] == "png"


def test_jpg_job_roundtrip_keeps_layout_and_is_close():
    mi = _model_inputs()
    # flat frames: JPEG error is then only quantisation noise, layout errors would be huge
    mi["image_frames"] = np.full_like(mi["image_frames"], 120)
    mi["image_frames"][:, :, 1] = 40  # distinct per-channel values catch CHW/HWC mix-ups
    payload = encode_job(mi, run_id="r1", seq=0, frame_id=0, sim_time=0.0, codec="jpg")
    out = decode_job(payload)
    assert out["image_frames"].shape == mi["image_frames"].shape
    diff = np.abs(out["image_frames"].astype(int) - mi["image_frames"].astype(int))
    assert diff.max() <= 3


def test_encode_job_rejects_wrong_shape():
    mi = _model_inputs()
    mi["image_frames"] = np.transpose(mi["image_frames"], (0, 1, 3, 4, 2))  # HWC by mistake
    with pytest.raises(ValueError):
        encode_job(mi, run_id="r", seq=0, frame_id=0, sim_time=0.0)


def test_plan_roundtrip_and_addressing():
    xyz = np.zeros((N_PLAN_WAYPOINTS, 3))
    xyz[:, 0] = np.arange(1, N_PLAN_WAYPOINTS + 1) * 0.5
    rot = np.tile(np.eye(3), (N_PLAN_WAYPOINTS, 1, 1))
    text = encode_plan(
        run_id="r1", seq=4, frame_id=90, sim_time=4.5, waypoints_xyz=xyz, waypoints_rot=rot,
        reasoning="go", inference_ms=1234.5, model_config_hash="abc", worker_session="w1",
    )
    plan, doc = decode_plan(text, expect_run_id="r1", expect_seq=4)
    assert plan.frame_id == 90 and plan.reasoning == "go"
    assert plan.waypoints_xyz.dtype == np.float32
    np.testing.assert_allclose(plan.waypoints_xyz[:, 0], xyz[:, 0])
    assert doc["finite"] is True and doc["worker_session"] == "w1"
    with pytest.raises(ValueError):
        decode_plan(text, expect_run_id="r1", expect_seq=5)
    with pytest.raises(ValueError):
        decode_plan(text, expect_run_id="other", expect_seq=4)


def test_plan_non_finite_is_rejected():
    xyz = np.zeros((N_PLAN_WAYPOINTS, 3))
    xyz[10, 1] = np.nan
    rot = np.tile(np.eye(3), (N_PLAN_WAYPOINTS, 1, 1))
    text = encode_plan(
        run_id="r", seq=0, frame_id=0, sim_time=0.0, waypoints_xyz=xyz, waypoints_rot=rot,
        reasoning=None, inference_ms=0.0, model_config_hash="h", worker_session="w",
    )
    with pytest.raises(ValueError):
        decode_plan(text, expect_run_id="r", expect_seq=0)

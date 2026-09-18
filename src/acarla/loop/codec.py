"""Wire format between the local CARLA loop and the remote Alpamayo worker.

A *job* carries exactly the raw numpy inputs `sensor_packet_to_model_inputs`
produces for one `SensorPacket`, with the 16 camera frames JPEG- or
PNG-encoded so a job stays at a few megabytes instead of ~300 MB. A *plan*
is the worker's answer: one `PlanResult` plus provenance, as JSON.

Fidelity note: the verified batch chain fed the F-Theta-remapped frames to the
model uncompressed (`model_inputs.npz`). With `codec="jpg"` the remapped
frames are compressed a second time (the CARLA render was already stored as
JPEG q92); `codec="png"` reproduces the batch tensors bit-exactly at ~8x the
job size. The processor downsamples 1080x1920 -> 320x576 anyway.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image

from acarla.types import (
    N_HISTORY_FRAMES,
    N_PLAN_WAYPOINTS,
    PlanResult,
)

JOB_FORMAT = "acarla-loop-job-v1"
PLAN_FORMAT = "acarla-loop-plan-v1"
JPEG_QUALITY = 92  # same as the recorder writes to disk, so the second pass costs little


def encode_job(
    model_inputs: dict[str, np.ndarray],
    *,
    run_id: str,
    seq: int,
    frame_id: int,
    sim_time: float,
    codec: str = "jpg",
    quality: int = JPEG_QUALITY,
) -> bytes:
    """Serialize one packet's model inputs into an .npz byte string."""
    if codec not in ("jpg", "png"):
        raise ValueError(f"codec must be 'jpg' or 'png', got {codec!r}")
    frames = np.asarray(model_inputs["image_frames"])
    if frames.ndim != 5 or frames.shape[1] != N_HISTORY_FRAMES or frames.shape[2] != 3:
        raise ValueError(f"image_frames: expected (N_cam, 4, 3, H, W), got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"image_frames: expected uint8, got {frames.dtype}")
    camera_indices = np.asarray(model_inputs["camera_indices"], dtype=np.int64)
    if camera_indices.shape != (frames.shape[0],):
        raise ValueError("camera_indices must have one entry per camera")

    arrays: dict[str, np.ndarray] = {
        "camera_indices": camera_indices,
        "ego_history_xyz": np.asarray(model_inputs["ego_history_xyz"], dtype=np.float32),
        "ego_history_rot": np.asarray(model_inputs["ego_history_rot"], dtype=np.float32),
    }
    for n in range(frames.shape[0]):
        for k in range(N_HISTORY_FRAMES):
            # CHW -> HWC only for the image encoder; `decode_job` inverts it.
            hwc = np.ascontiguousarray(np.transpose(frames[n, k], (1, 2, 0)))
            buf = io.BytesIO()
            if codec == "jpg":
                Image.fromarray(hwc).save(buf, "JPEG", quality=quality)
            else:
                Image.fromarray(hwc).save(buf, "PNG")
            arrays[f"img_{n}_{k}"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    meta = {
        "format": JOB_FORMAT,
        "run_id": run_id,
        "seq": int(seq),
        "frame_id": int(frame_id),
        "sim_time": float(sim_time),
        "codec": codec,
        "quality": int(quality) if codec == "jpg" else None,
        "n_cameras": int(frames.shape[0]),
        "height": int(frames.shape[3]),
        "width": int(frames.shape[4]),
        "camera_prompt_names": list(model_inputs.get("camera_prompt_names", [])),
    }
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def decode_job(payload: bytes) -> dict[str, Any]:
    """Inverse of `encode_job`; returns model inputs plus the `meta` dict."""
    with np.load(io.BytesIO(payload)) as pk:
        meta = json.loads(bytes(pk["meta_json"]).decode("utf-8"))
        if meta.get("format") != JOB_FORMAT:
            raise ValueError(f"unexpected job format {meta.get('format')!r}")
        n_cam, h, w = meta["n_cameras"], meta["height"], meta["width"]
        frames = np.empty((n_cam, N_HISTORY_FRAMES, 3, h, w), dtype=np.uint8)
        for n in range(n_cam):
            for k in range(N_HISTORY_FRAMES):
                img = Image.open(io.BytesIO(bytes(pk[f"img_{n}_{k}"]))).convert("RGB")
                hwc = np.asarray(img, dtype=np.uint8)
                if hwc.shape != (h, w, 3):
                    raise ValueError(f"img_{n}_{k}: expected {(h, w, 3)}, got {hwc.shape}")
                frames[n, k] = np.transpose(hwc, (2, 0, 1))  # back to CHW
        return {
            "image_frames": frames,
            "camera_indices": pk["camera_indices"].astype(np.int64),
            "ego_history_xyz": pk["ego_history_xyz"].astype(np.float32),
            "ego_history_rot": pk["ego_history_rot"].astype(np.float32),
            "meta": meta,
        }


def encode_plan(
    *,
    run_id: str,
    seq: int,
    frame_id: int,
    sim_time: float,
    waypoints_xyz: np.ndarray,
    waypoints_rot: np.ndarray,
    reasoning: str | None,
    inference_ms: float,
    model_config_hash: str,
    worker_session: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Serialize a worker answer as JSON text (plain lists, no numpy types)."""
    xyz = np.asarray(waypoints_xyz, dtype=np.float64)
    rot = np.asarray(waypoints_rot, dtype=np.float64)
    if xyz.shape != (N_PLAN_WAYPOINTS, 3) or rot.shape != (N_PLAN_WAYPOINTS, 3, 3):
        raise ValueError(f"waypoints: got {xyz.shape} / {rot.shape}")
    doc = {
        "format": PLAN_FORMAT,
        "run_id": run_id,
        "seq": int(seq),
        "frame_id": int(frame_id),
        "sim_time": float(sim_time),
        "waypoints_xyz": xyz.tolist(),
        "waypoints_rot": rot.tolist(),
        "reasoning": reasoning,
        "inference_ms": float(inference_ms),
        "model_config_hash": str(model_config_hash),
        "finite": bool(np.isfinite(xyz).all() and np.isfinite(rot).all()),
        "worker_session": worker_session,
        "extra": extra or {},
    }
    return json.dumps(doc)


def decode_plan(text: str, *, expect_run_id: str, expect_seq: int) -> tuple[PlanResult, dict[str, Any]]:
    """Parse a worker answer into a `PlanResult` (contract-validated) + raw doc."""
    doc = json.loads(text)
    if doc.get("format") != PLAN_FORMAT:
        raise ValueError(f"unexpected plan format {doc.get('format')!r}")
    if doc.get("run_id") != expect_run_id or int(doc.get("seq", -1)) != expect_seq:
        raise ValueError(
            f"plan addressed to run={doc.get('run_id')} seq={doc.get('seq')}, "
            f"expected run={expect_run_id} seq={expect_seq}"
        )
    if not doc.get("finite", False):
        raise ValueError("worker marked the plan non-finite")
    plan = PlanResult(
        frame_id=int(doc["frame_id"]),
        waypoints_xyz=np.asarray(doc["waypoints_xyz"], dtype=np.float32),
        waypoints_rot=np.asarray(doc["waypoints_rot"], dtype=np.float32),
        reasoning=doc.get("reasoning"),
        inference_ms=float(doc["inference_ms"]),
        model_config_hash=str(doc["model_config_hash"]),
    )
    return plan, doc

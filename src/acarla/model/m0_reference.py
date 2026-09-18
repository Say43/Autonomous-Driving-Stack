"""Pure validation and metric helpers for the M0 Alpamayo reference run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_CAMERA_INDICES = (0, 1, 2, 6)
N_CAMERAS = 4
N_IMAGE_FRAMES = 4
N_HISTORY_STEPS = 16
N_FUTURE_STEPS = 64


def to_numpy(value: Any) -> np.ndarray:
    """Convert numpy arrays or detached CPU-capable tensors to numpy."""
    if isinstance(value, np.ndarray):
        return value
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if operation is not None:
            current = operation()
    numpy_method = getattr(current, "numpy", None)
    if numpy_method is None:
        return np.asarray(current)
    return numpy_method()


def validate_upstream_sample(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact upstream four-camera sample contract used by M0."""
    required = {
        "image_frames",
        "camera_indices",
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"upstream sample is missing keys: {missing}")

    frames = to_numpy(data["image_frames"])
    camera_indices = tuple(int(index) for index in to_numpy(data["camera_indices"]).reshape(-1))
    history_xyz = to_numpy(data["ego_history_xyz"])
    history_rot = to_numpy(data["ego_history_rot"])
    future_xyz = to_numpy(data["ego_future_xyz"])

    _require_shape_prefix(frames, (N_CAMERAS, N_IMAGE_FRAMES, 3), "image_frames")
    if frames.dtype != np.uint8:
        raise ValueError(f"image_frames: expected uint8, got {frames.dtype}")
    if camera_indices != DEFAULT_CAMERA_INDICES:
        raise ValueError(
            f"camera_indices: expected {DEFAULT_CAMERA_INDICES}, got {camera_indices}"
        )
    _require_shape(history_xyz, (1, 1, N_HISTORY_STEPS, 3), "ego_history_xyz")
    _require_shape(history_rot, (1, 1, N_HISTORY_STEPS, 3, 3), "ego_history_rot")
    _require_shape(future_xyz, (1, 1, N_FUTURE_STEPS, 3), "ego_future_xyz")

    if not np.issubdtype(history_xyz.dtype, np.floating):
        raise ValueError(f"ego_history_xyz: expected floating dtype, got {history_xyz.dtype}")
    if not np.allclose(history_xyz[0, 0, -1], 0.0, atol=1e-6):
        raise ValueError("ego_history_xyz: final pose must be the t0 ego-frame origin")
    if not np.allclose(history_rot[0, 0, -1], np.eye(3), atol=1e-6):
        raise ValueError("ego_history_rot: final pose must be identity in the t0 ego frame")

    return {
        "image_frames_shape": list(frames.shape),
        "image_frames_dtype": str(frames.dtype),
        "camera_indices": list(camera_indices),
        "ego_history_xyz_shape": list(history_xyz.shape),
        "ego_history_rot_shape": list(history_rot.shape),
        "ego_future_xyz_shape": list(future_xyz.shape),
    }


def validate_model_outputs(pred_xyz: Any, pred_rot: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return Alpamayo outputs as CPU numpy arrays."""
    xyz = _validate_pred_xyz(pred_xyz)
    rot = to_numpy(pred_rot)
    expected_rot = (1, 1, xyz.shape[2], N_FUTURE_STEPS, 3, 3)
    _require_shape(rot, expected_rot, "pred_rot")
    if not np.isfinite(rot).all():
        raise ValueError("pred_rot contains non-finite values")
    return xyz, rot


def min_ade_meters(pred_xyz: Any, future_xyz: Any) -> float:
    """Compute upstream-compatible minADE over XY for one batch/group."""
    xyz = _validate_pred_xyz(pred_xyz)
    target = to_numpy(future_xyz)
    _require_shape(target, (1, 1, N_FUTURE_STEPS, 3), "ego_future_xyz")
    prediction_xy = xyz[0, 0, :, :, :2]
    target_xy = target[0, 0, :, :2]
    per_sample = np.linalg.norm(prediction_xy - target_xy[None, ...], axis=-1).mean(axis=-1)
    return float(per_sample.min())


def max_repeat_delta_meters(predictions: Sequence[Any]) -> float:
    """Maximum absolute XYZ difference from the first repeated seeded run."""
    if len(predictions) < 2:
        return 0.0
    arrays = [to_numpy(prediction) for prediction in predictions]
    reference_shape = arrays[0].shape
    if any(array.shape != reference_shape for array in arrays[1:]):
        raise ValueError("repeated predictions have different shapes")
    return max(float(np.max(np.abs(array - arrays[0]))) for array in arrays[1:])


def json_safe(value: Any) -> Any:
    """Recursively convert tensors, numpy values, and mappings for JSON output."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    array = to_numpy(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _validate_pred_xyz(pred_xyz: Any) -> np.ndarray:
    xyz = to_numpy(pred_xyz)
    if xyz.ndim != 5 or xyz.shape[0:2] != (1, 1) or xyz.shape[-2:] != (N_FUTURE_STEPS, 3):
        raise ValueError(
            "pred_xyz: expected (1, 1, num_samples, 64, 3), "
            f"got {xyz.shape}"
        )
    if not np.isfinite(xyz).all():
        raise ValueError("pred_xyz contains non-finite values")
    return xyz


def _require_shape(value: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if value.shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {value.shape}")


def _require_shape_prefix(value: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if value.ndim != len(expected) + 2 or value.shape[: len(expected)] != expected:
        raise ValueError(
            f"{name}: expected shape ({', '.join(map(str, expected))}, H, W), "
            f"got {value.shape}"
        )

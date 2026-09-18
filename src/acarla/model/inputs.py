"""Translate a `SensorPacket` into the numpy inputs the upstream model expects.

See `docs/alpamayo_api_findings.md` ("Konsequenzen fuer den Projektplan" #3
and the "Bilder" table) and
`third_party/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py` for the
target shapes/dtypes this mirrors. Pure numpy -- no torch. The worker running
on Kaggle is responsible for converting these arrays to tensors.
"""

from __future__ import annotations

import numpy as np

from acarla.types import (
    ALPAMAYO_CAMERA_INDICES,
    ALPAMAYO_CAMERA_PROMPT_NAMES,
    SensorPacket,
    alpamayo_camera_order,
)


def sensor_packet_to_model_inputs(packet: SensorPacket) -> dict[str, np.ndarray]:
    """Convert a `SensorPacket` into the upstream model's raw numpy inputs.

    Returns a dict with:
        - `image_frames`: `(N_cam, 4, 3, H, W)` uint8, cameras sorted
          ascending by Alpamayo camera index.
        - `camera_indices`: `(N_cam,)` int64, Alpamayo indices in the same
          camera order as `image_frames`.
        - `ego_history_xyz`: `(1, 1, 16, 3)` float32.
        - `ego_history_rot`: `(1, 1, 16, 3, 3)` float32.
        - `camera_prompt_names`: list[str] of prompt clear-text names, same
          camera order as `image_frames`.
    """
    cameras = alpamayo_camera_order(packet.images.keys())

    # HWC -> CHW transpose happens exactly here, and only here, in the whole
    # project (see docs/alpamayo_api_findings.md, "Bilder" table: upstream
    # does `rearrange(..., "t h w c -> t c h w")`).
    image_frames = np.stack(
        [np.transpose(packet.images[cam], (0, 3, 1, 2)) for cam in cameras], axis=0
    ).astype(np.uint8)

    camera_indices = np.array(
        [ALPAMAYO_CAMERA_INDICES[cam] for cam in cameras], dtype=np.int64
    )

    ego_history_xyz = packet.ego_translation.astype(np.float32)[np.newaxis, np.newaxis, ...]
    ego_history_rot = packet.ego_rotation.astype(np.float32)[np.newaxis, np.newaxis, ...]

    camera_prompt_names = [ALPAMAYO_CAMERA_PROMPT_NAMES[cam] for cam in cameras]

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "camera_prompt_names": camera_prompt_names,
    }

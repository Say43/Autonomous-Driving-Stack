"""Build a `SensorPacket` (see `acarla.types`) from raw sensor/ego history."""

from __future__ import annotations

import numpy as np

from acarla.adapter.ego_frame import world_to_ego_history
from acarla.types import SensorPacket


def build_sensor_packet(
    *,
    frame_id: int,
    sim_time: float,
    frames_hwc: dict[str, np.ndarray],
    image_timestamps: dict[str, list[float]],
    world_xyz: np.ndarray,
    world_quat_xyzw: np.ndarray,
    ego_timestamps: list[float],
    command: str = "drive normally",
    nav_guidance: str | None = None,
) -> SensorPacket:
    """Assemble a `SensorPacket` from raw camera frames and ego pose history.

    Args:
        frame_id: The simulation frame index this packet was built for.
        sim_time: The simulation time (seconds) this packet was built for.
        frames_hwc: Per-camera native-layout frames, `(4, H, W, 3)` uint8,
            oldest first -- CARLA and the reference dataset both hand out
            HWC, so no transpose happens here.
        image_timestamps: Per-camera image timestamps (seconds), oldest
            first, one list of length 4 per camera in `frames_hwc`.
        world_xyz: `(N, 3)` float64 world positions of the ego vehicle,
            oldest first, most recent entry is t0.
        world_quat_xyzw: `(N, 4)` float64 world orientations of the ego
            vehicle in scipy (x, y, z, w) order, oldest first.
        ego_timestamps: Timestamps (seconds) matching `world_xyz`/
            `world_quat_xyzw`, oldest first.
        command: Free-text driving command carried in the packet.
        nav_guidance: Optional free-text navigation guidance.

    Returns:
        A `SensorPacket`. All contract validation (camera names, the 10 Hz
        grid, rotation validity) happens inside `SensorPacket.__post_init__`
        and is not caught here.
    """
    xyz_local, rot_local = world_to_ego_history(world_xyz, world_quat_xyzw)

    return SensorPacket(
        frame_id=frame_id,
        sim_time=sim_time,
        images=dict(frames_hwc),
        image_timestamps={cam: list(ts) for cam, ts in image_timestamps.items()},
        ego_translation=xyz_local,
        ego_rotation=rot_local,
        ego_timestamps=list(ego_timestamps),
        command=command,
        nav_guidance=nav_guidance,
    )

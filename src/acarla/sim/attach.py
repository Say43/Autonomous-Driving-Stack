"""Rig-origin-to-actor-origin offset formula for attaching cameras to a CARLA
vehicle actor.

`configs/rig_alpamayo.yaml` (`rig_origin_note`) records a HYPOTHESIS, not
yet verified against any oracle: the rig origin (the (0, 0, 0) that every
camera `carla.location` in that file is relative to) is the rear axle at
ground height. CARLA, however, attaches sensors relative to the vehicle
actor's own origin, which for CARLA vehicle blueprints is conventionally the
bounding-box center (the `bounding_box.location` offset is usually small and
mostly vertical, but is not assumed to be exactly zero here).

Under the rear-axle-at-ground hypothesis, the rig origin sits at:

    rig_origin_in_actor_frame = bbox_location
                                 + (-rear_axle_to_bbox_center, 0, -bbox_extent_z)

i.e. `rear_axle_to_bbox_center` meters behind the bbox center (the rear axle
is behind the geometric center of a typical vehicle) and `bbox_extent_z`
meters below it (the bbox center is at half the vehicle's height, ground is
a full half-height below that). To attach a camera at a rig-frame location
`L` (as read from `configs/rig_alpamayo.yaml`) to the CARLA actor, the
actor-relative location CARLA needs is `L + rig_origin_in_actor_frame` --
that offset is what `rig_to_actor_offset` returns, and what
`apply_offset` adds to a `CameraSpec`.

If this hypothesis is wrong, the clearest symptom would show up in the
rendered images' horizon height: if the true rig origin's z is not actually
at ground level (e.g. it is really at the bbox center, or at axle height
rather than ground height), every camera z-translation above is offset by a
constant amount, so every rendered image would show the horizon at a
systematically wrong image row (too high if the true origin is below the
hypothesis, too low if above) -- a rigid vertical shift common to all four
cameras, not a per-camera distortion. A wrong longitudinal (x) origin would
similarly shift the apparent forward/backward placement of the hood/mirrors
relative to the vehicle's own body meshes when the ego actor is visible
in-frame (e.g. in a chase-camera debug view), rather than affecting the
external scene geometry. Verifying or falsifying this hypothesis against the
real CARLA vehicle bounding box is the job of Paket 2 (measuring
`rear_axle_to_bbox_center`, `bbox_extent_z`, and `bbox_location` against the
live CARLA server); this module only implements the arithmetic once those
numbers are known.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from acarla.sim.rig import CameraSpec


def rig_to_actor_offset(
    rear_axle_to_bbox_center: float,
    bbox_extent_z: float,
    bbox_location_xyz: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Return the rig origin's position in the CARLA actor's local frame,
    under the rear-axle-at-ground-height hypothesis (see module docstring).

    This is the offset to ADD to every camera's rig-frame `carla.location`
    (as read from `configs/rig_alpamayo.yaml`) to get the location CARLA
    needs when attaching that camera relative to the vehicle actor.

    Args:
        rear_axle_to_bbox_center: Meters the rear axle sits behind the
            bounding-box center (positive), i.e.
            `vehicle.rear_axle_to_bbox_center` from the rig YAML.
        bbox_extent_z: Half the vehicle's height (the bounding box's z
            half-extent), i.e. `vehicle.height / 2`.
        bbox_location_xyz: The actor's `bounding_box.location` (its offset
            from the actor's own transform origin), CARLA convention
            (x forward, y right, z up).

    Returns:
        `(3,)` float64 array: rig origin's (x, y, z) in the actor's local
        frame, CARLA convention.
    """
    bbox_location = np.asarray(bbox_location_xyz, dtype=float)
    if bbox_location.shape != (3,):
        raise ValueError(
            f"bbox_location_xyz: expected shape (3,), got {bbox_location.shape}"
        )
    return bbox_location + np.array(
        [-float(rear_axle_to_bbox_center), 0.0, -float(bbox_extent_z)]
    )


def apply_offset(spec: CameraSpec, offset: np.ndarray) -> CameraSpec:
    """Return a copy of `spec` with `offset` added to its (x, y, z) location.

    `spec.roll`/`pitch`/`yaw` are unchanged -- the offset is a pure
    translation of the attachment origin, not a rotation.
    """
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,):
        raise ValueError(f"offset: expected shape (3,), got {offset.shape}")
    return replace(
        spec,
        x=spec.x + float(offset[0]),
        y=spec.y + float(offset[1]),
        z=spec.z + float(offset[2]),
    )

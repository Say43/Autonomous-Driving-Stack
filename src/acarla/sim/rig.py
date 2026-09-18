"""Camera rig configuration: parsing rig YAML files.

This module deliberately has no dependency on `carla` at import time, so
that rig YAML files can be validated from the carla-less `.venv`
(Python 3.13). Only the functions that actually spawn CARLA sensors (in
`acarla.sim.client`) import `carla`, and they import it locally rather than
at module scope.

Two YAML schemas are supported, both a mapping under `cameras:` from camera
name to per-camera data:

- The provisional format (`configs/rig_provisional.yaml`, M4 placeholder):
  `fov`, `width`, `height`, `sensor_tick`, and `transform.{location,rotation}`
  directly at the camera level, in CARLA convention.
- The M1-calibrated format (`configs/rig_alpamayo.yaml`): each camera has a
  `carla:` section (`location`, `rotation`, `render_hfov_deg`, `render_size`)
  that maps onto the same CARLA-convention fields -- `fov` <-
  `render_hfov_deg`, `width`/`height` <- `render_size`, `sensor_tick` is
  fixed at 0.1s (the model's implicit 10 Hz grid, see
  `acarla.types.SAMPLE_DT`) -- plus an `intrinsics` section (F-Theta
  polynomials) consumed by `load_rig_intrinsics`, and a top-level `vehicle`
  section consumed by `load_rig_vehicle`.

Camera names must be keys of `acarla.types.ALPAMAYO_CAMERA_INDICES` --
that is the actual model contract, not a convenience naming choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from acarla.adapter.lens import FThetaModel
from acarla.types import ALPAMAYO_CAMERA_INDICES, alpamayo_camera_order

_CALIBRATED_SENSOR_TICK = 0.1
"""Fixed sensor_tick for the M1-calibrated rig format -- the model's
implicit 10 Hz grid (`acarla.types.SAMPLE_DT`), not read from the YAML."""


@dataclass(frozen=True)
class CameraSpec:
    """One camera's placement and sensor parameters, in CARLA convention."""

    name: str
    fov: float
    width: int
    height: int
    sensor_tick: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


def _parse_camera(name: str, d: dict[str, Any]) -> CameraSpec:
    if "carla" in d:
        return _parse_camera_calibrated(name, d["carla"])
    return _parse_camera_provisional(name, d)


def _parse_camera_provisional(name: str, d: dict[str, Any]) -> CameraSpec:
    required_top = ("fov", "width", "height", "sensor_tick", "transform")
    missing = [k for k in required_top if k not in d]
    if missing:
        raise ValueError(f"cameras[{name!r}]: missing required key(s) {missing}")

    transform = d["transform"]
    for section in ("location", "rotation"):
        if section not in transform:
            raise ValueError(f"cameras[{name!r}].transform: missing {section!r}")

    location = transform["location"]
    rotation = transform["rotation"]
    for axis in ("x", "y", "z"):
        if axis not in location:
            raise ValueError(f"cameras[{name!r}].transform.location: missing {axis!r}")
    for axis in ("roll", "pitch", "yaw"):
        if axis not in rotation:
            raise ValueError(f"cameras[{name!r}].transform.rotation: missing {axis!r}")

    return CameraSpec(
        name=name,
        fov=float(d["fov"]),
        width=int(d["width"]),
        height=int(d["height"]),
        sensor_tick=float(d["sensor_tick"]),
        x=float(location["x"]),
        y=float(location["y"]),
        z=float(location["z"]),
        roll=float(rotation["roll"]),
        pitch=float(rotation["pitch"]),
        yaw=float(rotation["yaw"]),
    )


def _parse_camera_calibrated(name: str, carla_section: dict[str, Any]) -> CameraSpec:
    required = ("location", "rotation", "render_hfov_deg", "render_size")
    missing = [k for k in required if k not in carla_section]
    if missing:
        raise ValueError(f"cameras[{name!r}].carla: missing required key(s) {missing}")

    location = carla_section["location"]
    rotation = carla_section["rotation"]
    for axis in ("x", "y", "z"):
        if axis not in location:
            raise ValueError(f"cameras[{name!r}].carla.location: missing {axis!r}")
    for axis in ("roll", "pitch", "yaw"):
        if axis not in rotation:
            raise ValueError(f"cameras[{name!r}].carla.rotation: missing {axis!r}")

    render_size = carla_section["render_size"]
    if len(render_size) != 2:
        raise ValueError(
            f"cameras[{name!r}].carla.render_size: expected [width, height], "
            f"got {render_size!r}"
        )
    width, height = render_size

    return CameraSpec(
        name=name,
        fov=float(carla_section["render_hfov_deg"]),
        width=int(width),
        height=int(height),
        sensor_tick=_CALIBRATED_SENSOR_TICK,
        x=float(location["x"]),
        y=float(location["y"]),
        z=float(location["z"]),
        roll=float(rotation["roll"]),
        pitch=float(rotation["pitch"]),
        yaw=float(rotation["yaw"]),
    )


def load_rig_config(
    path: str | Path, cameras: list[str] | None = None
) -> list[CameraSpec]:
    """Parse a rig YAML file into a list of `CameraSpec`.

    `cameras`, if given, restricts (and does not reorder) the result to that
    subset -- this is the CLI's `--cameras` selection mechanism. Every name
    in `cameras` must both exist in the YAML file and be a known Alpamayo
    camera name. Raises `ValueError` on any structural or naming problem,
    naming the offending key and the allowed values.

    The returned list is ordered by ascending Alpamayo camera index
    (`acarla.types.alpamayo_camera_order`), matching the model's own camera
    ordering contract -- this is also the order written into
    `RunHeader.cameras`.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "cameras" not in doc:
        raise ValueError(f"{path}: expected a top-level 'cameras' mapping")

    raw_cameras = doc["cameras"]
    if not isinstance(raw_cameras, dict) or not raw_cameras:
        raise ValueError(f"{path}: 'cameras' must be a non-empty mapping")

    unknown_in_file = sorted(set(raw_cameras) - set(ALPAMAYO_CAMERA_INDICES))
    if unknown_in_file:
        raise ValueError(
            f"{path}: cameras {unknown_in_file} are not known Alpamayo camera "
            f"names (allowed: {sorted(ALPAMAYO_CAMERA_INDICES)})"
        )

    if cameras is not None:
        unknown_requested = sorted(set(cameras) - set(ALPAMAYO_CAMERA_INDICES))
        if unknown_requested:
            raise ValueError(
                f"requested cameras {unknown_requested} are not known Alpamayo "
                f"camera names (allowed: {sorted(ALPAMAYO_CAMERA_INDICES)})"
            )
        missing_from_file = sorted(set(cameras) - set(raw_cameras))
        if missing_from_file:
            raise ValueError(
                f"{path}: requested cameras {missing_from_file} are not defined "
                f"in this rig file (defined: {sorted(raw_cameras)})"
            )
        selected_names = set(cameras)
    else:
        selected_names = set(raw_cameras)

    specs = [_parse_camera(name, raw_cameras[name]) for name in selected_names]
    ordered_names = alpamayo_camera_order(s.name for s in specs)
    by_name = {s.name: s for s in specs}
    return [by_name[name] for name in ordered_names]


def load_rig_intrinsics(path: str | Path) -> dict[str, FThetaModel]:
    """Parse a rig YAML file's per-camera `intrinsics` sections into
    `FThetaModel` instances, keyed by camera name.

    Only the M1-calibrated rig format (`configs/rig_alpamayo.yaml`) has an
    `intrinsics` section; raises `ValueError` naming the camera if it is
    missing, or if `intrinsics.model` is not `"ftheta"`.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "cameras" not in doc:
        raise ValueError(f"{path}: expected a top-level 'cameras' mapping")

    raw_cameras = doc["cameras"]
    result: dict[str, FThetaModel] = {}
    for name, d in raw_cameras.items():
        if "intrinsics" not in d:
            raise ValueError(f"{path}: cameras[{name!r}] has no 'intrinsics' section")
        intr = d["intrinsics"]
        model = intr.get("model")
        if model != "ftheta":
            raise ValueError(
                f"{path}: cameras[{name!r}].intrinsics.model = {model!r}, "
                "only 'ftheta' is supported"
            )
        result[name] = FThetaModel(
            width=int(intr["width"]),
            height=int(intr["height"]),
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            fw_coef=intr["fw_poly"],
            bw_coef=intr["bw_poly"],
        )
    return result


def load_rig_vehicle(path: str | Path) -> dict[str, Any]:
    """Return the top-level `vehicle` mapping from a rig YAML file as-is.

    Raises `ValueError` if the file has no top-level `vehicle` section.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "vehicle" not in doc:
        raise ValueError(f"{path}: expected a top-level 'vehicle' mapping")
    return dict(doc["vehicle"])

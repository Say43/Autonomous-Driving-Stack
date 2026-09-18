#!/usr/bin/env python
"""Build `configs/rig_alpamayo.yaml` from the golden M1 calibration CSVs.

Usage:
    python scripts/calibrate_rig.py \
        --calibration-dir golden/m2_reference --out configs/rig_alpamayo.yaml

Reads the three golden calibration CSVs (sensor extrinsics, camera
intrinsics, vehicle dimensions) and writes, for each of the four Alpamayo
cameras (`ALPAMAYO_CAMERA_INDICES` in `src/acarla/types.py`), its raw
intrinsics/extrinsics plus the derived CARLA transform and render FOV.
Every number in the output either comes straight from the CSVs or is a
deterministic computation on them (`src/acarla/sim/coords.py`,
`src/acarla/adapter/lens.py`) -- nothing is estimated or guessed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np
import yaml


def _to_plain(obj):
    """Recursively convert numpy scalars/arrays to plain Python types so
    `yaml.safe_dump` can serialize them."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_plain(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acarla.adapter.lens import FThetaModel, max_theta_deg, required_pinhole_hfov_deg  # noqa: E402
from acarla.sim import coords  # noqa: E402
from acarla.types import ALPAMAYO_CAMERA_INDICES  # noqa: E402

# Maps ALPAMAYO_CAMERA_INDICES keys to the dataset's sensor_name / feature
# naming in the golden CSVs.
ALPAMAYO_TO_DATASET = {
    "cross_left": "camera_cross_left_120fov",
    "front_wide": "camera_front_wide_120fov",
    "cross_right": "camera_cross_right_120fov",
    "front_tele": "camera_front_tele_30fov",
}


def _read_csv_rows(path: Path, key_column: str) -> dict[str, dict[str, str]]:
    with open(path, newline="") as f:
        return {row[key_column]: row for row in csv.DictReader(f)}


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


def build_rig_config(calibration_dir: Path, clip_id: str | None = None) -> dict:
    extrinsics = _read_csv_rows(
        calibration_dir / "calibration_sensor_extrinsics.csv", "sensor_name"
    )
    intrinsics = _read_csv_rows(
        calibration_dir / "calibration_camera_intrinsics.csv", "camera_name"
    )
    with open(calibration_dir / "calibration_vehicle_dimensions.csv", newline="") as f:
        vehicle_rows = list(csv.DictReader(f))
    vehicle_row = vehicle_rows[0]

    cameras: dict[str, dict] = {}
    for cam_key, dataset_name in ALPAMAYO_TO_DATASET.items():
        ext = extrinsics[dataset_name]
        intr = intrinsics[dataset_name]

        qx, qy, qz, qw = (float(ext["qx"]), float(ext["qy"]), float(ext["qz"]), float(ext["qw"]))
        t_rig = np.array([float(ext["x"]), float(ext["y"]), float(ext["z"])])
        r_opt2rig = _quat_xyzw_to_matrix(qx, qy, qz, qw)

        width = int(intr["width"])
        height = int(intr["height"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
        fw_poly = [float(intr[f"fw_poly_{i}"]) for i in range(5)]
        bw_poly = [float(intr[f"bw_poly_{i}"]) for i in range(5)]

        ftheta = FThetaModel(
            width=width, height=height, cx=cx, cy=cy,
            fw_coef=np.array(fw_poly), bw_coef=np.array(bw_poly),
        )
        theta_max = max_theta_deg(ftheta)
        render_hfov = required_pinhole_hfov_deg(ftheta, margin_deg=2.0)

        loc_carla, (pitch, yaw, roll) = coords.camera_extrinsics_to_carla_transform(
            r_opt2rig, t_rig
        )

        cameras[cam_key] = {
            "dataset_id": dataset_name,
            "alpamayo_index": ALPAMAYO_CAMERA_INDICES[cam_key],
            "prompt_name": cam_key,
            "intrinsics": {
                "width": width,
                "height": height,
                "cx": cx,
                "cy": cy,
                "fw_poly": fw_poly,
                "bw_poly": bw_poly,
                "model": "ftheta",
                "max_theta_deg": theta_max,
            },
            "extrinsics_rig": {
                "quat_xyzw": [qx, qy, qz, qw],
                "translation": [float(ext["x"]), float(ext["y"]), float(ext["z"])],
                "frame": "optical_opencv",
            },
            "carla": {
                "location": {
                    "x": float(loc_carla[0]),
                    "y": float(loc_carla[1]),
                    "z": float(loc_carla[2]),
                },
                "rotation": {"pitch": pitch, "yaw": yaw, "roll": roll},
                "render_hfov_deg": render_hfov,
                "render_size": [1920, 1080],
                "attach_note": (
                    "Location/rotation are relative to whatever origin the rig "
                    "translations above are relative to (see rig_origin_note); "
                    "CARLA attaches sensors relative to the vehicle actor's "
                    "bounding-box center, and the offset between that and the "
                    "rig origin is an open issue -- see "
                    "carla_attach_offset_open_issue."
                ),
            },
        }

    vehicle = {
        "length": float(vehicle_row["length"]),
        "width": float(vehicle_row["width"]),
        "height": float(vehicle_row["height"]),
        "rear_axle_to_bbox_center": float(vehicle_row["rear_axle_to_bbox_center"]),
        "wheelbase": float(vehicle_row["wheelbase"]),
        "track_width": float(vehicle_row["track_width"]),
    }

    config = {
        "vehicle": vehicle,
        "rig_origin_note": (
            "HYPOTHESIS, not verified: the rig origin (the (0,0,0) that all "
            "sensor translations above are given relative to) is plausibly the "
            "rear axle at ground height. Evidence: camera x-translations range "
            "1.7-2.5 m forward and z-translations 0.9-1.4 m up, which is "
            "consistent with cameras mounted on/above a windshield/roof measured "
            "from a rear-axle-at-ground origin; and the vehicle's own "
            f"rear_axle_to_bbox_center = {vehicle['rear_axle_to_bbox_center']} m "
            "is a plausible complementary offset if the bbox center is used as "
            "an alternative origin. This has NOT been confirmed against any "
            "oracle -- it is a hypothesis for downstream milestones to verify or "
            "falsify."
        ),
        "carla_attach_offset_open_issue": True,
        "cameras": cameras,
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = _to_plain(build_rig_config(args.calibration_dir))

    meta_path = args.calibration_dir / "meta.json"
    clip_id = None
    if meta_path.exists():
        import json

        with open(meta_path) as f:
            clip_id = json.load(f).get("clip_id")

    header = (
        "# GENERATED FILE -- do not edit by hand.\n"
        f"# Generated by scripts/calibrate_rig.py on "
        f"{datetime.datetime.now(datetime.UTC).isoformat()}\n"
        f"# Source: {args.calibration_dir}"
        + (f" (clip_id={clip_id})" if clip_id else "")
        + "\n"
        "# Replaces configs/rig_provisional.yaml as the M1 rig-calibration result.\n"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

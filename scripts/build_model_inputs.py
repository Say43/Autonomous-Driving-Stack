#!/usr/bin/env python3
"""Build Alpamayo model inputs (a stack of `SensorPacket`s) from a recorded
CARLA run's `trace.jsonl`.

Walks the trace, picks every `--every`-th image tick (a `TraceFrame` whose
`image_paths` is non-empty) once enough history exists (16 ego poses on the
model's implicit 0.1s grid, and 4 image ticks per camera), loads the
recorded PNGs, remaps them from CARLA's pinhole render through each
camera's F-Theta lens model (`acarla.adapter.lens.build_remap` /
`apply_remap`), builds one `SensorPacket` per selected tick via
`acarla.adapter.packet.build_sensor_packet`, converts each to the model's
raw numpy inputs via `acarla.model.inputs.sensor_packet_to_model_inputs`,
and stacks the result into a single `.npz` file plus a `meta.json`
sidecar.

This script depends on Pillow (`PIL.Image`) to decode the recorded PNGs,
which is not yet a declared dependency of this project (see the note this
script prints if Pillow is missing).

Usage:
    python scripts/build_model_inputs.py --run results/some_run \\
        --out results/some_run/model_inputs.npz --every 10 \\
        --rig configs/rig_alpamayo.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - reported to the user, not tested
    raise SystemExit(
        "build_model_inputs.py requires Pillow ('pip install pillow'), which "
        "is not currently declared as a dependency in pyproject.toml -- add "
        "it there once this script is adopted into the normal workflow."
    ) from exc

from scipy.spatial.transform import Rotation  # noqa: E402

from acarla.adapter.lens import PinholeModel, apply_remap, build_remap  # noqa: E402
from acarla.adapter.packet import build_sensor_packet  # noqa: E402
from acarla.model.inputs import sensor_packet_to_model_inputs  # noqa: E402
from acarla.record.reader import read_trace  # noqa: E402
from acarla.sim.coords import carla_rotation_to_rig_rotation, carla_to_rig_xyz  # noqa: E402
from acarla.sim.rig import load_rig_config, load_rig_intrinsics  # noqa: E402
from acarla.types import N_EGO_WAYPOINTS, N_HISTORY_FRAMES  # noqa: E402

MAX_OUTPUT_BYTES = 2 * 1024**3
"""Refuse to write an .npz larger than this (2 GiB)."""

_EGO_STRIDE = 2
"""Trace ticks between consecutive 0.1s-grid ego samples -- the trace runs
at half the model's 10 Hz grid (see SAMPLE_DT), so every 2nd trace tick
lands on that grid."""


def _load_and_remap_image(
    run_dir: Path, rel_path: str, ftheta, pinhole, map_x, map_y
) -> np.ndarray:
    img = Image.open(run_dir / rel_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape[:2] != (pinhole.height, pinhole.width):
        raise ValueError(
            f"{run_dir / rel_path}: expected size {pinhole.width}x{pinhole.height}, "
            f"got {arr.shape[1]}x{arr.shape[0]}"
        )
    return apply_remap(arr, map_x, map_y)


def build_model_inputs(
    run_dir: Path, rig_path: Path, every: int, max_packets: int | None
) -> tuple[dict[str, np.ndarray], dict]:
    specs = load_rig_config(rig_path)
    ftheta_models = load_rig_intrinsics(rig_path)
    camera_names = [s.name for s in specs]
    pinholes: dict[str, PinholeModel] = {}
    remaps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        ftheta = ftheta_models[spec.name]
        pinhole = PinholeModel(width=spec.width, height=spec.height, hfov_deg=spec.fov)
        pinholes[spec.name] = pinhole
        remaps[spec.name] = build_remap(ftheta, pinhole)

    trace_path = run_dir / "trace.jsonl"
    header, frame_iter = read_trace(trace_path)
    frames = list(frame_iter)

    image_tick_indices = [i for i, f in enumerate(frames) if f.image_paths]

    per_packet_inputs: list[dict[str, np.ndarray]] = []
    frame_ids: list[int] = []
    sim_times: list[float] = []
    camera_indices_ref: np.ndarray | None = None

    n_built = 0
    for pos, idx in enumerate(image_tick_indices):
        if pos < N_HISTORY_FRAMES - 1:
            continue  # not enough prior image ticks yet
        ego_start = idx - (N_EGO_WAYPOINTS - 1) * _EGO_STRIDE
        if ego_start < 0:
            continue  # not enough prior trace ticks yet

        if (pos - (N_HISTORY_FRAMES - 1)) % every != 0:
            continue

        if max_packets is not None and n_built >= max_packets:
            break

        image_tick_positions = image_tick_indices[pos - N_HISTORY_FRAMES + 1 : pos + 1]

        frames_hwc: dict[str, np.ndarray] = {}
        image_timestamps: dict[str, list[float]] = {}
        for cam in camera_names:
            map_x, map_y = remaps[cam]
            ftheta = ftheta_models[cam]
            pinhole = pinholes[cam]
            imgs = []
            ts = []
            for tick_idx in image_tick_positions:
                frame = frames[tick_idx]
                rel_path = frame.image_paths[cam][0]
                imgs.append(
                    _load_and_remap_image(run_dir, rel_path, ftheta, pinhole, map_x, map_y)
                )
                ts.append(round(frame.sim_time, 6))
            frames_hwc[cam] = np.stack(imgs, axis=0)
            image_timestamps[cam] = ts

        ego_tick_indices = [
            idx - k * _EGO_STRIDE for k in range(N_EGO_WAYPOINTS - 1, -1, -1)
        ]
        world_xyz = np.zeros((N_EGO_WAYPOINTS, 3), dtype=np.float64)
        world_quat_xyzw = np.zeros((N_EGO_WAYPOINTS, 4), dtype=np.float64)
        ego_timestamps: list[float] = []
        for row, tick_idx in enumerate(ego_tick_indices):
            pose = frames[tick_idx].ego_pose_world
            world_xyz[row] = carla_to_rig_xyz(pose.translation.astype(np.float64))
            r_rig = carla_rotation_to_rig_rotation(pose.rotation.astype(np.float64))
            world_quat_xyzw[row] = Rotation.from_matrix(r_rig).as_quat()
            ego_timestamps.append(round(frames[tick_idx].sim_time, 6))

        current_frame = frames[idx]
        packet = build_sensor_packet(
            frame_id=current_frame.frame_id,
            sim_time=current_frame.sim_time,
            frames_hwc=frames_hwc,
            image_timestamps=image_timestamps,
            world_xyz=world_xyz,
            world_quat_xyzw=world_quat_xyzw,
            ego_timestamps=ego_timestamps,
        )
        model_inputs = sensor_packet_to_model_inputs(packet)

        if camera_indices_ref is None:
            camera_indices_ref = model_inputs["camera_indices"]

        per_packet_inputs.append(model_inputs)
        frame_ids.append(current_frame.frame_id)
        sim_times.append(current_frame.sim_time)
        n_built += 1

    if not per_packet_inputs:
        raise ValueError(
            f"{trace_path}: no image tick had enough history "
            f"({N_HISTORY_FRAMES} image ticks, {N_EGO_WAYPOINTS} ego poses on "
            "the 0.1s grid) to build a single packet"
        )

    image_frames = np.stack([p["image_frames"] for p in per_packet_inputs], axis=0)
    ego_history_xyz = np.stack([p["ego_history_xyz"] for p in per_packet_inputs], axis=0)
    ego_history_rot = np.stack([p["ego_history_rot"] for p in per_packet_inputs], axis=0)

    out_arrays = {
        "image_frames": image_frames,
        "camera_indices": camera_indices_ref,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "sim_times": np.array(sim_times, dtype=np.float64),
    }

    total_bytes = sum(arr.nbytes for arr in out_arrays.values())
    meta = {
        "rig_path": str(rig_path),
        "run_header": header.to_json_dict(),
        "n_packets": len(per_packet_inputs),
        "total_bytes": total_bytes,
        "camera_names": camera_names,
    }
    return out_arrays, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="recorded run directory")
    parser.add_argument("--out", required=True, type=Path, help="output .npz path")
    parser.add_argument(
        "--every", type=int, default=1, help="use every Nth eligible image tick"
    )
    parser.add_argument(
        "--max-packets", type=int, default=None, help="cap the number of packets built"
    )
    parser.add_argument(
        "--rig",
        type=Path,
        default=ROOT / "configs" / "rig_alpamayo.yaml",
        help="rig YAML file",
    )
    args = parser.parse_args(argv)

    if args.every < 1:
        parser.error("--every must be >= 1")

    out_arrays, meta = build_model_inputs(args.run, args.rig, args.every, args.max_packets)

    if meta["total_bytes"] > MAX_OUTPUT_BYTES:
        raise SystemExit(
            f"refusing to write {meta['total_bytes']} bytes "
            f"(> {MAX_OUTPUT_BYTES} byte limit) -- use --every/--max-packets "
            "to shrink the output"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out_arrays)

    meta_path = args.out.parent / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(
        f"wrote {args.out} ({meta['n_packets']} packets, {meta['total_bytes']} bytes) "
        f"and {meta_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

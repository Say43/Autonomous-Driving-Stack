#!/usr/bin/env python3
"""Render a reproducible pinhole/F-Theta comparison from a CARLA trace.

The source frame is read from ``trace.jsonl`` rather than chosen by file
name, so the preview records exactly which simulation frame and camera were
used. Two PNGs plus a small JSON provenance sidecar are written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.adapter.lens import PinholeModel, apply_remap, build_remap  # noqa: E402
from acarla.record.reader import read_trace  # noqa: E402
from acarla.sim.rig import load_rig_config, load_rig_intrinsics  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="recorded run directory")
    parser.add_argument("--out", required=True, type=Path, help="preview output directory")
    parser.add_argument("--camera", default="front_wide")
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="specific image-bearing frame; defaults to the middle available frame",
    )
    parser.add_argument(
        "--rig",
        type=Path,
        default=ROOT / "configs" / "rig_alpamayo.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = {spec.name: spec for spec in load_rig_config(args.rig)}
    intrinsics = load_rig_intrinsics(args.rig)
    if args.camera not in specs or args.camera not in intrinsics:
        available = sorted(set(specs) & set(intrinsics))
        raise SystemExit(f"unknown camera {args.camera!r}; available: {available}")

    _, frame_iter = read_trace(args.run / "trace.jsonl")
    candidates = [frame for frame in frame_iter if args.camera in frame.image_paths]
    if not candidates:
        raise SystemExit(f"trace contains no images for camera {args.camera!r}")
    if args.frame_id is None:
        frame = candidates[len(candidates) // 2]
    else:
        matching = [frame for frame in candidates if frame.frame_id == args.frame_id]
        if not matching:
            raise SystemExit(
                f"frame {args.frame_id} has no image for camera {args.camera!r}"
            )
        frame = matching[0]

    rel_path = frame.image_paths[args.camera][0]
    source_path = args.run / rel_path
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)

    spec = specs[args.camera]
    if source.shape != (spec.height, spec.width, 3):
        raise SystemExit(
            f"{source_path}: expected {(spec.height, spec.width, 3)}, got {source.shape}"
        )
    ftheta = intrinsics[args.camera]
    pinhole = PinholeModel(width=spec.width, height=spec.height, hfov_deg=spec.fov)
    map_x, map_y = build_remap(ftheta, pinhole)
    remapped = apply_remap(source, map_x, map_y)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"m4_{args.camera}"
    pinhole_path = args.out / f"{stem}_pinhole.png"
    ftheta_path = args.out / f"{stem}_ftheta.png"
    meta_path = args.out / f"{stem}_preview.json"
    Image.fromarray(source).save(pinhole_path)
    Image.fromarray(remapped).save(ftheta_path)
    meta_path.write_text(
        json.dumps(
            {
                "camera": args.camera,
                "frame_id": frame.frame_id,
                "sim_time": frame.sim_time,
                "source_image": rel_path,
                "source_hfov_deg": spec.fov,
                "rig_path": str(args.rig),
                "pinhole_output": pinhole_path.name,
                "ftheta_output": ftheta_path.name,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {pinhole_path}, {ftheta_path}, and {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

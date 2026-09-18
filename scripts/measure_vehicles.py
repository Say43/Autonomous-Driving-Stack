#!/usr/bin/env python3
"""M4 Paket 2: measure every CARLA vehicle blueprint's bounding box against
the live server, to pick which real vehicle best matches the M1-calibrated
rig's reference vehicle (`configs/rig_alpamayo.yaml: vehicle:`, length
the rig file) and to determine, empirically, whether a CARLA
vehicle actor's own transform origin sits at ground height or at the
bounding-box center.

Must run from `.venv-sim` (Python 3.12, has `carla`) against a running
CARLA server. Spawns each `vehicle.*` blueprint one at a time at a free
spawn point (`try_spawn_actor`, skipping failures), reads
`bounding_box.extent` (half-extents) and `bounding_box.location` (box
center relative to the actor's own origin), destroys it immediately, and
moves on. Two-wheeled vehicles (motorcycles/bicycles) are skipped.

Output: a Markdown table to stdout and appended into `docs/m4_real_rig.md`
(replacing any previous "## Vehicle measurements" section written by this
script), sorted by closeness to the reference vehicle.

Usage:
    .venv-sim\\Scripts\\python.exe scripts\\measure_vehicles.py \\
        --host 127.0.0.1 --port 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import carla  # noqa: E402

# Reference vehicle dimensions are read from the rig file's `vehicle:` section at runtime
# (they originate from the dataset calibration and are not hard-coded here).

SECTION_HEADER = "## Vehicle measurements"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig", type=Path, default=ROOT / "configs" / "rig_alpamayo.yaml",
        help="rig YAML whose `vehicle:` section holds the reference dimensions and whose "
             "front_wide camera height is used for the plausibility note",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs" / "m4_real_rig.md",
        help="markdown file to append the measurement table into",
    )
    return parser.parse_args(argv)


def measure_all(client: carla.Client, world: carla.World) -> list[dict]:
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("map has no spawn points")

    results: list[dict] = []
    for bp in blueprint_library.filter("vehicle.*"):
        n_wheels = 4
        if bp.has_attribute("number_of_wheels"):
            try:
                n_wheels = int(bp.get_attribute("number_of_wheels"))
            except ValueError:
                n_wheels = 4
        if n_wheels != 4:
            continue  # skip motorcycles/bicycles

        actor = None
        for sp in spawn_points:
            actor = world.try_spawn_actor(bp, sp)
            if actor is not None:
                break
        if actor is None:
            continue  # this blueprint didn't spawn anywhere -- skip

        try:
            bbox = actor.bounding_box
            extent = bbox.extent
            location = bbox.location
            results.append(
                {
                    "blueprint": bp.id,
                    "length": 2.0 * extent.x,
                    "width": 2.0 * extent.y,
                    "height": 2.0 * extent.z,
                    "extent_z": extent.z,
                    "bbox_location": (location.x, location.y, location.z),
                }
            )
        finally:
            try:
                actor.destroy()
            except Exception:  # noqa: BLE001
                pass

    return results


def rank(results: list[dict], ref_length_m: float, ref_width_m: float) -> list[dict]:
    def score(r: dict) -> float:
        return abs(r["length"] - ref_length_m) + abs(r["width"] - ref_width_m)

    return sorted(results, key=score)


def render_table(ranked: list[dict]) -> str:
    lines = [
        "| blueprint | length (m) | width (m) | height (m) | bbox_location (x,y,z) |",
        "|---|---|---|---|---|",
    ]
    for r in ranked:
        loc = r["bbox_location"]
        lines.append(
            f"| {r['blueprint']} | {r['length']:.4f} | {r['width']:.4f} | "
            f"{r['height']:.4f} | ({loc[0]:.4f}, {loc[1]:.4f}, {loc[2]:.4f}) |"
        )
    return "\n".join(lines)


def render_section(ranked: list[dict], ref_length_m: float, ref_width_m: float, ref_height_m: float, front_wide_z: float) -> str:
    table = render_table(ranked)
    best3 = ranked[:3]
    best_lines = "\n".join(
        f"{i + 1}. `{r['blueprint']}` -- length {r['length']:.4f} m, "
        f"width {r['width']:.4f} m, bbox_location.z {r['bbox_location'][2]:.4f} m, "
        f"extent.z {r['extent_z']:.4f} m"
        for i, r in enumerate(best3)
    )

    top = ranked[0]
    loc_z = top["bbox_location"][2]
    ez = top["extent_z"]
    if abs(loc_z - ez) < abs(loc_z - 0.0):
        origin_hypothesis = (
            f"bbox_location.z ({loc_z:.4f}) is close to +extent.z ({ez:.4f}), "
            "not to 0 -- the actor's own transform origin sits at the "
            "vehicle's GROUND level (the bbox center is a full half-height "
            "above the origin)."
        )
    else:
        origin_hypothesis = (
            f"bbox_location.z ({loc_z:.4f}) is close to 0, not to +extent.z "
            f"({ez:.4f}) -- the actor's own transform origin sits at the "
            "BOUNDING-BOX CENTER, not at ground level."
        )

    plausibility = (
        f"Plausibility check for `front_wide` (rig-frame z = {front_wide_z:.7f} m above "
        "the rig origin, from `configs/rig_alpamayo.yaml`): if the rig origin "
        "coincides with the actor's ground-level transform origin, the camera "
        f"sits {front_wide_z:.4f} m above the road -- plausible for a reference vehicle "
        f"{ref_length_m:.3f} m long / {ref_height_m:.3f} m tall (windshield/roof height). "
        "If instead the rig origin were at rear-axle height (~0.35 m above "
        "ground, not ground itself), the camera would sit at "
        f"~{front_wide_z:.4f} + 0.35 = {front_wide_z + 0.35:.2f} m -- above the vehicle's own {ref_height_m:.3f} m roof, "
        "which is implausible for a windshield-mounted camera. This supports "
        "the rear-axle-AT-GROUND-height hypothesis in `src/acarla/sim/attach.py`."
    )

    return (
        f"{SECTION_HEADER}\n\n"
        f"Reference vehicle (from `configs/rig_alpamayo.yaml: vehicle:`): "
        f"length {ref_length_m} m, width {ref_width_m} m.\n\n"
        f"{table}\n\n"
        f"### Best 3 matches (by |length diff| + |width diff|)\n\n"
        f"{best_lines}\n\n"
        f"### Actor-origin measurement\n\n"
        f"For the best match, {origin_hypothesis}\n\n"
        f"{plausibility}\n"
    )


def _splice_section(doc_path: Path, section_text: str) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    if SECTION_HEADER in existing:
        before, _, rest = existing.partition(SECTION_HEADER)
        # Find the next top-level "## " heading after this section, if any.
        rest_after_header = rest[len(SECTION_HEADER):]
        next_idx = rest_after_header.find("\n## ")
        tail = rest_after_header[next_idx:] if next_idx != -1 else ""
        new_content = before.rstrip("\n") + "\n\n" + section_text.rstrip("\n") + "\n" + tail
    else:
        sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
        new_content = existing + sep + section_text.rstrip("\n") + "\n"
    doc_path.write_text(new_content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    from acarla.sim.rig import load_rig_config, load_rig_vehicle  # noqa: E402

    vehicle = load_rig_vehicle(args.rig)
    front_wide = [c for c in load_rig_config(args.rig) if c.name == "front_wide"][0]
    results = measure_all(client, world)
    ranked = rank(results, float(vehicle["length"]), float(vehicle["width"]))

    section_text = render_section(
        ranked, float(vehicle["length"]), float(vehicle["width"]), float(vehicle["height"]), float(front_wide.z)
    )
    print(section_text)

    _splice_section(args.doc, section_text)
    print(f"\nwrote measurements into {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""M4: CARLA open-loop recording.

Drives the ego vehicle via CARLA's traffic-manager Autopilot, ticks the
world synchronously, and records ground truth (actors, lanes, traffic
lights, ego pose, control) plus (optionally) camera images to a JSONL trace
compatible with `acarla.record`. No model runs -- `TraceFrame.plan` is
always `None`. This script must run from `.venv-sim` (Python 3.12, has
`carla`); `acarla.record`/`acarla.sim.rig` alone stay importable from the
carla-less `.venv` and are unit-tested there.

Usage:
    python scripts/run_open_loop.py --out results/m4_smoke --ticks 100 \\
        --seed 1 --traffic 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acarla.control.frames import offset_pose  # noqa: E402
from acarla.record.writer import TraceWriter  # noqa: E402
from acarla.sim import groundtruth  # noqa: E402
from acarla.sim.attach import apply_offset, rig_to_actor_offset  # noqa: E402
from acarla.sim.client import SimSession  # noqa: E402
from acarla.sim.rig import load_rig_config, load_rig_vehicle  # noqa: E402
from acarla.types import ControlCommand, RunHeader, TraceFrame  # noqa: E402

DEFAULT_RIG_PATH = ROOT / "configs" / "rig_alpamayo.yaml"
DEFAULT_MAP = "Town01"
GROUND_TRUTH_RADIUS_M = 60.0

DEFAULT_VEHICLE_BLUEPRINT = "vehicle.lincoln.mkz_2017"
# Empirische starre Verschiebung des GESAMTEN Rigs entlang +x, pro CARLA-Fahrzeugmesh. Die
# Kalibrierung setzt front_wide 0.34 m vor das Bounding-Box-Zentrum; beim Lincoln-Mesh liegt die
# Kamera dort mitten in der Kabine (Lenkrad im Bild). Probe-Renders (docs/m4_real_rig.md) zeigen
# ab +0.7 m nur noch Motorhaube wie in der Referenzaufnahme; alle vier Kameras bleiben frei vom
# Karosserie-Mesh. Getrennt vom gemessenen Achsen-Offset gehalten: das hier ist Anpassung an das
# Fahrzeugmodell, keine Kalibrierung. Relative Kamerageometrie bleibt exakt erhalten.
VEHICLE_MESH_SHIFT_X = {"vehicle.lincoln.mkz_2017": 0.7}
"""Best real-vehicle match for the M1-calibrated rig's reference vehicle
(`configs/rig_alpamayo.yaml: vehicle:`, dataset-derived and therefore not in the public repo),
per `scripts/measure_vehicles.py` run against the live CARLA 0.9.16
blueprint library (see `docs/m4_real_rig.md`, "Vehicle measurements"):
length 4.9017 m, width 2.1283 m -- closest of all four-wheeled blueprints
by |length diff| + |width diff|."""


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--ticks", type=int, default=100, help="number of simulation ticks")
    parser.add_argument("--seed", type=int, required=True, help="world + traffic manager seed")
    parser.add_argument("--map", default=DEFAULT_MAP, help="CARLA map to load (e.g. Town01)")
    parser.add_argument(
        "--cameras",
        default="front_wide,front_tele,cross_left,cross_right",
        help="comma-separated subset of rig camera names to use",
    )
    parser.add_argument(
        "--rig", type=Path, default=DEFAULT_RIG_PATH, help="path to the rig YAML file"
    )
    parser.add_argument("--traffic", type=int, default=0, help="number of background vehicles")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="record ground truth only, skip camera spawning/capture (fast determinism checks)",
    )
    parser.add_argument(
        "--vehicle",
        default=DEFAULT_VEHICLE_BLUEPRINT,
        help=(
            "ego vehicle blueprint id (default: the real vehicle closest to "
            "the M1-calibrated rig's reference vehicle, see "
            "docs/m4_real_rig.md)"
        ),
    )
    parser.add_argument(
        "--attach-offset",
        choices=("auto", "none"),
        default="auto",
        help=(
            "'auto' (default): measure the spawned ego's bounding_box.extent/"
            "location and apply rig_to_actor_offset()/apply_offset() (see "
            "src/acarla/sim/attach.py) to every camera before attaching. "
            "'none': attach cameras at the rig-frame location/rotation "
            "verbatim, no offset."
        ),
    )
    parser.add_argument(
        "--rig-shift-x",
        type=float,
        default=None,
        help=(
            "Starre Verschiebung des ganzen Rigs entlang +x (m) zusaetzlich zum Achsen-Offset. "
            "Default: Wert aus VEHICLE_MESH_SHIFT_X fuer das gewaehlte Fahrzeug, sonst 0."
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help=(
            "png: CARLA save_to_disk (verlustfrei, ~5 MB/Bild). jpg: PIL aus dem Rohpuffer, "
            "Qualitaet 92, ~0.3 MB/Bild -- fuer laengere Laeufe. Die echten Trainingsbilder sind "
            "H.264-dekodiert, also ebenfalls verlustbehaftet."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0, help="CARLA client timeout, seconds")
    parser.add_argument(
        "--purge-actors",
        action="store_true",
        help=(
            "destroy every vehicle.*/walker.*/sensor.* actor already present "
            "in the world before spawning (default off). A previous run "
            "killed hard enough to skip its own cleanup (crash, taskkill) "
            "leaves actors behind on a server that outlives it; those stale "
            "actors can collide with new spawn points or contaminate "
            "ground-truth radius queries in the next run against the same "
            "live server"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    specs = load_rig_config(args.rig, cameras=camera_names) if camera_names else []

    rng = random.Random(args.seed)

    with SimSession(
        host=args.host,
        port=args.port,
        map_name=args.map,
        timeout_s=args.timeout,
    ) as session:
        session.configure_sync(seed_world=args.seed, seed_traffic_manager=args.seed)

        if args.purge_actors:
            n_purged = session.purge_actors()
            print(f"purged {n_purged} pre-existing actor(s) from a dirty server")

        ego = session.spawn_ego(rng, blueprint_filter=args.vehicle)
        # A freshly spawned CARLA actor reports the default transform until
        # the next synchronous world tick. Materialize the ego in the world
        # snapshot before using its location to select nearby traffic spawn
        # points; otherwise `ego.get_transform().location` is (0, 0, 0) and
        # traffic is placed near the map origin instead of near the ego.
        session.tick()
        # Restrict traffic spawn points to within GROUND_TRUTH_RADIUS_M of the
        # ego's own spawn point -- on a town-scale map (e.g. Town01) spawn
        # points are spread across the whole map, so unrestricted random
        # placement can put every background vehicle outside the
        # fixed-radius ground-truth query and make `actors` empty for the
        # whole run despite traffic having spawned and moved correctly (see
        # docs/m4_real_rig.md).
        spawned_traffic = session.spawn_traffic(
            args.traffic,
            rng,
            exclude_spawn_point=ego.get_transform(),
            near_location=ego.get_transform().location,
            max_distance_m=GROUND_TRUTH_RADIUS_M,
        )
        print(f"spawned {len(spawned_traffic)}/{args.traffic} requested traffic vehicle(s)")
        # Materialize the newly spawned traffic actors as well. Besides making
        # their transforms available for diagnostics, this gives us a cheap
        # fail-fast check before a multi-gigabyte camera recording begins.
        session.tick()
        if spawned_traffic:
            ego_location_at_spawn = ego.get_transform().location
            traffic_distances = [
                actor.get_transform().location.distance(ego_location_at_spawn)
                for actor in spawned_traffic
            ]
            print(
                "traffic spawn distances from ego: "
                + ", ".join(f"{distance:.2f}" for distance in sorted(traffic_distances))
            )
            visible_traffic = groundtruth.actor_states(session.world, ego.id, GROUND_TRUTH_RADIUS_M)
            if not visible_traffic:
                raise RuntimeError(
                    "background traffic was spawned but no actor is visible "
                    f"within the {GROUND_TRUTH_RADIUS_M:.1f} m ground-truth radius"
                )
            print(f"ground-truth preflight: {len(visible_traffic)} actor(s) visible")

        rig_used: dict[str, object] = {
            "image_format": args.image_format,
            "rig_path": str(args.rig),
            "vehicle_blueprint": args.vehicle,
            "attach_offset_mode": args.attach_offset,
        }
        model_offset = np.zeros(3)
        rear_offset = np.zeros(3)

        if args.attach_offset == "auto" and specs:
            bbox = ego.bounding_box
            extent = bbox.extent
            location = bbox.location
            vehicle_cfg = load_rig_vehicle(args.rig)
            rear_axle_to_bbox_center = float(vehicle_cfg["rear_axle_to_bbox_center"])
            offset = rig_to_actor_offset(
                rear_axle_to_bbox_center,
                extent.z,
                (location.x, location.y, location.z),
            )
            specs = [apply_offset(spec, offset) for spec in specs]
            shift_x = (
                args.rig_shift_x
                if args.rig_shift_x is not None
                else VEHICLE_MESH_SHIFT_X.get(args.vehicle, 0.0)
            )
            if shift_x:
                specs = [apply_offset(spec, np.array([shift_x, 0.0, 0.0])) for spec in specs]
            rig_used["vehicle_mesh_shift_x"] = float(shift_x)
            model_offset = offset + np.array([shift_x, 0.0, 0.0])
            rear_offset = offset
            print(
                "attach-offset auto: measured bounding_box.extent="
                f"({extent.x:.4f}, {extent.y:.4f}, {extent.z:.4f}), "
                f"bounding_box.location=({location.x:.4f}, {location.y:.4f}, "
                f"{location.z:.4f}), rear_axle_to_bbox_center="
                f"{rear_axle_to_bbox_center:.6f} -> offset="
                f"({offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f})"
            )
            rig_used["measured_bbox_extent"] = [extent.x, extent.y, extent.z]
            rig_used["measured_bbox_location"] = [location.x, location.y, location.z]
            rig_used["rear_axle_to_bbox_center"] = rear_axle_to_bbox_center
            rig_used["applied_offset"] = list(np.asarray(offset, dtype=float))
        else:
            rig_used["measured_bbox_extent"] = None
            rig_used["measured_bbox_location"] = None
            rig_used["applied_offset"] = None
            rig_used["vehicle_mesh_shift_x"] = 0.0

        vehicle_geometry = {
            "revision": "camera-rig-model-origin-and-rear-axle-v1",
            "model_origin_in_actor": model_offset.tolist(),
            "rear_axle_in_actor": rear_offset.tolist(),
            "rear_axle_source": "configured rig hypothesis or unshifted recording",
        }
        rig_used["vehicle_geometry"] = vehicle_geometry
        rig_used["cameras"] = [
            {
                "name": s.name,
                "x": s.x,
                "y": s.y,
                "z": s.z,
                "roll": s.roll,
                "pitch": s.pitch,
                "yaw": s.yaw,
                "fov": s.fov,
                "width": s.width,
                "height": s.height,
            }
            for s in specs
        ]

        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "rig_used.json").open("w", encoding="utf-8") as f:
            json.dump(rig_used, f, indent=2, sort_keys=True, default=float)

        cameras: dict[str, object] = {}
        if not args.no_images and specs:
            cameras = session.spawn_cameras(ego, specs)

        scene = groundtruth.SceneCache(session.world)
        header = RunHeader(
            carla_version=session.client.get_client_version(),
            map_name=session.world.get_map().name,
            seed_world=args.seed,
            seed_traffic_manager=args.seed,
            model_config_name="none-m4-open-loop",
            model_config_hash="none",
            git_sha=_git_sha(),
            run_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            cameras=[s.name for s in specs],
            fixed_delta_seconds=session.fixed_delta_seconds,
            environment_objects=scene.objects,
            vehicle_geometry=vehicle_geometry,
        )

        expected_camera_names = list(cameras.keys())
        n_images_written = 0
        max_actors_seen = 0
        max_traffic_lights_seen = 0
        frames_with_actors = 0
        frames_with_traffic_lights = 0

        with TraceWriter(args.out, header, image_ext=args.image_format) as writer:
            for local_frame in range(args.ticks):
                carla_frame = session.tick()

                images = {}
                if expected_camera_names:
                    images = session.collect_images(carla_frame, expected_camera_names)

                image_paths: dict[str, list[str]] = {}
                for camera_name, image in images.items():
                    abs_path = writer.image_path(camera_name, local_frame)
                    if args.image_format == "jpg":
                        from PIL import Image as _PILImage

                        _arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                            image.height, image.width, 4
                        )[:, :, :3][:, :, ::-1]  # BGRA -> RGB
                        _PILImage.fromarray(np.ascontiguousarray(_arr)).save(
                            str(abs_path), quality=92
                        )
                    else:
                        image.save_to_disk(str(abs_path))
                    image_paths[camera_name] = [
                        writer.relative_image_path(camera_name, local_frame)
                    ]
                    n_images_written += 1

                ego_pose = groundtruth.transform_to_pose(ego.get_transform())
                ego_location = ego.get_transform().location
                control = ego.get_control()

                frame_obj = TraceFrame(
                    frame_id=local_frame,
                    sim_time=round(local_frame * session.fixed_delta_seconds, 6),
                    ego_pose_world=ego_pose,
                    model_pose_world=offset_pose(ego_pose, model_offset),
                    actors=groundtruth.actor_states(session.world, ego.id, GROUND_TRUTH_RADIUS_M),
                    lanes=scene.lanes(ego_location, GROUND_TRUTH_RADIUS_M),
                    traffic_lights=groundtruth.traffic_light_states(
                        session.world, ego_location, GROUND_TRUTH_RADIUS_M
                    ),
                    plan=None,
                    control=ControlCommand(
                        steer=float(control.steer),
                        throttle=float(control.throttle),
                        brake=float(control.brake),
                    ),
                    image_paths=image_paths,
                )
                writer.write_frame(frame_obj)

                max_actors_seen = max(max_actors_seen, len(frame_obj.actors))
                max_traffic_lights_seen = max(
                    max_traffic_lights_seen, len(frame_obj.traffic_lights)
                )
                if frame_obj.actors:
                    frames_with_actors += 1
                if frame_obj.traffic_lights:
                    frames_with_traffic_lights += 1

            n_lines_written = writer.n_frames_written + 1  # + RunHeader line

    print(f"wrote {args.out / 'trace.jsonl'} ({n_lines_written} lines, {n_images_written} images)")
    print(
        f"ground truth: {frames_with_actors}/{args.ticks} frames had actors "
        f"(max {max_actors_seen} at once), {frames_with_traffic_lights}/{args.ticks} "
        f"frames had traffic lights (max {max_traffic_lights_seen} at once)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CARLA client/session management for M4 open-loop recording.

Imports `carla` at module scope -- this module is only ever imported from
`.venv-sim` (see `scripts/run_open_loop.py`), never from the carla-less test
suite.

Owns: connecting, synchronous-mode world settings (with all seeds captured
for the `RunHeader`), spawning the ego vehicle, background traffic, and the
camera rig, the frame-synchronized tick loop, and actor/world-settings
cleanup on exit (including on exception or Ctrl-C).

Frame-to-image correlation (the most dangerous mistake possible here, per
the project brief): cameras are attached with `sensor_tick=0.1` against a
`fixed_delta_seconds=0.05` world, so only roughly every second `world.tick()`
actually produces images, and which ticks those are is not guaranteed to be
perfectly in lockstep across cameras (see docs/sim_env_setup.md). Camera
callbacks therefore never write directly into "the current frame" -- they
push `(image.frame, camera_name, image)` into a single thread-safe pending
dict, keyed by `image.frame`. After `world.tick()` returns the frame number
that was just completed, `SimSession.collect_images` waits (briefly, since
in synchronous mode the data is already computed server-side by the time
`tick()` returns) for that dict entry to contain all requested cameras, then
pops and returns whatever arrived -- an empty or partial dict is a normal,
expected outcome for a tick with no (or not-yet-arrived) camera data, not an
error.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import carla

from acarla.sim.rig import CameraSpec

DEFAULT_TM_PORT = 8000
_IMAGE_WAIT_TOTAL_S = 1.0
_IMAGE_WAIT_POLL_S = 0.01


@dataclass
class TickResult:
    """Everything gathered for one completed simulation tick."""

    frame: int
    sim_time: float
    images: dict[str, carla.Image] = field(default_factory=dict)


class SimSession:
    """Owns one CARLA client/world connection and every actor spawned
    through it. Use as a context manager -- `__exit__` destroys all spawned
    actors (in reverse spawn order) and restores the world's original
    settings, even on exception or KeyboardInterrupt (Fallstrick 6)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        map_name: str | None = None,
        fixed_delta_seconds: float = 0.05,
        timeout_s: float = 60.0,
        tm_port: int = DEFAULT_TM_PORT,
    ) -> None:
        self.fixed_delta_seconds = fixed_delta_seconds

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)

        current_world = self.client.get_world()
        # `client.load_world()` is expensive and, on a tightly-VRAM-budgeted
        # GPU (this project's target hardware has 6144 MiB, already ~93%
        # used by a full 4-camera rig, see docs/sim_env_setup.md), briefly
        # holds two maps' worth of GPU memory while swapping and can crash
        # the server outright. Only reload if the server isn't already on
        # the requested map -- compare the last path segment, since
        # `Map.name` is a full content-browser path (e.g.
        # "Carla/Maps/Town01") while callers pass the bare map name
        # ("Town01"). If you need a specific map and the server keeps
        # dying here, start CarlaUE4.exe directly with that map instead of
        # relying on `load_world`, e.g.:
        # `CarlaUE4.exe /Game/Carla/Maps/Town01 -carla-rpc-port=2000 ...`.
        current_map_name = current_world.get_map().name.rsplit("/", maxsplit=1)[-1]
        if map_name and current_map_name != map_name:
            self.world = self.client.load_world(map_name)
        else:
            self.world = current_world

        self._original_settings = self.world.get_settings()
        self._actors: list[carla.Actor] = []
        self._camera_names: list[str] = []
        self._pending_lock = threading.Lock()
        self._pending: dict[int, dict[str, carla.Image]] = {}
        self._traffic_manager: carla.TrafficManager | None = None
        self.tm_port = tm_port
        self.has_pedestrian_seed = hasattr(self.world, "set_pedestrians_seed")

    # -- setup ---------------------------------------------------------

    def configure_sync(self, seed_world: int, seed_traffic_manager: int) -> None:
        """Enable synchronous mode for the world and the traffic manager,
        and seed every source of randomness this session controls.
        Determinism verification (two runs, same seed, identical logs
        modulo timing fields) is the whole point of M4."""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

        if self.has_pedestrian_seed:
            self.world.set_pedestrians_seed(seed_world)

        tm = self.client.get_trafficmanager(self.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed_traffic_manager)
        self._traffic_manager = tm

    def purge_actors(self) -> int:
        """Destroy every `vehicle.*`/`walker.*`/`sensor.*` actor currently in
        the world, including ones this session did not spawn itself.

        Not called automatically -- a session only knows about (and cleans
        up) the actors it spawned. If a previous run's process was killed
        hard enough to skip `destroy_all()` (crash, `taskkill`, a server
        that outlives a dead client), its actors persist across `run_open_loop.py`
        invocations that reuse the same live server and can collide with new
        spawn attempts or contaminate ground-truth radius queries. Call this
        explicitly (the CLI's `--purge-actors`) when starting a run against a
        server that might be dirty. Returns the number of actors destroyed.
        """
        stale = list(self.world.get_actors().filter("vehicle.*"))
        stale += list(self.world.get_actors().filter("walker.*"))
        stale += list(self.world.get_actors().filter("sensor.*"))

        listening = [a for a in stale if hasattr(a, "is_listening") and a.is_listening]
        for actor in listening:
            try:
                actor.stop()
            except Exception:  # noqa: BLE001 -- purge must not abort partway.
                pass
        if listening:
            time.sleep(0.2)

        alive_ids = []
        for actor in stale:
            try:
                if actor.is_alive:
                    alive_ids.append(actor.id)
            except Exception:  # noqa: BLE001 -- purge must not abort partway.
                pass
        if not alive_ids:
            return 0
        try:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(actor_id) for actor_id in alive_ids], True
            )
        except Exception:  # noqa: BLE001 -- purge must not abort partway.
            return 0
        return len(alive_ids)

    @property
    def traffic_manager(self) -> carla.TrafficManager:
        if self._traffic_manager is None:
            raise RuntimeError("configure_sync() must be called before using the traffic manager")
        return self._traffic_manager

    def spawn_ego(
        self,
        rng: random.Random,
        blueprint_filter: str = "vehicle.tesla.model3",
        autopilot: bool = True,
    ) -> carla.Vehicle:
        """Spawn the ego vehicle at a deterministically-chosen spawn point
        (chosen via `rng`, not CARLA's own unseeded randomness). By default it
        is handed to traffic-manager Autopilot for M4; M5 disables autopilot
        so its trajectory controller can apply commands directly."""
        blueprint_library = self.world.get_blueprint_library()
        blueprint = blueprint_library.filter(blueprint_filter)[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("map has no spawn points")
        spawn_point = rng.choice(spawn_points)

        vehicle = self.world.spawn_actor(blueprint, spawn_point)
        self._actors.append(vehicle)
        if autopilot:
            vehicle.set_autopilot(True, self.tm_port)
        return vehicle

    def spawn_collision_sensor(self, vehicle: carla.Vehicle, callback: Any) -> carla.Sensor:
        """Attach a collision sensor owned by this session."""
        blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        self._actors.append(sensor)
        sensor.listen(callback)
        return sensor

    def spawn_traffic(
        self,
        n: int,
        rng: random.Random,
        exclude_spawn_point: carla.Transform | None = None,
        near_location: carla.Location | None = None,
        max_distance_m: float | None = None,
    ) -> list[carla.Vehicle]:
        """Spawn `n` background vehicles at deterministically-chosen spawn
        points (distinct from each other and from `exclude_spawn_point`),
        each under traffic-manager Autopilot.

        `near_location`/`max_distance_m` (pass both or neither): restrict
        the candidate spawn points to those within `max_distance_m` of
        `near_location` before shuffling. The restriction is strict: if
        fewer than `n` nearby points can be spawned, fewer vehicles are
        returned instead of silently placing the remainder outside the
        requested radius. Town-scale maps (e.g. Town01) spread their spawn
        points across the whole map, so an unrestricted fallback would make
        a fixed-radius ground-truth query appear empty even though traffic
        was spawned successfully elsewhere.
        """
        if n <= 0:
            return []

        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = list(blueprint_library.filter("vehicle.*"))
        # Four-wheeled vehicles only -- bikes/motorcycles need a rider and
        # are a needless source of spawn failures for a background-traffic
        # smoke test.
        vehicle_blueprints = [
            bp for bp in vehicle_blueprints if int(bp.get_attribute("number_of_wheels")) == 4
        ] or vehicle_blueprints

        if (near_location is None) != (max_distance_m is None):
            raise ValueError("near_location and max_distance_m must be passed together")
        if max_distance_m is not None and max_distance_m < 0:
            raise ValueError("max_distance_m must be non-negative")

        spawn_points = list(self.world.get_map().get_spawn_points())
        if exclude_spawn_point is not None:
            spawn_points = [
                sp for sp in spawn_points if sp.location != exclude_spawn_point.location
            ]
        if near_location is not None and max_distance_m is not None:
            spawn_points = [
                sp for sp in spawn_points if sp.location.distance(near_location) <= max_distance_m
            ]
        rng.shuffle(spawn_points)

        vehicles: list[carla.Vehicle] = []
        for spawn_point in spawn_points:
            if len(vehicles) >= n:
                break
            blueprint = rng.choice(vehicle_blueprints)
            if blueprint.has_attribute("color"):
                colors = blueprint.get_attribute("color").recommended_values
                blueprint.set_attribute("color", rng.choice(colors))
            actor = self.world.try_spawn_actor(blueprint, spawn_point)
            if actor is None:
                continue
            actor.set_autopilot(True, self.tm_port)
            self._actors.append(actor)
            vehicles.append(actor)
        return vehicles

    def spawn_cameras(
        self, vehicle: carla.Vehicle, specs: list[CameraSpec]
    ) -> dict[str, carla.Sensor]:
        """Spawn and attach one RGB camera per `CameraSpec`, registering a
        callback that files each captured `carla.Image` into the pending-
        images dict keyed by `image.frame` -- never by callback order or
        count (see module docstring)."""
        blueprint_library = self.world.get_blueprint_library()
        sensors: dict[str, carla.Sensor] = {}
        for spec in specs:
            bp = blueprint_library.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(spec.width))
            bp.set_attribute("image_size_y", str(spec.height))
            bp.set_attribute("fov", str(spec.fov))
            bp.set_attribute("sensor_tick", str(spec.sensor_tick))

            transform = carla.Transform(
                carla.Location(x=spec.x, y=spec.y, z=spec.z),
                carla.Rotation(roll=spec.roll, pitch=spec.pitch, yaw=spec.yaw),
            )
            sensor = self.world.spawn_actor(bp, transform, attach_to=vehicle)
            self._actors.append(sensor)
            self._camera_names.append(spec.name)

            camera_name = spec.name
            sensor.listen(lambda image, name=camera_name: self._on_image(name, image))
            sensors[spec.name] = sensor
        return sensors

    def _on_image(self, camera_name: str, image: carla.Image) -> None:
        with self._pending_lock:
            self._pending.setdefault(image.frame, {})[camera_name] = image

    # -- tick loop -------------------------------------------------------

    def tick(self) -> int:
        """Advance the simulation by exactly one fixed-size step. Returns
        the frame number of the tick just completed."""
        return self.world.tick()

    def collect_images(
        self,
        frame: int,
        expected_cameras: list[str],
        timeout_s: float = _IMAGE_WAIT_TOTAL_S,
    ) -> dict[str, carla.Image]:
        """Return whatever camera images have arrived for `frame`.

        Waits up to `timeout_s` for *all* `expected_cameras` to report in
        (cameras run at half the world's tick rate, so most ticks legitimately
        deliver nothing at all -- that is expected, not an error). Whatever
        is present when the budget runs out (including an empty dict) is
        returned; the entry is removed from the pending store either way so
        it cannot grow unbounded across a long run.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            with self._pending_lock:
                images = self._pending.get(frame, {})
                if expected_cameras and len(images) >= len(expected_cameras):
                    return self._pending.pop(frame)
            if time.monotonic() >= deadline:
                with self._pending_lock:
                    return self._pending.pop(frame, {})
            time.sleep(_IMAGE_WAIT_POLL_S)

    # -- cleanup -----------------------------------------------------------

    def destroy_all(self) -> None:
        """Destroy every actor spawned through this session, tolerating
        actors already gone (e.g. a vehicle that crashed and was
        auto-destroyed by CARLA).

        Sensors are `stop()`-ed before `destroy()`, with a brief pause
        afterwards: destroying a listening sensor without stopping it first
        leaves its background streaming client thread trying to reconnect
        to a now-gone stream indefinitely (observed as a "no stream
        available with id N" spam loop against the server); destroying it
        immediately after `stop()`, with a callback still in flight from the
        tick that just completed, was observed to crash the CARLA client
        process outright (no Python exception -- the process simply died).
        The pause gives any in-flight camera callback time to finish before
        the actor underneath it is destroyed.

        Actors are destroyed via a single batched
        `client.apply_batch_sync([carla.command.DestroyActor(...), ...])`
        rather than one `actor.destroy()` RPC per actor -- the batched form
        is what CARLA's own example scripts use for teardown and proved
        markedly more reliable here than individual synchronous destroy
        calls against a world that had just been ticking with 4 active
        camera sensors.
        """
        listening = [
            actor
            for actor in self._actors
            if hasattr(actor, "is_listening") and getattr(actor, "is_listening", False)
        ]
        for actor in listening:
            try:
                if actor.is_alive:
                    actor.stop()
            except Exception:  # noqa: BLE001 -- cleanup must never raise; a
                # dead RPC connection, an actor CARLA already reclaimed, or a
                # crashed server can all surface here as different exception
                # types and none of them should stop the rest of cleanup.
                pass
        if listening:
            time.sleep(0.2)

        alive_ids = []
        for actor in self._actors:
            try:
                if actor.is_alive:
                    alive_ids.append(actor.id)
            except Exception:  # noqa: BLE001 -- see above.
                pass
        if alive_ids:
            try:
                self.client.apply_batch_sync(
                    [carla.command.DestroyActor(actor_id) for actor_id in alive_ids], True
                )
            except Exception:  # noqa: BLE001 -- see above.
                pass
        self._actors.clear()

    def restore_settings(self) -> None:
        try:
            self.world.apply_settings(self._original_settings)
        except Exception:  # noqa: BLE001 -- cleanup must never raise.
            pass
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except Exception:  # noqa: BLE001 -- cleanup must never raise.
                pass

    def __enter__(self) -> SimSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.destroy_all()
        self.restore_settings()

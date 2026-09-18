import math
from types import SimpleNamespace

import numpy as np

from acarla.control.evaluation import evaluate_run
from acarla.control.supervisor import check_environment, footprint_is_drivable
from acarla.sim.groundtruth import SceneCache
from acarla.types import ActorState, BoundingBox, ControlCommand, Pose, RunHeader


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return Pose(
        np.array([x, y, z], np.float32), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    )


def box(x=2.35, y=1.0, z=0.8, offset=(0, 0, 0.8)):
    return BoundingBox(np.array([x, y, z], np.float32), np.array(offset, np.float32))


def car(x, y=0, *, z=0, yaw=0, velocity=(0, 0, 0), kind="static.vehicles"):
    return ActorState(1, kind, box(), pose(x, y, z, yaw), np.array(velocity, np.float32))


def check(objects=(), path=None, command=None, speed=5, road=None):
    return check_environment(
        pose(),
        speed,
        command or ControlCommand(0, 0.4, 0),
        path,
        list(objects),
        road or (lambda p: abs(p[1]) < 1.8),
        box(),
    )


def test_clear_straight_lane_does_not_brake():
    assert not check().brake


def test_static_parked_car_causes_early_braking():
    result = check([car(9)])
    assert result.brake and result.object_id == 1
    assert result.hazard_distance_m < 5


def test_adjacent_lane_car_does_not_brake():
    assert not check([car(7, 3.6)]).brake


def test_following_moving_lead_at_safe_gap_is_not_treated_as_static():
    assert not check([car(16, velocity=(8, 0, 0), kind="vehicle.test")], speed=8).brake


def test_fake_worker_can_never_be_reported_as_model_pass():
    result = evaluate_run(
        {
            "ticks_completed": 1200,
            "ticks_requested": 1200,
            "distance_m": 80,
            "n_plans": 58,
            "alpamayo_used": False,
        }
    )
    assert result["safe_completion"] and not result["unassisted_model_pass"]


def test_world_anchored_rightward_plan_cannot_drive_onto_sidewalk():
    path = np.array([[x, 0.10 * x * x, 0] for x in np.linspace(0, 20, 64)])
    result = check(path=path)
    assert result.brake and "footprint" in result.reason


def test_applied_steering_checked_even_when_plan_is_straight():
    path = np.array([[x, 0, 0] for x in np.linspace(0, 20, 64)])
    result = check(path=path, command=ControlCommand(0.7, 0.4, 0))
    assert result.brake and result.reason.startswith("steering:")


def test_vehicle_corner_is_checked_not_just_center():
    assert check(road=lambda p: abs(p[1]) < 0.9).brake


def test_measured_footprint_metric_works_without_supervisor_override():
    assert not footprint_is_drivable(pose(y=1.0), box(), lambda p: abs(p[1]) < 1.8)
    assert footprint_is_drivable(pose(), box(), lambda p: abs(p[1]) < 1.8)


def test_exception_after_last_tick_cannot_be_reported_as_success():
    result = evaluate_run(
        {
            "ticks_completed": 1200,
            "ticks_requested": 1200,
            "distance_m": 80,
            "n_plans": 58,
            "error": "failed cleanup",
        }
    )
    assert not result["safe_completion"]


def test_wheel_departure_fails_even_if_center_stays_on_road():
    result = evaluate_run(
        {
            "ticks_completed": 1200,
            "ticks_requested": 1200,
            "distance_m": 80,
            "n_plans": 58,
            "footprint_offroad_ticks": 1,
        }
    )
    assert result["failure_reasons"] == ["footprint_offroad"]


def test_crossing_vehicle_predicted_into_path():
    crossing = car(6, 4, yaw=-math.pi / 2, velocity=(0, -4, 0), kind="vehicle.test")
    assert check([crossing]).brake


def test_overhead_structure_does_not_brake():
    assert not check([car(6, z=8)]).brake


def test_flat_manhole_cover_does_not_block_driving():
    cover = car(5, kind="static.dynamic")
    cover.bounding_box = box(0.25, 0.25, 0.008, (0, 0, 0.006))
    assert not check([cover]).brake


def test_static_box_offset_is_applied():
    obstacle = car(6, 8)
    obstacle.bounding_box.location[1] = -8
    assert check([obstacle]).brake


def test_standstill_checks_space_before_accelerating():
    assert check([car(6)], speed=0).brake


def test_unsafe_speed_fails_closed():
    assert check(speed=float("nan")).brake


def test_complete_timer_with_collision_is_failure():
    result = evaluate_run(
        {
            "ticks_completed": 1200,
            "ticks_requested": 1200,
            "n_collision_frames": 3,
            "distance_m": 316,
            "n_plans": 58,
        }
    )
    assert not result["safe_completion"] and result["failure_reasons"] == ["collision"]


def test_stationary_run_does_not_pass_and_assistance_is_visible():
    result = evaluate_run(
        {
            "ticks_completed": 1200,
            "ticks_requested": 1200,
            "distance_m": 0,
            "n_plans": 58,
            "supervisor_ticks": 1100,
        }
    )
    assert not result["safe_completion"] and result["assisted"]
    assert "insufficient_progress" in result["failure_reasons"]


def test_assisted_safe_completion_not_counted_as_model_pass():
    result = evaluate_run(
        {
            "ticks_completed": 1200,
            "ticks_requested": 1200,
            "distance_m": 80,
            "n_plans": 58,
            "aeb_ticks": 2,
        }
    )
    assert result["safe_completion"] and not result["unassisted_model_pass"]


def test_environment_box_uses_world_location_without_double_transform():
    def vec(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    rotation = SimpleNamespace(
        get_forward_vector=lambda: vec(0, 1, 0),
        get_right_vector=lambda: vec(-1, 0, 0),
        get_up_vector=lambda: vec(0, 0, 1),
    )
    obj = SimpleNamespace(
        id=2**60,
        type="CityObjectLabel.Vehicles",
        bounding_box=SimpleNamespace(
            location=vec(10, 20, 1), extent=vec(2, 1, 1), rotation=rotation
        ),
        transform=SimpleNamespace(location=vec(100, 200, 1)),
    )
    map_ = SimpleNamespace(generate_waypoints=lambda spacing: [])
    world = SimpleNamespace(get_map=lambda: map_, get_environment_objects=lambda: [obj])
    scene = SceneCache(world)
    assert len(scene.nearby(np.array([10, 20, 0]), 5)) == 1
    np.testing.assert_array_equal(scene.objects[0].transform.translation, [10, 20, 1])
    assert scene.objects[0].id == -1  # JS-safe ID, not a lossy uint64 conversion
    assert scene.objects[0].transform.rotation[1, 0] == 1
    header = RunHeader(
        "test",
        "test",
        1,
        1,
        "test",
        "test",
        "test",
        "test",
        [],
        0.05,
        environment_objects=scene.objects,
    )
    restored = RunHeader.from_json_dict(header.to_json_dict())
    assert restored.environment_objects[0].type_id == "static.vehicles"


def test_street_light_arm_over_the_lane_is_not_an_obstacle():
    # run 7: mast at the kerb, 3.8 m arm over the lane, box centre 4 m up, 7.8 m tall
    arm = ActorState(
        -40769, "static.poles", BoundingBox(np.array([1.92, 0.22, 3.88], np.float32),
                                            np.zeros(3, np.float32)),
        pose(6.0, 1.5, 4.03), np.zeros(3, np.float32),
    )
    assert not check([arm]).brake
    # a plain bollard-sized pole in the lane still counts
    bollard = ActorState(
        -1, "static.poles", BoundingBox(np.array([0.15, 0.15, 0.6], np.float32),
                                        np.zeros(3, np.float32)),
        pose(6.0, 0.0, 0.6), np.zeros(3, np.float32),
    )
    assert check([bollard]).brake

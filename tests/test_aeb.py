import numpy as np

from acarla.control.aeb import apply_aeb, check_forward_corridor
from acarla.types import ActorState, BoundingBox, ControlCommand, Pose


def _pose(x=0.0, y=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return Pose(translation=np.array([x, y, 0], dtype=np.float32), rotation=rot)


def _car(x, y, vx=0.0, yaw=0.0, aid=1):
    return ActorState(
        id=aid,
        type_id="vehicle.test",
        transform=_pose(x, y, yaw),
        bounding_box=BoundingBox(
            extent=np.array([2.2, 1.0, 0.8], np.float32), location=np.zeros(3, np.float32)
        ),
        velocity=np.array([vx, 0, 0], np.float32),
    )


def test_stopped_lead_at_bumper_distance_triggers():
    d = check_forward_corridor(_pose(), 0.3, [_car(4.8, 0.3)])
    assert d.brake and d.gap_m is not None and d.gap_m < 0.5
    cmd = apply_aeb(ControlCommand(steer=0.1, throttle=0.5, brake=0.0), d)
    assert cmd.brake == 1.0 and cmd.throttle == 0.0 and cmd.steer == 0.1


def test_lead_in_adjacent_lane_is_ignored():
    assert not check_forward_corridor(_pose(), 5.0, [_car(6.0, 3.6)]).brake


def test_lead_at_same_speed_far_enough_is_ignored():
    # 8 m/s both, 10 m gap: closing speed 0 -> only the standoff counts
    assert not check_forward_corridor(_pose(), 8.0, [_car(14.65, 0.0, vx=8.0)]).brake


def test_fast_closing_on_stopped_lead_triggers_early():
    # 8 m/s onto a stopped car 10 m ahead: 1.5 + 2.4 + 8 = 11.9 m > 10 m gap
    d = check_forward_corridor(_pose(), 8.0, [_car(14.65, 0.0)])
    assert d.brake


def test_actor_behind_is_ignored():
    assert not check_forward_corridor(_pose(), 5.0, [_car(-6.0, 0.0)]).brake


def test_rotated_ego_uses_its_own_axis():
    ego = _pose(yaw=np.pi / 2)  # facing +y
    assert check_forward_corridor(ego, 0.0, [_car(0.0, 5.0, yaw=np.pi / 2)]).brake
    assert not check_forward_corridor(ego, 0.0, [_car(5.0, 0.0)]).brake


def test_bounding_box_offset_is_not_ignored():
    actor = _car(5.0, 5.0)
    actor.bounding_box.location[1] = -5.0
    assert check_forward_corridor(_pose(), 3.0, [actor]).brake


def test_oncoming_car_in_corridor_includes_its_closing_speed():
    assert check_forward_corridor(_pose(), 5.0, [_car(20, 0, vx=-8)]).brake


def test_overhead_bbox_is_not_a_road_obstacle():
    actor = _car(5.0, 0.0)
    actor.bounding_box.location[2] = 9.0
    assert not check_forward_corridor(_pose(), 5, [actor]).brake

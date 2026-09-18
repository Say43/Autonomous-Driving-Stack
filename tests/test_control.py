"""Unit tests for the M5 controller baseline (no CARLA dependency)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acarla.control import (
    PIDConfig,
    PIDController,
    PurePursuitConfig,
    TrajectoryController,
    check_trajectory,
    flu_points_to_world,
    pure_pursuit_steer,
    world_points_to_flu,
)
from acarla.types import N_PLAN_WAYPOINTS, PlanResult, Pose


def pose(x: float = 0.0, y: float = 0.0, yaw_rad: float = 0.0) -> Pose:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return Pose(np.array([x, y, 0.0], dtype=np.float32), rotation)


def plan_from_xy(xy: np.ndarray) -> PlanResult:
    xyz = np.zeros((N_PLAN_WAYPOINTS, 3), dtype=np.float32)
    xyz[:, :2] = xy
    rotations = np.tile(np.eye(3, dtype=np.float32), (N_PLAN_WAYPOINTS, 1, 1))
    return PlanResult(
        frame_id=0,
        waypoints_xyz=xyz,
        waypoints_rot=rotations,
        reasoning=None,
        inference_ms=0.0,
        model_config_hash="test",
    )


def straight_plan(speed_mps: float = 5.0, lateral_m: float = 0.0) -> PlanResult:
    distance_per_step = speed_mps * 0.1
    x = np.arange(1, N_PLAN_WAYPOINTS + 1, dtype=np.float32) * distance_per_step
    y = np.full_like(x, lateral_m)
    return plan_from_xy(np.column_stack((x, y)))


def test_flu_world_roundtrip_includes_carla_y_reflection() -> None:
    origin = pose(10.0, -3.0, math.pi / 2.0)
    points = np.array([[4.0, 2.0, 0.0], [1.0, -5.0, 0.5]], dtype=np.float32)
    world = flu_points_to_world(points, origin)
    restored = world_points_to_flu(world, origin)
    np.testing.assert_allclose(restored, points, atol=1e-6)

    # At yaw=0, Alpamayo-left (+y FLU) is CARLA-right-negative (-y world).
    world_at_zero = flu_points_to_world(np.array([[0.0, 2.0, 0.0]], dtype=np.float32), pose())
    np.testing.assert_array_equal(world_at_zero, np.array([[0.0, -2.0, 0.0]], dtype=np.float32))


def test_pure_pursuit_straight_is_zero() -> None:
    assert pure_pursuit_steer(straight_plan().waypoints_xyz, 5.0) == pytest.approx(0.0)


def test_pure_pursuit_left_maps_to_negative_carla_steer() -> None:
    left = straight_plan(lateral_m=2.0)
    right = straight_plan(lateral_m=-2.0)
    assert pure_pursuit_steer(left.waypoints_xyz, 5.0) < 0.0
    assert pure_pursuit_steer(right.waypoints_xyz, 5.0) > 0.0


def test_pure_pursuit_is_bounded() -> None:
    points = straight_plan(lateral_m=100.0).waypoints_xyz
    value = pure_pursuit_steer(
        points,
        0.0,
        PurePursuitConfig(max_steer_angle_rad=math.radians(5.0)),
    )
    assert -1.0 <= value <= 1.0


def test_pid_accelerates_and_brakes_without_mixing_outputs() -> None:
    pid = PIDController(PIDConfig(kp=0.2, ki=0.0, kd=0.0))
    assert pid.update(2.0, 0.05) == pytest.approx(0.4)
    assert pid.update(-2.0, 0.05) == pytest.approx(-0.4)


def test_pid_rejects_invalid_dt() -> None:
    with pytest.raises(ValueError, match="positive"):
        PIDController().update(1.0, 0.0)


def test_safety_accepts_constant_speed_straight_plan() -> None:
    result = check_trajectory(straight_plan(speed_mps=8.0))
    assert result.safe
    assert result.max_speed_mps == pytest.approx(8.0, abs=1e-4)
    assert result.max_abs_acceleration_mps2 == pytest.approx(0.0, abs=1e-3)
    assert result.max_abs_curvature_inv_m == pytest.approx(0.0)


def test_safety_rejects_acceleration_spike() -> None:
    steps = np.full(N_PLAN_WAYPOINTS, 0.1, dtype=np.float32)
    steps[2:] = 1.5
    x = np.cumsum(steps)
    result = check_trajectory(plan_from_xy(np.column_stack((x, np.zeros_like(x)))))
    assert not result.safe
    assert "acceleration" in result.reason


def test_safety_rejects_excess_curvature() -> None:
    radius = 2.0
    theta = np.arange(1, N_PLAN_WAYPOINTS + 1, dtype=np.float32) * 0.05
    x = radius * np.sin(theta)
    y = radius * (1.0 - np.cos(theta))
    result = check_trajectory(plan_from_xy(np.column_stack((x, y))))
    assert not result.safe
    assert "curvature" in result.reason


def test_controller_brakes_without_plan() -> None:
    decision = TrajectoryController().step(pose(), 0.0, 0.0, 0.05)
    assert not decision.safe
    assert decision.command.throttle == 0.0
    assert decision.command.brake > 0.0


def test_controller_tracks_safe_plan_and_replans_at_two_hz() -> None:
    controller = TrajectoryController()
    assert controller.needs_replan(0.0)
    safety = controller.set_plan(straight_plan(6.0), pose(), 0.0)
    assert safety.safe
    assert not controller.needs_replan(0.49)
    assert controller.needs_replan(0.5)

    decision = controller.step(pose(), current_speed_mps=0.0, sim_time_s=0.0, dt=0.05)
    assert decision.safe
    assert decision.target_speed_mps == pytest.approx(6.0, abs=1e-4)
    assert decision.command.throttle > 0.0
    assert decision.command.brake == 0.0
    assert decision.command.steer == pytest.approx(0.0)


def test_controller_tracks_world_anchored_plan_after_ego_moves() -> None:
    controller = TrajectoryController()
    controller.set_plan(straight_plan(5.0), pose(), 0.0)
    decision = controller.step(pose(x=1.0), current_speed_mps=5.0, sim_time_s=0.2, dt=0.05)
    assert decision.safe
    assert decision.command.steer == pytest.approx(0.0)


def test_controller_brakes_on_stale_plan() -> None:
    controller = TrajectoryController()
    controller.set_plan(straight_plan(), pose(), 0.0)
    decision = controller.step(pose(), 5.0, 1.0, 0.05)
    assert not decision.safe
    assert "stale" in decision.reason
    assert decision.command.brake > 0.0


def test_controller_brakes_on_nonfinite_plan() -> None:
    plan = straight_plan()
    plan.waypoints_xyz[5, 0] = np.nan
    controller = TrajectoryController()
    safety = controller.set_plan(plan, pose(), 0.0)
    assert not safety.safe
    decision = controller.step(pose(), 0.0, 0.0, 0.05)
    assert not decision.safe
    assert "non-finite" in decision.reason


def test_progress_tracking_closes_lag_behind_a_gentle_start() -> None:
    """An Alpamayo-style start ramps slowly; a speed-at-age tracker would
    demand ~0 m/s at age 0. Tracking progress along the plan instead asks for
    the average speed needed to be on schedule 2 s from now."""
    ramp = np.zeros((64, 3), dtype=np.float32)
    speeds = np.minimum(np.arange(1, 65) * 0.1 * 2.2, 9.0)  # 2.2 m/s^2 up to 9 m/s
    ramp[:, 0] = np.cumsum(speeds * 0.1)
    rot = np.tile(np.eye(3, dtype=np.float32), (64, 1, 1))
    plan = PlanResult(
        frame_id=0,
        waypoints_xyz=ramp,
        waypoints_rot=rot,
        reasoning=None,
        inference_ms=0.0,
        model_config_hash="h",
    )
    controller = TrajectoryController()
    assert controller.set_plan(plan, pose(), 0.0).safe
    decision = controller.step(pose(), current_speed_mps=0.0, sim_time_s=0.0, dt=0.05)
    expected = float(ramp[19, 0] / 2.0)  # distance scheduled by t=2.0 s over 2 s
    assert decision.target_speed_mps == pytest.approx(expected, abs=1e-3)
    assert decision.target_speed_mps > 1.5
    assert decision.command.throttle > 0.0
    # ego already ahead of schedule -> no positive speed demand
    ahead = controller.step(
        pose(x=float(ramp[25, 0])), current_speed_mps=3.0, sim_time_s=0.0, dt=0.05
    )
    assert ahead.target_speed_mps == pytest.approx(0.0)


def test_progress_tracking_stops_where_the_plan_stops() -> None:
    xyz = np.zeros((64, 3), dtype=np.float32)
    xyz[:, 0] = np.minimum(np.arange(1, 65) * 0.3, 3.0)  # 3 m/s, halts after 3 m
    rot = np.tile(np.eye(3, dtype=np.float32), (64, 1, 1))
    plan = PlanResult(
        frame_id=0,
        waypoints_xyz=xyz,
        waypoints_rot=rot,
        reasoning=None,
        inference_ms=0.0,
        model_config_hash="h",
    )
    controller = TrajectoryController()
    controller.set_plan(plan, pose(), 0.0)
    at_stop = controller.step(pose(x=3.0), current_speed_mps=1.0, sim_time_s=0.0, dt=0.05)
    assert at_stop.target_speed_mps == pytest.approx(0.0)
    assert at_stop.command.brake > 0.0


def test_speed_target_never_exceeds_the_plans_own_speed_or_the_cap() -> None:
    controller = TrajectoryController()
    controller.set_plan(straight_plan(6.0), pose(), 0.0)
    # an ego behind the plan origin projects onto its start: no catch-up beyond the plan speed
    lagging = controller.step(pose(x=-8.0), current_speed_mps=0.0, sim_time_s=0.0, dt=0.05)
    assert lagging.target_speed_mps <= 6.0 + 1.0 + 1e-6
    fast = TrajectoryController()
    fast.set_plan(straight_plan(14.0), pose(), 0.0)
    assert fast.step(
        pose(), current_speed_mps=0.0, sim_time_s=0.0, dt=0.05
    ).target_speed_mps == pytest.approx(10.0)


def test_invalidated_worker_plan_brakes_immediately():
    controller = TrajectoryController()
    controller.set_plan(straight_plan(), pose(), 0)
    controller.invalidate_plan("worker error")
    result = controller.step(pose(), 5, 0.1, 0.05)
    assert not result.safe and result.command.brake > 0
    assert result.command.throttle == 0


def test_curve_speed_is_limited_by_lateral_acceleration():
    theta = np.arange(1, 65, dtype=np.float32) * 0.04
    curved = plan_from_xy(np.column_stack((20 * np.sin(theta), 20 * (1 - np.cos(theta)))))
    controller = TrajectoryController()
    assert controller.set_plan(curved, pose(), 0).safe
    decision = controller.step(pose(), 8, 0, 0.05)
    assert decision.target_speed_mps < 7.2  # sqrt(2.5 * 20) = 7.07 m/s

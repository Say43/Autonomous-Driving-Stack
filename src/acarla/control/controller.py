"""Trajectory tracking with Pure Pursuit, speed PID, and fail-safe braking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from acarla.control.frames import (
    offset_pose,
    plan_reference_points,
    plan_world_yaws,
    world_points_to_flu,
)
from acarla.control.pid import PIDConfig, PIDController
from acarla.control.pure_pursuit import PurePursuitConfig, pure_pursuit_steer
from acarla.control.safety import SafetyCheck, SafetyLimits, check_trajectory
from acarla.types import SAMPLE_DT, ControlCommand, PlanResult, Pose


@dataclass(frozen=True)
class ControllerConfig:
    replan_interval_s: float = 0.5
    max_plan_age_s: float = 0.8
    speed_lookahead_s: float = 2.0
    max_speed_mps: float = 10.0
    max_lateral_accel_mps2: float = 2.5
    """Hard cap on the speed target (urban runs). The plan's own speed profile
    is respected below this: the progress target never exceeds the plan's
    instantaneous speed at the lookahead point by more than
    `catch_up_margin_mps`, so catching up on a lag cannot turn a 10 m/s plan
    into a 15 m/s corner entry (run 5, results/loop_town10_seed21, t=30-35 s)."""
    catch_up_margin_mps: float = 1.0
    """Speed target = arc length from the ego's progress along the plan to
    the plan point this far in the future, divided by this horizon. Tracking
    *position* instead of the plan's instantaneous speed closes any lag: an
    Alpamayo start from standstill ramps to only 0.7 m/s within its first
    second, and with 1 s replans a speed-at-age tracker saw a near-zero target
    forever (the ego never moved, so every new plan restarted the ramp --
    results/loop_town10_seed21_partial49s stood still for 15 s)."""
    brake_on_fault: float = 0.6
    max_steer_rate_per_s: float = 1.5
    pure_pursuit: PurePursuitConfig = field(default_factory=PurePursuitConfig)
    speed_pid: PIDConfig = field(default_factory=PIDConfig)
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    # CARLA-local offsets; appended so legacy positional arguments keep their meaning.
    model_origin_in_actor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rear_axle_in_actor: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ControlDecision:
    command: ControlCommand
    safe: bool
    reason: str
    target_speed_mps: float
    plan_age_s: float | None


class TrajectoryController:
    """Tracks the newest accepted plan and brakes on stale/unsafe input."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._pid = PIDController(self.config.speed_pid)
        self._plan: PlanResult | None = None
        self._plan_time_s: float | None = None
        self._world_waypoints: np.ndarray | None = None
        self._actor_waypoints: np.ndarray | None = None
        self._world_yaws: np.ndarray | None = None
        self._plan_origin_xy = np.zeros(2, dtype=np.float64)
        self._safety = SafetyCheck(False, "no plan")
        self._last_steer = 0.0

    @property
    def last_safety_check(self) -> SafetyCheck:
        return self._safety

    @property
    def world_waypoints(self) -> np.ndarray | None:
        """Actor-origin path for footprint checks, NOT the rear-axle tracking path."""
        return self._actor_waypoints

    def invalidate_plan(self, reason: str) -> None:
        self._world_waypoints = None
        self._actor_waypoints = None
        self._world_yaws = None
        self._safety = SafetyCheck(False, reason)
        self._pid.reset()

    @property
    def world_yaws(self) -> np.ndarray | None:
        return self._world_yaws

    def reset_after_override(self) -> None:
        """Avoid integral wind-up while an external layer holds the brake."""
        self._pid.reset()

    def needs_replan(self, sim_time_s: float) -> bool:
        return self._plan_time_s is None or (
            sim_time_s - self._plan_time_s >= self.config.replan_interval_s - 1e-9
        )

    def set_plan(self, plan: PlanResult, origin_world: Pose, sim_time_s: float) -> SafetyCheck:
        safety = check_trajectory(plan, self.config.safety)
        self._safety = safety
        self._plan = plan
        self._plan_time_s = float(sim_time_s)
        model_offset = np.asarray(self.config.model_origin_in_actor)
        rear_offset = np.asarray(self.config.rear_axle_in_actor)
        model_pose = offset_pose(origin_world, model_offset)
        self._world_waypoints = (
            plan_reference_points(plan, model_pose, rear_offset - model_offset)
            if safety.safe
            else None
        )
        self._actor_waypoints = (
            plan_reference_points(plan, model_pose, -model_offset) if safety.safe else None
        )
        self._world_yaws = plan_world_yaws(plan, model_pose) if safety.safe else None
        self._plan_origin_xy = offset_pose(origin_world, rear_offset).translation[:2].copy()
        return safety

    def _fault(self, reason: str, plan_age_s: float | None, dt: float) -> ControlDecision:
        self._pid.reset()
        max_delta = self.config.max_steer_rate_per_s * dt
        self._last_steer = float(
            np.clip(0.0, self._last_steer - max_delta, self._last_steer + max_delta)
        )
        return ControlDecision(
            command=ControlCommand(
                steer=self._last_steer,
                throttle=0.0,
                brake=self.config.brake_on_fault,
            ),
            safe=False,
            reason=reason,
            target_speed_mps=0.0,
            plan_age_s=plan_age_s,
        )

    def _progress_speed_target(self, ego_world: Pose, age: float) -> float:
        """Arc length the ego still has to cover to be on schedule
        `speed_lookahead_s` from now, divided by that horizon (>= 0)."""
        assert self._plan is not None and self._world_waypoints is not None
        polyline = np.concatenate(
            (self._plan_origin_xy[None, :], self._world_waypoints[:, :2]), axis=0
        ).astype(np.float64)
        seg = np.diff(polyline, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
        # project the ego onto the polyline -> progress s0
        ego_xy = ego_world.translation[:2].astype(np.float64)
        rel = ego_xy[None, :] - polyline[:-1]
        denom = np.maximum(seg_len**2, 1e-9)
        t = np.clip(np.einsum("ij,ij->i", rel, seg) / denom, 0.0, 1.0)
        nearest = polyline[:-1] + t[:, None] * seg
        dist = np.linalg.norm(nearest - ego_xy[None, :], axis=1)
        i = int(np.argmin(dist))
        s0 = cumulative[i] + t[i] * seg_len[i]
        # plan point scheduled for (age + lookahead): waypoint k is at (k+1)*SAMPLE_DT
        target_time = max(age, 0.0) + self.config.speed_lookahead_s
        k = min(max(int(round(target_time / SAMPLE_DT)) - 1, 0), len(self._world_waypoints) - 1)
        horizon = max((k + 1) * SAMPLE_DT - max(age, 0.0), SAMPLE_DT)
        s1 = cumulative[k + 1]
        progress_target = max(0.0, (s1 - s0) / horizon)
        planned_speed = seg_len[k] / SAMPLE_DT  # the plan's own speed around the lookahead point
        return float(
            min(
                progress_target,
                planned_speed + self.config.catch_up_margin_mps,
                self.config.max_speed_mps,
            )
        )

    def step(
        self,
        ego_world: Pose,
        current_speed_mps: float,
        sim_time_s: float,
        dt: float,
    ) -> ControlDecision:
        if not math.isfinite(current_speed_mps) or current_speed_mps < 0.0:
            return self._fault("current speed is invalid", None, dt)
        if self._plan is None or self._plan_time_s is None or self._world_waypoints is None:
            return self._fault(self._safety.reason, None, dt)

        age = float(sim_time_s - self._plan_time_s)
        if age < -1e-6:
            return self._fault("simulation time precedes plan time", age, dt)
        if age > self.config.max_plan_age_s:
            return self._fault(
                f"plan is stale ({age:.3f}s > {self.config.max_plan_age_s:.3f}s)",
                age,
                dt,
            )

        # Pure Pursuit's bicycle geometry is defined at the rear axle.
        rear_pose = offset_pose(ego_world, self.config.rear_axle_in_actor)
        current_path = world_points_to_flu(self._world_waypoints, rear_pose)
        try:
            desired_steer = pure_pursuit_steer(
                current_path, current_speed_mps, self.config.pure_pursuit
            )
        except ValueError as exc:
            return self._fault(str(exc), age, dt)

        max_delta = self.config.max_steer_rate_per_s * dt
        steer = float(
            np.clip(
                desired_steer,
                self._last_steer - max_delta,
                self._last_steer + max_delta,
            )
        )
        self._last_steer = steer

        target_speed = self._progress_speed_target(rear_pose, age)
        curvature = abs(
            math.tan(desired_steer * self.config.pure_pursuit.max_steer_angle_rad)
            / self.config.pure_pursuit.wheelbase_m
        )
        if curvature > 1e-6:
            target_speed = min(
                target_speed, math.sqrt(self.config.max_lateral_accel_mps2 / curvature)
            )
        effort = self._pid.update(target_speed - current_speed_mps, dt)
        throttle = max(0.0, effort)
        brake = max(0.0, -effort)
        return ControlDecision(
            command=ControlCommand(steer=steer, throttle=throttle, brake=brake),
            safe=True,
            reason="ok",
            target_speed_mps=target_speed,
            plan_age_s=age,
        )

"""Automatic emergency braking from simulator ground truth.

A last-resort layer *below* the planner: it never steers and never
accelerates, it only replaces the command with full brake when another actor
sits inside the ego's forward corridor closer than the distance the ego needs
to stop. It uses CARLA ground truth (actor poses and boxes), so every
intervention is a documented rescue of the model, not part of its behaviour;
the loop script counts them and stamps the trace.

Why it exists: the first closed-loop run (results/loop_town10_seed21, 2026-09-16)
crept at 0.3 m/s into a stopped lead vehicle while every plan said
"Stop to keep distance" but still extended 6-9 m -- the model overestimates the
gap to a stopped lead on CARLA imagery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acarla.types import ActorState, ControlCommand, Pose


@dataclass(frozen=True)
class AebConfig:
    corridor_half_width_m: float = 1.6
    """Lateral half-width of the checked corridor around the ego centreline."""
    standoff_m: float = 1.5
    """Bumper-to-bumper gap below which we brake regardless of speed."""
    decel_mps2: float = 4.0
    """Assumed braking deceleration for the stopping-distance term."""
    reaction_s: float = 0.3
    ego_front_m: float = 2.45
    ego_rear_m: float = 2.45
    """Distance from the ego origin to its front bumper (Lincoln MKZ 2017 half-length)."""


@dataclass(frozen=True)
class AebDecision:
    brake: bool
    reason: str
    gap_m: float | None = None
    actor_id: int | None = None


def check_forward_corridor(
    ego: Pose,
    speed_mps: float,
    actors: list[ActorState],
    config: AebConfig | None = None,
) -> AebDecision:
    """Return whether an actor blocks the corridor within the stopping distance."""
    cfg = config or AebConfig()
    if not np.isfinite(speed_mps) or speed_mps < 0:
        return AebDecision(True, "aeb: invalid speed")
    r_t = ego.rotation.T.astype(np.float64)
    worst: AebDecision | None = None
    for actor in actors:
        center = (
            actor.transform.translation.astype(np.float64)
            + actor.transform.rotation @ actor.bounding_box.location
        )
        local = r_t @ (center - ego.translation.astype(np.float64))
        # actor half-extent along the ego axis (rectangle projected conservatively)
        ex, ey = float(actor.bounding_box.extent[0]), float(actor.bounding_box.extent[1])
        rel_rot = r_t @ actor.transform.rotation.astype(np.float64)
        half_len = abs(rel_rot[0, 0]) * ex + abs(rel_rot[0, 1]) * ey
        half_wid = abs(rel_rot[1, 0]) * ex + abs(rel_rot[1, 1]) * ey
        if local[0] + half_len < -cfg.ego_rear_m:
            continue
        half_height = float(np.abs(rel_rot[2]) @ actor.bounding_box.extent)
        if local[2] - half_height > 2.0 or local[2] + half_height < -0.5:
            continue  # overhead structure / different road level
        if abs(local[1]) - half_wid > cfg.corridor_half_width_m:
            continue
        gap = float(local[0] - half_len - cfg.ego_front_m)
        # stopping distance from the *closing* speed, so a lead moving at our
        # speed 10 m ahead is not a permanent AEB trigger
        lead_forward_v = float((r_t @ actor.velocity.astype(np.float64))[0])
        closing = max(0.0, speed_mps - lead_forward_v)
        braking = max(0.0, speed_mps**2 - max(lead_forward_v, 0.0) ** 2)
        if lead_forward_v < 0:
            braking = closing**2
        stop_dist = (
            cfg.standoff_m
            + max(speed_mps, closing) * cfg.reaction_s
            + braking / (2.0 * cfg.decel_mps2)
        )
        if gap <= stop_dist and (worst is None or gap < worst.gap_m):
            worst = AebDecision(
                True, f"aeb: {actor.type_id} {gap:.2f} m ahead <= {stop_dist:.2f} m", gap, actor.id
            )
    return worst or AebDecision(False, "clear")


def apply_aeb(command: ControlCommand, decision: AebDecision) -> ControlCommand:
    if not decision.brake:
        return command
    return ControlCommand(steer=command.steer, throttle=0.0, brake=1.0)

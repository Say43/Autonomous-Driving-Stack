"""Small deterministic PID implementation for longitudinal speed control."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 0.35
    ki: float = 0.08
    kd: float = 0.04
    integral_limit: float = 5.0
    derivative_time_constant_s: float = 0.1
    output_min: float = -1.0
    output_max: float = 1.0


class PIDController:
    """PID with derivative filtering, integral clamping, and anti-windup."""

    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = config or PIDConfig()
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error: float | None = None
        self._filtered_derivative = 0.0

    def update(self, error: float, dt: float) -> float:
        if not math.isfinite(error):
            raise ValueError(f"error must be finite, got {error}")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt}")

        derivative = (
            0.0 if self._previous_error is None else (error - self._previous_error) / dt
        )
        tau = max(0.0, self.config.derivative_time_constant_s)
        alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
        self._filtered_derivative += alpha * (derivative - self._filtered_derivative)

        candidate_integral = max(
            -self.config.integral_limit,
            min(self.config.integral_limit, self._integral + error * dt),
        )
        candidate = (
            self.config.kp * error
            + self.config.ki * candidate_integral
            + self.config.kd * self._filtered_derivative
        )
        output = max(self.config.output_min, min(self.config.output_max, candidate))

        # Do not accumulate more integral when the output is saturated in the
        # same direction as the current error.
        saturated_high = candidate > self.config.output_max and error > 0.0
        saturated_low = candidate < self.config.output_min and error < 0.0
        if not (saturated_high or saturated_low):
            self._integral = candidate_integral
        self._previous_error = error
        return output

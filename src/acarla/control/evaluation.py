"""Explicit closed-loop outcomes; finishing a timer is not a safety pass."""


def evaluate_run(summary: dict, min_distance_m: float = 25.0) -> dict:
    reasons = []
    if summary.get("error") or summary.get("status") == "aborted":
        reasons.append("execution_error")
    if summary.get("ticks_completed", 0) != summary.get("ticks_requested", -1):
        reasons.append("incomplete")
    if summary.get("n_collision_frames", 0):
        reasons.append("collision")
    if summary.get("offroad_ticks", 0):
        reasons.append("offroad")
    if summary.get("footprint_offroad_ticks", 0):
        reasons.append("footprint_offroad")
    if summary.get("fault_ticks", 0):
        reasons.append("controller_fault")
    if summary.get("distance_m", 0) < min_distance_m:
        reasons.append("insufficient_progress")
    if summary.get("n_plans", 0) < 1:
        reasons.append("no_model_plans")
    assisted = bool(summary.get("aeb_ticks", 0) or summary.get("supervisor_ticks", 0))
    return {
        "safe_completion": not reasons,
        "unassisted_model_pass": (
            not reasons and not assisted and summary.get("alpamayo_used", False)
        ),
        "assisted": assisted,
        "failure_reasons": reasons,
        "min_distance_m": min_distance_m,
        "scope": "this simulation run only; not a real-world safety certification",
    }

/* Shared by the offline viewer and Node regression tests. No external dependencies. */
(function (root) {
  "use strict";
  function array(value, shape) {
    const data = value?.data ?? value;
    function valid(v, dims) {
      if (!dims.length) return Number.isFinite(v);
      return (
        Array.isArray(v) &&
        (dims[0] === null || v.length === dims[0]) &&
        v.every((item) => valid(item, dims.slice(1)))
      );
    }
    if (!valid(data, shape)) throw new Error("Invalid numeric array");
    return data;
  }
  function pose(d) {
    return {
      translation: array(d.translation, [3]),
      rotation: array(d.rotation, [3, 3]),
    };
  }
  function actor(d) {
    return {
      ...d,
      transform: pose(d.transform),
      velocity: array(d.velocity, [3]),
      type_id: String(d.type_id),
      bounding_box: {
        extent: array(d.bounding_box.extent, [3]),
        location: array(d.bounding_box.location, [3]),
      },
    };
  }
  function worldPoint(point, origin, flu = false) {
    const p = [point[0], flu ? -point[1] : point[1], point[2]];
    return origin.translation.map(
      (t, i) => t + origin.rotation[i].reduce((s, r, j) => s + r * p[j], 0),
    );
  }
  function parse(text) {
    const lines = text
      .replace(/^\uFEFF/, "")
      .split(/\r?\n/)
      .filter((line) => line.trim());
    if (lines.length < 2) throw new Error("Die Datei enthält keine Frames.");
    const header = JSON.parse(lines[0]);
    if (!(header.fixed_delta_seconds > 0) || header.frame_id !== undefined)
      throw new Error("Ungültiger Trace-Header.");
    const environment = (header.environment_objects ?? []).map(actor);
    const frames = [],
      byId = new Map(),
      plans = new Map();
    let skipped = 0,
      missingOrigins = 0;
    for (const line of lines.slice(1)) {
      try {
        const d = JSON.parse(line);
        if (
          !Number.isInteger(d.frame_id) ||
          !Number.isFinite(d.sim_time) ||
          (frames.length &&
            (d.sim_time <= frames.at(-1).sim_time || byId.has(d.frame_id)))
        )
          throw new Error("Invalid frame order");
        const f = {
          ...d,
          ego_pose_world: pose(d.ego_pose_world),
          model_pose_world: d.model_pose_world ? pose(d.model_pose_world) : null,
          actors: (d.actors ?? []).map(actor),
          lanes: (d.lanes ?? []).map((l) => ({
            ...l,
            polyline: array(l.polyline, [null, 3]),
          })),
          traffic_lights: (d.traffic_lights ?? []).map((l) => ({
            ...l,
            position: array(l.position, [3]),
          })),
          control: d.control ?? { steer: 0, throttle: 0, brake: 0 },
          diagnostics: d.diagnostics ?? {},
          plan: null,
        };
        if (d.plan) {
          const key = `${d.plan.frame_id}:${d.plan.model_config_hash}`;
          if (!plans.has(key))
            plans.set(key, {
              ...d.plan,
              waypoints_xyz: array(d.plan.waypoints_xyz, [null, 3]),
            });
          f.plan = plans.get(key);
        }
        byId.set(f.frame_id, f);
        frames.push(f);
      } catch (_) {
        skipped++;
      }
    }
    if (!frames.length) throw new Error("Keine gültigen Frames gefunden.");
    let latest = null,
      reasoning = null;
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      if (f.plan) {
        const origin = byId.get(f.plan.frame_id);
        if (origin && origin.sim_time <= f.sim_time) {
          // Repeated plans must keep the inference-frame pose, never the current ego pose.
          if (!f.plan.worldPoints) {
            f.plan.worldPoints = f.plan.waypoints_xyz.map((p) =>
              worldPoint(p, origin.model_pose_world ?? origin.ego_pose_world, true),
            );
            f.plan.originTime = origin.sim_time;
            f.plan.originPose = origin.model_pose_world ?? origin.ego_pose_world;
          }
          latest = f.plan;
          if (typeof f.plan.reasoning === "string" && f.plan.reasoning.trim())
            reasoning = f.plan;
        } else {
          missingOrigins++;
          latest = null;
        }
      }
      f.displayPlan = latest;
      f.displayReasoning = reasoning;
      const previous = frames[Math.max(0, i - 1)],
        dt = f.sim_time - previous.sim_time;
      f.speed = Number.isFinite(f.diagnostics.speed_mps)
        ? f.diagnostics.speed_mps
        : dt > 0
          ? Math.hypot(
              ...f.ego_pose_world.translation.map(
                (v, j) => v - previous.ego_pose_world.translation[j],
              ),
            ) / dt
          : 0;
    }
    return {
      header,
      frames,
      environment,
      skipped,
      missingOrigins,
      staticRecorded: Array.isArray(header.environment_objects),
    };
  }
  const api = { parse, pose, actor, worldPoint };
  if (typeof module !== "undefined") module.exports = api;
  root.TraceData = api;
})(typeof globalThis !== "undefined" ? globalThis : this);

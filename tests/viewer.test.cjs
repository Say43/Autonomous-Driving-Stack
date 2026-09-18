const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const data = require("../viewer/trace-data.js");
const origin = (x) => ({
  translation: [x, 0, 0],
  rotation: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
});
const plan = {
  frame_id: 0,
  model_config_hash: "model",
  waypoints_xyz: [
    [1, 2, 0],
    [2, 2, 0],
  ],
  reasoning: "left",
};
const frame = (id, p = plan) => ({
  frame_id: id,
  sim_time: id * 0.05,
  ego_pose_world: origin(id),
  actors: [],
  lanes: [],
  traffic_lights: [],
  plan: p,
});
const trace = (frames) =>
  [
    JSON.stringify({ fixed_delta_seconds: 0.05 }),
    ...frames.map((f) => (typeof f === "string" ? f : JSON.stringify(f))),
  ].join("\n");
test("repeated plan retains inference pose and FLU reflection", () => {
  const parsed = data.parse(trace([frame(0), frame(1), frame(2)]));
  assert.deepEqual(parsed.frames[2].displayPlan.worldPoints[0], [1, -2, 0]);
  assert.equal(parsed.frames[2].displayReasoning.originTime, 0);
});
test("missing origin never reanchors to a different vehicle pose", () => {
  const parsed = data.parse(trace([frame(1)]));
  assert.equal(parsed.frames[0].displayPlan, null);
  assert.equal(parsed.missingOrigins, 1);
});
test("malformed and nonfinite data is skipped, playback order preserved", () => {
  const bad = frame(1);
  bad.ego_pose_world.translation[0] = null;
  const parsed = data.parse(
    trace([frame(0), "{broken", bad, frame(2), frame(2)]),
  );
  assert.equal(parsed.skipped, 3);
  assert.equal(parsed.frames.length, 2);
});
test("sparse plans remain anchored between inference frames", () => {
  const parsed = data.parse(trace([frame(0), frame(1, null)]));
  assert.equal(parsed.frames[1].displayPlan.frame_id, 0);
});
test("legacy trace does not invent static object coverage", () => {
  assert.equal(data.parse(trace([frame(0)])).staticRecorded, false);
});

test("model-origin plans stay anchored while the ego mesh stays at the actor", () => {
  const first = frame(0);
  first.model_pose_world = origin(-0.6);
  const next = frame(1);
  next.model_pose_world = origin(0.4);
  const parsed = data.parse(trace([first, next]));
  assert.deepEqual(parsed.frames[0].ego_pose_world.translation, [0, 0, 0]);
  assert.deepEqual(parsed.frames[0].displayPlan.worldPoints[0], [0.4, -2, 0]);
  assert.deepEqual(parsed.frames[1].displayPlan.worldPoints[0], [0.4, -2, 0]);
});
test("rotated inference pose projects left correctly", () => {
  const pose = {
    translation: [10, 20, 0],
    rotation: [
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ],
  };
  assert.deepEqual(data.worldPoint([4, 2, 0], pose, true), [12, 24, 0]);
});
test("viewer renders demo and trace without nonfinite canvas geometry", () => {
  let polygons = 0;
  const ctx = new Proxy(
    {},
    {
      get: (_, name) =>
        name === "createLinearGradient"
          ? () => ({ addColorStop() {} })
          : (...args) => {
              if (name === "fill") polygons++;
              for (const a of args)
                if (typeof a === "number")
                  assert.ok(Number.isFinite(a), `${name}: nonfinite`);
            },
      set: () => true,
    },
  );
  const elements = new Map();
  function element(id) {
    if (!elements.has(id))
      elements.set(id, {
        style: {},
        classList: { toggle() {}, add() {}, remove() {} },
        setAttribute() {},
        addEventListener() {},
        getContext: () => ctx,
        getBoundingClientRect: () => ({ width: 1440, height: 900 }),
      });
    return elements.get(id);
  }
  const context = vm.createContext({
    TraceData: data,
    console,
    Math,
    Map,
    Set,
    Number,
    Array,
    String,
    Boolean,
    performance: { now: () => 0 },
    URLSearchParams,
    URL,
    location: { search: "", protocol: "file:" },
    requestAnimationFrame: () => 0,
    document: { getElementById: element, addEventListener() {} },
    window: { innerWidth: 1440, devicePixelRatio: 1, addEventListener() {} },
  });
  for (const file of ["scene.js", "viewer.js"])
    vm.runInContext(
      fs.readFileSync(path.join(__dirname, "../viewer", file), "utf8"),
      context,
    );
  assert.ok(polygons > 100, "3D demo meshes rendered");
  vm.runInContext(
    `state.data=TraceData.parse(${JSON.stringify(trace([frame(0), frame(1)]))});state.demo=false;setIndex(1);renderer.top=true;render();`,
    context,
  );
  assert.equal(elements.get("modeBadge").textContent, "AUFZEICHNUNG · REPLAY");
});

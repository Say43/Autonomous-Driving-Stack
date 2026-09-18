"use strict";
const $ = (id) => document.getElementById(id);
const renderer = new SceneRenderer($("scene"));
const state = {
  data: null,
  index: 0,
  playing: false,
  rate: 1,
  wallStart: 0,
  simStart: 0,
  loadToken: 0,
  demo: true,
  summary: null,
  details: window.innerWidth > 850,
};
function status(text, error = false) {
  $("loadStatus").textContent = text;
  $("loadStatus").classList.toggle("error", error);
}
function timeLabel(t) {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}
function pause() {
  state.playing = false;
  $("playPauseBtn").textContent = "▶";
  $("playPauseBtn").setAttribute("aria-label", "Abspielen");
}
function play() {
  if (!state.data) return;
  if (state.index >= state.data.frames.length - 1) state.index = 0;
  state.playing = true;
  state.wallStart = performance.now();
  state.simStart = state.data.frames[state.index].sim_time;
  $("playPauseBtn").textContent = "Ⅱ";
  $("playPauseBtn").setAttribute("aria-label", "Pause");
  render();
}
function toggle() {
  state.playing ? pause() : play();
}
function setIndex(index) {
  if (!state.data) return;
  state.index = Math.max(0, Math.min(state.data.frames.length - 1, index));
  render();
}
function render() {
  if (!state.data) return;
  const data = state.data,
    frame = data.frames[state.index];
  const testWorker = /synthetic|transport-smoke|fake/i.test(data.header.model_config_name ?? "");
  const objects = renderer.render(frame, data.environment, state.details);
  $("details").hidden = !state.details;
  $("detailsBtn").setAttribute("aria-expanded", String(state.details));
  $("speedValue").textContent = String(Math.round(frame.speed * 3.6));
  $("modeBadge").textContent = state.demo
    ? "DEMO · SYNTHETISCH"
    : testWorker ? "TEST-WORKER · REPLAY" : "AUFZEICHNUNG · REPLAY";
  $("mapName").textContent = (data.header.map_name ?? "Unbekannt")
    .split("/")
    .at(-1);
  $("frameLabel").textContent = `Frame ${frame.frame_id}`;
  $("timeLabel").textContent = `${frame.sim_time.toFixed(2)} s`;
  $("scrubber").max = data.frames.length - 1;
  $("scrubber").value = state.index;
  $("elapsed").textContent = timeLabel(
    frame.sim_time - data.frames[0].sim_time,
  );
  $("duration").textContent = timeLabel(
    data.frames.at(-1).sim_time - data.frames[0].sim_time,
  );
  const vehicles = objects.filter((a) =>
    /vehicle|static\.(car|truck|bus)/.test(a.type_id),
  ).length;
  const people = objects.filter((a) =>
    /walker|pedestrian/.test(a.type_id),
  ).length;
  $("vehiclesCount").textContent = vehicles;
  $("peopleCount").textContent = people;
  $("objectsCount").textContent =
    objects.length - vehicles - people + frame.traffic_lights.length;
  $("coverage").textContent = state.demo
    ? "Synthetische Vorschauszenen. Keine gemessene Fahrleistung."
    : data.staticRecorded
      ? "Dynamische und statische Objekte im Umkreis. 3D-Modelle aus aufgezeichneten Bounding-Boxen."
      : "Alter Trace: statische Objekte fehlen. Sichtbar sind nur aufgezeichnete Actors und Ampeln. Spurbreite ggf. angenähert.";
  const reason = frame.displayReasoning;
  $("reasoningText").textContent =
    reason?.reasoning ??
    "Für diesen Zeitpunkt liegt keine Modellbegründung vor.";
  $("reasoningSource").textContent = reason
    ? `${state.demo ? "Demo-Text" : testWorker ? "Test-Worker" : "Modellausgabe"} · Frame ${reason.frame_id} · ${(frame.sim_time - reason.originTime).toFixed(2)} s alt`
    : "";
  const plan = frame.displayPlan;
  $("driveState").textContent = state.demo
    ? "Design-Vorschau"
    : testWorker ? "Test-Worker im Replay" : frame.diagnostics.warmup
      ? "Sensor-Warm-up"
      : !plan
        ? "Kein Modellplan"
        : frame.sim_time - plan.originTime > 6.4
          ? "Plan abgelaufen"
          : "Modellplan im Replay";
  for (const key of ["steer", "throttle", "brake"]) {
    const value = Number.isFinite(frame.control[key]) ? frame.control[key] : 0;
    $(key + "Bar").style.width = `${Math.min(1, Math.abs(value)) * 100}%`;
    $(key + "Value").textContent = value.toFixed(2);
  }
  const d = frame.diagnostics,
    collision =
      d.collision || state.summary?.collision_frames?.includes(d.carla_frame);
  const warning = collision
    ? "Kontakt aufgezeichnet"
    : d.offroad || d.footprint_offroad
      ? "Außerhalb einer Driving-Lane"
      : d.supervisor?.brake
        ? `Schutzbremse · ${d.supervisor.reason}`
        : d.aeb
          ? "Notbremse · Simulator-Eingriff"
          : "";
  $("intervention").hidden = !warning;
  $("intervention").textContent = warning;
  $("intervention").classList.toggle("danger", !!collision || !!d.offroad || !!d.footprint_offroad);
  const summary = state.summary;
  $("runResult").textContent = summary
    ? `${summary.n_collision_frames ?? 0} Kollisionsframes · ${summary.offroad_ticks ?? 0} Offroad-Ticks · ${summary.aeb_ticks ?? 0} AEB-Ticks${summary.supervisor_ticks !== undefined ? ` · ${summary.supervisor_ticks} Schutz-Ticks` : ""}. ${summary.evaluation?.safe_completion ? "Sicherheitskriterien dieses Laufs erfüllt." : "Laufabschluss allein belegt keine sichere Fahrt."}`
    : state.demo
      ? "Vorschau · keine Verbindung zum Simulator"
      : "Replay · Sicherheitsstatus nur mit aufgezeichneten Diagnosen belegbar.";
}
async function loadText(text, label, token) {
  await new Promise((resolve) => requestAnimationFrame(resolve));
  if (token !== state.loadToken) return;
  const data = TraceData.parse(text);
  if (token !== state.loadToken) return;
  state.data = data;
  state.index = 0;
  state.demo = false;
  state.summary = null;
  pause();
  status(
    `${label} · ${data.frames.length} Frames${data.skipped ? ` · ${data.skipped} ungültige Zeilen übersprungen` : ""}${data.missingOrigins ? ` · ${data.missingOrigins} Pläne ohne Ursprung` : ""}`,
    data.skipped > 0 || data.missingOrigins > 0,
  );
  render();
}
async function loadFile(file) {
  if (!file) return;
  pause();
  const token = ++state.loadToken;
  status(`${file.name} wird geladen …`);
  try {
    await loadText(await file.text(), file.name, token);
  } catch (error) {
    if (token === state.loadToken) status(error.message, true);
  }
}
$("fileInput").addEventListener("change", (e) => loadFile(e.target.files[0]));
for (const event of ["dragenter", "dragover"])
  document.addEventListener(event, (e) => {
    e.preventDefault();
    $("dropZone").classList.add("dragover");
  });
document.addEventListener("dragleave", () =>
  $("dropZone").classList.remove("dragover"),
);
document.addEventListener("drop", (e) => {
  e.preventDefault();
  $("dropZone").classList.remove("dragover");
  loadFile(e.dataTransfer.files[0]);
});
$("playPauseBtn").onclick = toggle;
$("stepBackBtn").onclick = () => {
  pause();
  setIndex(state.index - 1);
};
$("stepFwdBtn").onclick = () => {
  pause();
  setIndex(state.index + 1);
};
$("scrubber").oninput = (e) => {
  pause();
  setIndex(Number(e.target.value));
};
$("playbackRate").onchange = (e) => {
  state.rate = Number(e.target.value);
  if (state.playing) play();
};
$("detailsBtn").onclick = () => {
  state.details = !state.details;
  render();
};
function setView(top) {
  renderer.top = top;
  $("rearBtn").classList.toggle("selected", !top);
  $("topBtn").classList.toggle("selected", top);
  render();
}
$("rearBtn").onclick = () => setView(false);
$("topBtn").onclick = () => setView(true);
$("resetViewBtn").onclick = () => {
  renderer.orbit = 0;
  renderer.zoom = 1;
  setView(false);
};
let dragging = null;
$("scene").addEventListener("pointerdown", (e) => {
  dragging = { x: e.clientX, orbit: renderer.orbit };
  e.target.setPointerCapture(e.pointerId);
});
$("scene").addEventListener("pointermove", (e) => {
  if (dragging) {
    renderer.orbit = dragging.orbit + (e.clientX - dragging.x) * 0.005;
    render();
  }
});
for (const event of ["pointerup", "pointercancel", "lostpointercapture"])
  $("scene").addEventListener(event, () => {
    dragging = null;
  });
$("scene").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    renderer.zoom = Math.max(
      0.5,
      Math.min(2, renderer.zoom * Math.exp(-e.deltaY * 0.001)),
    );
    render();
  },
  { passive: false },
);
window.addEventListener("resize", render);
window.addEventListener("keydown", (e) => {
  if (/INPUT|SELECT|BUTTON/.test(e.target.tagName)) return;
  if (e.code === "Space") {
    e.preventDefault();
    toggle();
  }
  if (e.code === "ArrowLeft" || e.code === "ArrowRight") {
    e.preventDefault();
    pause();
    setIndex(state.index + (e.code === "ArrowLeft" ? -1 : 1));
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pause();
});
function animate(now) {
  if (state.playing && state.data) {
    const frames = state.data.frames;
    const desired =
      state.simStart + ((now - state.wallStart) / 1000) * state.rate;
    let idx = state.index;
    while (idx < frames.length - 1 && frames[idx + 1].sim_time <= desired)
      idx++;
    if (idx !== state.index) setIndex(idx);
    if (idx === frames.length - 1) pause();
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

function demo() {
  ++state.loadToken;
  pause();
  state.demo = true;
  state.summary = null;
  state.index = 0;
  const rot = (y) => [
    [Math.cos(y), -Math.sin(y), 0],
    [Math.sin(y), Math.cos(y), 0],
    [0, 0, 1],
  ];
  const pose = (x, y, z = 0, yaw = 0) => ({
    translation: [x, y, z],
    rotation: rot(yaw),
  });
  const actor = (id, type, x, y, l = 2.2, w = 0.95, h = 0.75) => ({
    id,
    type_id: type,
    transform: pose(x, y),
    velocity: [0, 0, 0],
    bounding_box: { extent: [l, w, h], location: [0, 0, h] },
  });
  const environment = [];
  for (let i = 0; i < 10; i++) {
    environment.push(
      actor(-i * 2 - 1, "static.vegetation", i * 18 - 15, -9, 0.9, 0.9, 3.1),
    );
    environment.push(
      actor(-i * 2 - 2, "static.vegetation", i * 18 - 8, 9, 0.9, 0.9, 3.1),
    );
  }
  environment.push(actor(-40, "static.vehicles", 40, 6.4));
  environment.push(actor(-41, "static.buildings", 46, 22, 12, 6, 6));
  const lanes = [-3.6, 0, 3.6].map((y, i) => ({
    lane_id: i,
    width_m: 3.6,
    polyline: Array.from({ length: 130 }, (_, j) => [j * 2 - 30, y, 0]),
  }));
  const frames = [];
  for (let i = 0; i < 600; i++) {
    const t = i * 0.05,
      x = t * 6.4,
      ego = pose(x, 0);
    const plan = {
      frame_id: i,
      originTime: t,
      originPose: ego,
      reasoning:
        "Synthetische Vorschau: der Spur folgen und Abstand zum vorausfahrenden Fahrzeug halten.",
      worldPoints: Array.from({ length: 64 }, (_, j) => [x + j * 0.6, 0, 0]),
    };
    frames.push({
      frame_id: i,
      sim_time: t,
      ego_pose_world: ego,
      speed: 6.4,
      lanes,
      actors: [
        actor(1, "vehicle.sedan", x + 23, 0),
        actor(2, "vehicle.suv", x + 11, -3.6, 2.4, 1.05, 0.9),
        actor(3, "vehicle.sedan", x + 44, 3.6),
        actor(4, "vehicle.sedan", x + 59, -3.6),
        actor(5, "walker.pedestrian", x + 25, 7.2, 0.25, 0.25, 0.87),
      ],
      traffic_lights: [{ id: 1, state: "green", position: [82, 6.5, 4] }],
      displayPlan: plan,
      displayReasoning: plan,
      control: { steer: 0, throttle: 0.16, brake: 0 },
      diagnostics: {},
    });
  }
  state.data = {
    header: { map_name: "DESIGN DEMO" },
    frames,
    environment,
    staticRecorded: true,
    skipped: 0,
  };
  status("Offline-Vorschau · eigenen Trace öffnen für aufgezeichnete Fahrten");
  render();
}
$("demoBtn").onclick = demo;
demo();
// Optional localhost server. file:// mode continues to use only the File API.
async function loadFromUrl() {
  const params = new URLSearchParams(location.search),
    path = params.get("trace");
  if (!path || !/^https?:$/.test(location.protocol)) return;
  const url = new URL(path, location.href);
  if (url.origin !== location.origin) {
    status("Nur Trace-URLs vom selben Server sind erlaubt.", true);
    return;
  }
  const token = ++state.loadToken;
  status("Aufzeichnung wird geladen …");
  try {
    const response = await fetch(url);
    if (!response.ok)
      throw new Error(`Trace nicht erreichbar (${response.status})`);
    await loadText(
      await response.text(),
      url.pathname.split("/").at(-1),
      token,
    );
    if (token !== state.loadToken) return;
    if (params.has("t")) {
      const requested = Number(params.get("t"));
      if (Number.isFinite(requested)) {
        const index = state.data.frames.findIndex(
          (f) => f.sim_time >= requested,
        );
        setIndex(index < 0 ? state.data.frames.length - 1 : index);
      }
    }
    const summaryPath = params.get("summary");
    if (summaryPath) {
      const u = new URL(summaryPath, location.href);
      if (u.origin === location.origin) {
        const s = await fetch(u);
        if (s.ok) {
          const summary = await s.json();
          if (token === state.loadToken) {
            state.summary = summary;
            render();
          }
        }
      }
    }
  } catch (error) {
    if (token === state.loadToken) status(error.message, true);
  }
}
loadFromUrl();

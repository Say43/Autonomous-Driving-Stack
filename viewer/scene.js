/* Dependency-free 3D mesh renderer. World coordinates stay in CARLA's frame.
 * Perspective camera, near-plane clipping, depth-sorted shaded faces and fog.
 * Vehicle meshes are visual proxies scaled to recorded bounding boxes, not perception.
 */
"use strict";
class SceneRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.orbit = 0;
    this.zoom = 1;
    this.top = false;
    this.faces = [];
  }
  cameraPoint(p) {
    const dx = p[0] - this.ego.translation[0],
      dy = p[1] - this.ego.translation[1];
    const forward = dx * this.cos + dy * this.sin,
      right = -dx * this.sin + dy * this.cos;
    const z = p[2] - this.ego.translation[2];
    if (this.top) return [right, forward, 100 - z];
    const f = forward + 11 / this.zoom,
      h = z - 8 / this.zoom;
    return [
      right,
      f * this.tiltSin + h * this.tiltCos,
      f * this.tiltCos - h * this.tiltSin,
    ];
  }
  screen(p) {
    if (this.top)
      return [
        this.cx + p[0] * 9 * this.zoom,
        this.h * 0.62 - p[1] * 9 * this.zoom,
      ];
    return [
      this.cx + (this.focal * p[0]) / p[2],
      this.h * 0.36 - (this.focal * p[1]) / p[2],
    ];
  }
  polygon(points, fill, stroke = null, width = 0.65) {
    let vertices = points.map((p) => this.cameraPoint(p));
    if (!this.top) {
      const clipped = [];
      for (let i = 0; i < vertices.length; i++) {
        const a = vertices[i],
          b = vertices[(i + 1) % vertices.length];
        if (a[2] >= 0.8) clipped.push(a);
        if (a[2] >= 0.8 !== b[2] >= 0.8) {
          const t = (0.8 - a[2]) / (b[2] - a[2]);
          clipped.push(a.map((v, j) => v + (b[j] - v) * t));
        }
      }
      vertices = clipped;
    }
    if (vertices.length < 3) return;
    const screen = vertices.map((p) => this.screen(p));
    if (
      screen.every((p) => p[0] < -20) ||
      screen.every((p) => p[0] > this.w + 20) ||
      screen.every((p) => p[1] < -20) ||
      screen.every((p) => p[1] > this.h + 20)
    )
      return;
    this.faces.push({
      screen,
      fill,
      stroke,
      width,
      depth: vertices.reduce((s, p) => s + p[2], 0) / vertices.length,
    });
  }
  ribbon(points, width, color) {
    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i],
        b = points[i + 1],
        dx = b[0] - a[0],
        dy = b[1] - a[1];
      const length = Math.hypot(dx, dy);
      if (length < 0.001 || length > 12) continue;
      const nx = ((-dy / length) * width) / 2,
        ny = ((dx / length) * width) / 2;
      this.polygon(
        [
          [a[0] + nx, a[1] + ny, a[2]],
          [b[0] + nx, b[1] + ny, b[2]],
          [b[0] - nx, b[1] - ny, b[2]],
          [a[0] - nx, a[1] - ny, a[2]],
        ],
        color,
      );
    }
  }
  flush() {
    const ctx = this.ctx;
    this.faces.sort((a, b) => b.depth - a.depth);
    for (const f of this.faces) {
      ctx.globalAlpha = this.top
        ? 1
        : Math.max(0.07, Math.min(1, (115 - f.depth) / 65));
      ctx.beginPath();
      f.screen.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
      ctx.closePath();
      if (f.fill) {
        ctx.fillStyle = f.fill;
        ctx.fill();
      }
      if (f.stroke) {
        ctx.strokeStyle = f.stroke;
        ctx.lineWidth = f.width;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    this.faces = [];
  }
  local(p, transform) {
    return TraceData.worldPoint(p, transform);
  }
  box(transform, center, extents, palette, outline = "#c0c6cc") {
    const vertices = [];
    for (const z of [-1, 1])
      for (const [x, y] of [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ]) {
        vertices.push(
          this.local(
            [
              center[0] + x * extents[0],
              center[1] + y * extents[1],
              center[2] + z * extents[2],
            ],
            transform,
          ),
        );
      }
    const faces = [
      [0, 1, 2, 3],
      [0, 4, 5, 1],
      [1, 5, 6, 2],
      [2, 6, 7, 3],
      [3, 7, 4, 0],
      [4, 7, 6, 5],
    ];
    faces.forEach((ids, i) =>
      this.polygon(
        ids.map((j) => vertices[j]),
        palette[i % palette.length],
        outline,
      ),
    );
  }
  vehicle(actor, ego = false) {
    const { extent: e, location: b } = actor.bounding_box,
      transform = actor.transform;
    const l = e[0],
      w = e[1],
      h = Math.max(e[2] * 2, 1.1),
      z = b[2] - e[2];
    const body = ego
      ? ["#bac0c7", "#e5e8ec", "#edf0f3", "#dce1e6", "#ccd3db", "#fff"]
      : ["#919ba5", "#b6bfc8", "#c8d0d7", "#aeb8c3", "#a8b3be", "#dce1e7"];
    // Flattened shadow on the local ground plane.
    this.polygon(
      [
        [-l - 0.16, -w - 0.1, z + 0.03],
        [l + 0.16, -w - 0.1, z + 0.03],
        [l + 0.16, w + 0.1, z + 0.03],
        [-l - 0.16, w + 0.1, z + 0.03],
      ].map((p) => this.local([p[0] + b[0], p[1] + b[1], p[2]], transform)),
      "#46526417",
    );
    // Rounded body rings create a sculpted silhouette instead of an actor-sized cuboid.
    const ring = (sx, sy, height) => {
      const pts = [],
        radius = Math.min(w * 0.25, l * 0.13);
      for (const [cx, cy, start] of [
        [l * sx - radius, w * sy - radius, 0],
        [-l * sx + radius, w * sy - radius, Math.PI / 2],
        [-l * sx + radius, -w * sy + radius, Math.PI],
        [l * sx - radius, -w * sy + radius, Math.PI * 1.5],
      ]) {
        for (let j = 0; j < 4; j++) {
          const a = start + (j * Math.PI) / 6;
          pts.push(
            this.local(
              [
                b[0] + cx + Math.cos(a) * radius,
                b[1] + cy + Math.sin(a) * radius,
                z + height,
              ],
              transform,
            ),
          );
        }
      }
      return pts;
    };
    const rings = [
      ring(0.91, 0.9, h * 0.18),
      ring(0.98, 1, h * 0.34),
      ring(0.96, 0.94, h * 0.53),
      ring(0.87, 0.87, h * 0.58),
    ];
    for (let k = 0; k < rings.length - 1; k++)
      for (let i = 0; i < 16; i++) {
        const j = (i + 1) % 16;
        this.polygon(
          [rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]],
          body[(Math.floor(i / 4) + k) % body.length],
        );
      }
    this.polygon(rings[3], body[5], ego ? "#cbd2da" : "#a5b0bc", 0.45);
    // Tapered glasshouse: a genuine 3D frustum with a sloping rear/front windshield.
    const bottom = [
      [-l * 0.66, -w * 0.89, z + h * 0.55],
      [l * 0.5, -w * 0.89, z + h * 0.55],
      [l * 0.5, w * 0.89, z + h * 0.55],
      [-l * 0.66, w * 0.89, z + h * 0.55],
    ];
    const top = [
      [-l * 0.35, -w * 0.72, z + h * 0.94],
      [l * 0.22, -w * 0.72, z + h * 0.94],
      [l * 0.22, w * 0.72, z + h * 0.94],
      [-l * 0.35, w * 0.72, z + h * 0.94],
    ];
    const v = [...bottom, ...top].map((p) =>
      this.local([p[0] + b[0], p[1] + b[1], p[2]], transform),
    );
    [
      [0, 1, 5, 4],
      [1, 2, 6, 5],
      [2, 3, 7, 6],
      [3, 0, 4, 7],
    ].forEach((ids, i) =>
      this.polygon(
        ids.map((j) => v[j]),
        ego ? ["#526172", "#617388", "#728294", "#465466"][i] : "#778591",
        "#aeb7c2",
      ),
    );
    this.polygon(
      [v[4], v[5], v[6], v[7]],
      ego ? "#e9edf2" : "#c7cfd8",
      "#aeb7c2",
    );
    // Narrow pillars and mirrors give scale without turning the visual into a bounding-box debug view.
    for (const side of [-1, 1])
      this.box(
        transform,
        [b[0] + l * 0.34, b[1] + side * w * 1.04, z + h * 0.59],
        [l * 0.08, w * 0.13, h * 0.04],
        [ego ? "#e8edf3" : "#b5c0ca"],
        null,
      );
    // Four wheels, oriented on the vehicle's lateral axle.
    for (const x of [-l * 0.62, l * 0.6])
      for (const side of [-1, 1]) {
        const ring = [];
        for (let j = 0; j < 12; j++) {
          const a = (j * Math.PI) / 6;
          ring.push(
            this.local(
              [
                b[0] + x + Math.cos(a) * h * 0.2,
                b[1] + side * (w + 0.025),
                z + h * 0.23 + Math.sin(a) * h * 0.2,
              ],
              transform,
            ),
          );
        }
        this.polygon(ring, ego ? "#38414c" : "#56616c", "#68717a");
        const hub = ring.map((p) =>
          p.map(
            (n, j) =>
              n * 0.55 +
              this.local(
                [b[0] + x, b[1] + side * (w + 0.03), z + h * 0.23],
                transform,
              )[j] *
                0.45,
          ),
        );
        this.polygon(hub, ego ? "#b8c1cb" : "#9ea9b4");
      }
    for (const side of [-1, 1]) {
      this.box(
        transform,
        [b[0] - l * 0.966, b[1] + side * w * 0.62, z + h * 0.43],
        [0.015, w * 0.24, h * 0.022],
        [ego ? "#d96363" : "#b38e8e"],
        null,
      );
      this.box(
        transform,
        [b[0] + l * 0.966, b[1] + side * w * 0.62, z + h * 0.43],
        [0.015, w * 0.25, h * 0.035],
        ["#f8fbff"],
        null,
      );
    }
  }
  person(actor) {
    const t = actor.transform,
      b = actor.bounding_box.location,
      h = actor.bounding_box.extent[2] * 2;
    const z = b[2] - h / 2;
    this.box(
      t,
      [b[0], b[1], z + h * 0.58],
      [0.2, 0.13, h * 0.24],
      ["#9aa5b0", "#b6bfc7", "#c2cad2"],
      null,
    );
    for (const side of [-1, 1])
      this.box(
        t,
        [b[0], b[1] + side * 0.12, z + h * 0.2],
        [0.08, 0.08, h * 0.2],
        ["#8c99a6", "#a1acb8"],
        null,
      );
    this.box(
      t,
      [b[0], b[1], z + h * 0.9],
      [0.12, 0.12, h * 0.1],
      ["#bdc6ce", "#d6dce2"],
      null,
    );
  }
  object(actor) {
    const type = actor.type_id.toLowerCase();
    if (
      type.startsWith("vehicle.") ||
      /static\.(vehicles|car|cars|truck|bus)/.test(type)
    )
      return this.vehicle(actor);
    if (type.startsWith("walker.") || type.includes("pedestrian"))
      return this.person(actor);
    const building = type.includes("building"),
      vegetation = type.includes("vegetation");
    const e = actor.bounding_box.extent,
      b = actor.bounding_box.location;
    if (vegetation) {
      this.box(
        actor.transform,
        [b[0], b[1], b[2] - e[2] * 0.35],
        [0.15, 0.15, e[2] * 0.65],
        ["#bdc6c9", "#d2d9da"],
        null,
      );
      this.box(
        actor.transform,
        [b[0], b[1], b[2] + e[2] * 0.4],
        [e[0] * 0.7, e[1] * 0.7, e[2] * 0.4],
        ["#dce3e2", "#edf0ef"],
        null,
      );
    } else
      this.box(
        actor.transform,
        b,
        e,
        building
          ? ["#edf0f240", "#e5e9ed45", "#f4f6f865"]
          : ["#b8c1cb", "#d1d8df", "#e4e9ed"],
        building ? "#d8dfe54a" : "#c5ced6",
      );
  }
  trafficLight(light) {
    const z = this.ego.translation[2],
      [x, y] = light.position;
    const t = {
      translation: [x, y, z],
      rotation: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    };
    this.box(t, [0, 0, 2.0], [0.045, 0.045, 2], ["#a4afb9"], null);
    this.box(t, [0, 0, 4.05], [0.15, 0.15, 0.42], ["#66727e", "#81909e"], null);
    const p = this.cameraPoint([x, y, z + 4.1]);
    if (p[2] > 0.8) {
      const [sx, sy] = this.screen(p);
      this.lightDots.push({
        sx,
        sy,
        r: this.top ? 4 : Math.max(2, Math.min(7, 95 / p[2])),
        color:
          { red: "#e47372", yellow: "#e4b75c", green: "#69ab8d" }[
            light.state
          ] ?? "#a1acb5",
      });
    }
  }
  render(frame, environment, detailsVisible) {
    const rect = this.canvas.getBoundingClientRect(),
      dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = rect.width;
    this.h = rect.height;
    if (
      this.canvas.width !== Math.round(this.w * dpr) ||
      this.canvas.height !== Math.round(this.h * dpr)
    ) {
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, this.w, this.h);
    this.ego = frame.ego_pose_world;
    const yaw =
      Math.atan2(this.ego.rotation[1][0], this.ego.rotation[0][0]) + this.orbit;
    this.cos = Math.cos(yaw);
    this.sin = Math.sin(yaw);
    this.tiltSin = 0.3;
    this.tiltCos = Math.sqrt(1 - 0.3 ** 2);
    this.cx = (this.w - (detailsVisible && this.w > 700 ? 285 : 0)) * 0.52;
    this.focal = Math.min(this.h * 1.04, this.w * 1.15);
    this.faces = [];
    this.lightDots = [];
    for (const lane of frame.lanes) {
      const pts = lane.polyline.map((p) => [p[0], p[1], p[2] + 0.03]);
      this.ribbon(pts, lane.width_m ?? 3.5, "#edefef");
    }
    this.flush();
    for (const lane of frame.lanes) {
      const pts = lane.polyline,
        width = lane.width_m ?? 3.5;
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i],
          b = pts[i + 1],
          dx = b[0] - a[0],
          dy = b[1] - a[1],
          len = Math.hypot(dx, dy);
        if (len < 0.01 || len > 6) continue;
        for (const sign of [-1, 1]) {
          const nx = (((-dy / len) * width) / 2) * sign,
            ny = (((dx / len) * width) / 2) * sign;
          this.ribbon(
            [
              [a[0] + nx, a[1] + ny, a[2] + 0.055],
              [b[0] + nx, b[1] + ny, b[2] + 0.055],
            ],
            0.065,
            "#c1c6cc",
          );
        }
      }
    }
    this.flush();
    const plan = frame.displayPlan;
    if (plan && frame.sim_time - plan.originTime <= 6.4) {
      const path = plan.worldPoints.map((p) => [p[0], p[1], p[2] + 0.1]);
      this.ribbon(path, 0.65, "#3395f326");
      this.flush();
      this.ribbon(path, 0.16, "#318df0");
      this.flush();
    }
    const objects = [...environment, ...frame.actors].filter(
      (a) =>
        Math.hypot(
          a.transform.translation[0] - this.ego.translation[0],
          a.transform.translation[1] - this.ego.translation[1],
        ) <
        75 + Math.max(...a.bounding_box.extent),
    );
    objects.forEach((a) => this.object(a));
    frame.traffic_lights.forEach((l) => this.trafficLight(l));
    const ego = {
      transform: this.ego,
      bounding_box: { extent: [2.35, 1.0, 0.76], location: [0, 0, 0.76] },
    };
    this.vehicle(ego, true);
    this.flush();
    for (const l of this.lightDots) {
      ctx.beginPath();
      ctx.arc(l.sx, l.sy, l.r, 0, Math.PI * 2);
      ctx.fillStyle = l.color;
      ctx.fill();
    }
    // Distance fog fades the geometric horizon into the white interface.
    if (!this.top) {
      const fog = ctx.createLinearGradient(0, this.h * 0.14, 0, this.h * 0.34);
      fog.addColorStop(0, "#fafafa");
      fog.addColorStop(1, "#fafafa00");
      ctx.fillStyle = fog;
      ctx.fillRect(0, 0, this.w, this.h * 0.34);
    }
    return objects;
  }
}

const RAD = Math.PI / 180;
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const scale = (a, s) => a.map(v => v * s);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => scale(a, 1 / (Math.hypot(...a) || 1));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export const JOINTS = ["extension_m", "lift_m", "swing_deg", "pitch_deg"];
export const PHASE = {ready: "待机", approach: "高位伸入", dip: "入渣", skim: "贴面", sweep: "回拉扒渣",
  discharge: "越口排渣", lift: "抬起", retreat: "退出", tilt: "倾罐"};
const INK = {
  slag: ["#f099ce", "#c75a9b4f"], iron: ["#ffd179", "#ffc05553"],
  tank: ["#b5c2d1", "transparent"], tool: ["#65ddff", "#35b9ea59"],
  steam: ["#ced7e2aa", "#c8d5e427"], unknown: ["#8793a77a", "#9195ab22"],
};

function headPosition(arm, q) {
  return add(arm.base_m, [q[0] * Math.sin(q[2] * RAD), q[0] * Math.cos(q[2] * RAD), q[1]]);
}

export function initialPose(cfg) {
  const q = [cfg.arm.extension, cfg.arm.lift, cfg.arm.swing, cfg.arm.pitch].map(axis => axis.initial);
  return {
    tilt_deg: cfg.tilt.initial_deg,
    joints: Object.fromEntries(JOINTS.map((key, i) => [key, q[i]])),
    head_m: headPosition(cfg.arm, q),
  };
}

export function recordedPose(cfg, reconstruction) {
  const pose = initialPose(cfg);
  pose.head_m = null;
  if (!reconstruction) return pose;
  pose.tilt_deg = reconstruction.tilt_deg;
  pose.bath = reconstruction.bath;
  const xy = reconstruction.tool_xy;
  if (xy && pose.bath) {
    const {center_xy, radii_m, z_m} = pose.bath;
    const [a, b] = cfg.tank.inner_radii_m;
    pose.head_m = [center_xy[0] + xy[0] / a * radii_m[0],
      center_xy[1] + xy[1] / b * radii_m[1], z_m + cfg.arm.tool_size_m[2] / 2 + .075];
    const [dx, dy, dz] = sub(pose.head_m, cfg.arm.base_m);
    pose.joints = {extension_m:Math.hypot(dx, dy), lift_m:dz, swing_deg:Math.atan2(dx, dy) / RAD, pitch_deg:0};
  }
  return pose;
}

export function trajectoryPose(cfg, path, t) {
  const knots = path.trajectory;
  const next = knots.findIndex(knot => knot.t_s > t);
  const index = next < 0 ? knots.length - 1 : Math.max(1, next);
  const [from, to] = [knots[index - 1], knots[index]];
  const u = clamp((t - from.t_s) / (to.t_s - from.t_s), 0, 1);
  const blend = u * u * (3 - 2 * u);
  const q = JOINTS.map(key => from.joints[key] + (to.joints[key] - from.joints[key]) * blend);
  const tilt = from.tilt_deg + (to.tilt_deg - from.tilt_deg) * blend;
  const bath = {
    center_xy:add(from.bath.center_xy, scale(sub(to.bath.center_xy, from.bath.center_xy), blend)),
    radii_m:add(from.bath.radii_m, scale(sub(to.bath.radii_m, from.bath.radii_m), blend)),
    z_m:from.bath.z_m + (to.bath.z_m - from.bath.z_m) * blend,
  };
  return {joints:Object.fromEntries(JOINTS.map((key, i) => [key, q[i]])),
    head_m:headPosition(cfg.arm, q), tilt_deg:tilt, bath, phase:to.phase};
}

function trajectorySegments(path, arm) {
  const knots = path.trajectory;
  return knots.slice(1).map((knot, i) => {
    const a = JOINTS.map(key => knots[i].joints[key]);
    const b = JOINTS.map(key => knot.joints[key]);
    const points = Array.from({length:13}, (_, j) => {
      const u = j / 12, easing = u * u * (3 - 2 * u);
      return headPosition(arm, a.map((q, k) => q + easing * (b[k] - q)));
    });
    return {phase:knot.phase, points};
  });
}

function dimensions(canvas) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return null;
  const ratio = Math.min(2, devicePixelRatio || 1);
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, w, h};
}

function trace(ctx, points, closed = true) {
  if (!points.length) return;
  ctx.beginPath();
  points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  if (closed) ctx.closePath();
}

function outline(ctx, points, color, width = 1, dash = []) {
  trace(ctx, points, false);
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.setLineDash([]);
}

function localTransform(tank, angle) {
  const c = Math.cos(angle * RAD), s = Math.sin(angle * RAD);
  return ([x, y, z]) => add(tank.pivot_m, [x, y * c - z * s, y * s + z * c]);
}

function virtualProject(camera, xyz) {
  let p;
  if (camera.device_to_camera) {
    p = camera.device_to_camera.slice(0, 3).map(row => dot(row, [...xyz, 1]));
  } else {
    const f = unit(sub(camera.look_at_m, camera.position_m));
    const r = unit(sub([1, 0, 0], scale(f, f[0])));
    const d = cross(f, r), v = sub(xyz, camera.position_m);
    p = [dot(r, v), dot(d, v), dot(f, v)];
  }
  const [fx, fy, cx, cy] = camera.intrinsics;
  return p[2] > .07 ? [p[0] / p[2] * fx + cx, p[1] / p[2] * fy + cy] : [-9999, -9999];
}

const images = new Map();

function cachedImage(src) {
  if (!src) return null;
  if (images.has(src)) return images.get(src);
  if (images.size >= 72) images.delete(images.keys().next().value);
  const img = new Image();
  img.src = src;
  images.set(src, img);
  return img;
}

export class CameraView {
  constructor(canvas) {
    this.canvas = canvas;
    this.sample = null;
    this.displayed = null;
    this.image = null;
    this.mask = true;
    this.frameKey = 0;
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
  }
  dispose() {
    this.observer.disconnect();
  }
  set(sample, path, cfg, mask = true, recording = false, onPresent = null) {
    const source = sample?.image_url;
    const changed = source !== this.sample?.image_url;
    this.sample = sample;
    this.path = path;
    this.cfg = cfg;
    this.mask = mask;
    this.recording = recording;
    this.onPresent = onPresent;
    if (!changed) {
      if (source === this.displayed?.image_url) this.present(this.image);
      return;
    }
    const token = ++this.frameKey;
    const image = recording && source ? new Image() : cachedImage(source);
    if (recording && source) image.src = source;
    if (image?.complete && image.naturalWidth) {
      this.present(image);
    } else if (image) {
      image.addEventListener("load", () => {
        if (token !== this.frameKey) return;
        this.present(image);
      }, {once: true});
    } else {
      this.present(null);
    }
  }
  present(image) {
    this.image = image;
    this.displayed = this.sample;
    this.draw();
    this.onPresent?.(this.sample);
  }
  draw() {
    const dims = dimensions(this.canvas);
    if (!dims) return;
    const {ctx, w, h} = dims;
    ctx.fillStyle = "#0a1016";
    ctx.fillRect(0, 0, w, h);
    const img = this.image;
    if (!img?.complete || !img.naturalWidth) {
      ctx.fillStyle = "#788b9a";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("等待相机画面", w / 2, h / 2);
      ctx.textAlign = "left";
      return;
    }
    const s = Math.min(w / img.naturalWidth, h / img.naturalHeight);
    const iw = s * img.naturalWidth, ih = s * img.naturalHeight;
    const x = (w - iw) / 2, y = (h - ih) / 2;
    ctx.drawImage(img, x, y, iw, ih);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, iw, ih);
    ctx.clip();
    const obs = this.displayed?.observation;
    const transform = ([px, py]) => [x + px * s, y + py * s];
    const regions = (items, [stroke, fill], lineWidth = 1.4, dash = []) => {
      ctx.strokeStyle = stroke;
      ctx.fillStyle = fill;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dash);
      for (const rings of items || []) {
        ctx.beginPath();
        for (const polygon of rings) {
          polygon.map(transform).forEach(([px, py], i) => i ? ctx.lineTo(px, py) : ctx.moveTo(px, py));
          ctx.closePath();
        }
        ctx.fill("evenodd");
        ctx.stroke();
      }
      ctx.setLineDash([]);
    };
    const simple = (items, ...styling) => regions(items?.map(points => [points]), ...styling);
    if (this.mask && obs) {
      simple(obs.mouth, INK.tank, 1.55, [6, 4]);
      if (obs.iron_regions) regions(obs.iron_regions, INK.iron);
      else simple(obs.iron || obs.liquid, INK.iron, 1);
      if (obs.slag_regions) regions(obs.slag_regions, INK.slag);
      else simple(obs.slag.map(p => p.contour_px), INK.slag);
      simple(obs.steam, INK.steam, 1, [3, 3]);
      simple(obs.unknown, INK.unknown, 1, [3, 3]);
      simple(obs.tool, INK.tool, 1.85);
      const labels = [];
      const tag = (text, px, py, foreground, background) => {
        ctx.font = "10px 'Segoe UI', 'Microsoft YaHei', sans-serif";
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = background;
        ctx.fillRect(px + 4, py - 14, tw + 9, 16);
        ctx.fillStyle = foreground;
        ctx.fillText(text, px + 8, py - 3);
      };
      if (obs.tool_center_px && obs.tool.length) {
        const [px, py] = transform(obs.tool_center_px);
        tag("机械臂", px, py, "#aeeeff", "#0e3a4def");
        labels.push([px, py]);
      }
      for (const patch of obs.slag.slice(0, this.recording ? 20 : 14)) {
        if (patch.pixel_area > 65) {
          const [px, py] = transform(patch.center_px);
          if (labels.some(([lx, ly]) => Math.abs(lx - px) < 47 && Math.abs(ly - py) < 17)) continue;
          labels.push([px, py]);
          tag(patch.id, px, py, "#ffd3ee", "#512c48ee");
          const v = patch.velocity_px_s;
          if (Math.hypot(...v) > .7) {
            outline(ctx, [[px, py], [px + clamp(v[0], -18, 18), py + clamp(v[1], -18, 18)]], "#f4b9de");
          }
        }
      }
    }
    if (this.path?.trajectory && this.cfg && !this.recording) {
      const segments = trajectorySegments(this.path, this.cfg.arm);
      for (const segment of segments) {
        const vertices = segment.points.map(xyz => transform(virtualProject(this.cfg.camera, xyz)));
        outline(ctx, vertices, "#a6ffdfbb", 1.6, [5, 5]);
      }
      const sweep = segments.find(s => s.phase === "sweep");
      if (sweep) {
        ctx.beginPath();
        ctx.arc(...transform(virtualProject(this.cfg.camera, sweep.points[0])), 4, 0, Math.PI * 2);
        ctx.fillStyle = "#9afbe2";
        ctx.fill();
      }
    }
    ctx.restore();
  }
}

function surfaceTexture(field) {
  if (!field) return null;
  const {width, height, data} = field;
  const raster = document.createElement("canvas");
  raster.width = width;
  raster.height = height;
  const drawing = raster.getContext("2d");
  const image = drawing.createImageData(width, height);
  const values = atob(data);
  for (let i = 0; i < values.length; i++) {
    const sample = values.charCodeAt(i);
    if (sample === 254) continue;
    let color;
    if (sample === 255) color = i % 5 === 0 ? [83, 91, 106] : [67, 74, 88];
    else if (sample === 0) color = [255, 174 + 30 * Math.sin(i / 23) ** 2, 75];
    else {
      const d = sample / 250;
      color = [218 - 140 * d, 144 - 106 * d, 175 - 88 * d];
    }
    image.data.set([...color, 255], i * 4);
  }
  drawing.putImageData(image, 0, 0);
  return raster;
}

export class MachineView {
  constructor(canvas) {
    this.canvas = canvas;
    this.reset(false);
    this.drag = null;
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    canvas.addEventListener("pointerdown", event => {
      canvas.setPointerCapture(event.pointerId);
      this.drag = [event.clientX, event.clientY];
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", event => {
      if (!this.drag) return;
      this.yaw -= (event.clientX - this.drag[0]) * .009;
      this.elevation = clamp(this.elevation + (event.clientY - this.drag[1]) * .008, .14, 1.55);
      this.drag = [event.clientX, event.clientY];
      this.draw();
    });
    const release = () => {
      this.drag = null;
      canvas.classList.remove("dragging");
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      this.zoom = clamp(this.zoom * Math.exp(event.deltaY * .0011), .66, 2.0);
      this.draw();
    }, {passive: false});
  }
  dispose() {
    this.observer.disconnect();
  }
  reset(redraw = true) {
    this.yaw = .72;
    this.elevation = .54;
    this.zoom = 1.13;
    if (redraw) this.draw();
  }
  top() {
    this.yaw = 0;
    this.elevation = 1.54;
    this.draw();
  }
  set(cfg, state, obs, path, field = obs?.surface, recording = false) {
    this.cfg = cfg;
    this.state = state;
    this.obs = obs;
    this.path = path;
    if (field?.data !== this.field?.data) this.texture = surfaceTexture(field);
    this.field = field;
    this.recording = recording;
    this.draw();
  }
  draw() {
    const dims = dimensions(this.canvas);
    if (!dims) return;
    const {ctx, w, h} = dims;
    const backdrop = ctx.createRadialGradient(w * .52, h * .54, 5, w * .52, h * .54, w * .62);
    backdrop.addColorStop(0, "#19303d");
    backdrop.addColorStop(1, "#0c151e");
    ctx.fillStyle = backdrop;
    ctx.fillRect(0, 0, w, h);
    if (!this.cfg || !this.state) return;
    const {tank, arm} = this.cfg, state = this.state;
    const size = Math.max(tank.length_m / 5.25, tank.width_m / 4.5, arm.boom_z_m / 5.55, 0.75);
    const target = [tank.pivot_m[0], tank.pivot_m[1] - .62 * size, tank.pivot_m[2] + .03 * size];
    const a = this.yaw, e = this.elevation;
    const eye = add(target, scale([Math.sin(a) * Math.cos(e), -Math.cos(a) * Math.cos(e), Math.sin(e)], 12.2 * size * this.zoom));
    const forward = unit(sub(target, eye)), right = [Math.cos(a), Math.sin(a), 0];
    const up = cross(right, forward);
    const focal = Math.min(w, h * 1.60) * 1.28;
    const project = point => {
      const v = sub(point, eye), depth = Math.max(.2, dot(forward, v));
      return [w * .5 + dot(right, v) * focal / depth, h * .49 - dot(up, v) * focal / depth, depth];
    };
    const shape = [], transform = localTransform(tank, state.tilt_deg);
    const face = (points, color, edge = null, order = 0, tile = null) => {
      const vertices = points.map(project);
      const depth = vertices.reduce((sum, v) => sum + v[2], 0) / vertices.length;
      shape.push({vertices, depth: depth + order, color, edge, tile});
    };
    const drawFace = ({vertices, color, edge, tile}) => {
      if (tile) {
        const [p, right, , down] = vertices;
        const [sx, sy, tw, th] = tile;
        ctx.save();
        ctx.transform((right[0] - p[0]) / tw, (right[1] - p[1]) / tw,
          (down[0] - p[0]) / th, (down[1] - p[1]) / th, p[0], p[1]);
        ctx.drawImage(this.texture, sx, sy, tw, th, -.04, -.04, tw + .10, th + .10);
        ctx.restore();
        return;
      }
      trace(ctx, vertices);
      ctx.fillStyle = color;
      ctx.fill();
      if (edge) {
        ctx.strokeStyle = edge;
        ctx.lineWidth = .75;
        ctx.stroke();
      }
    };
    const gridSize = 8 * size;
    for (let i = -8; i <= 8; i++) {
      const t = i * size;
      const color = i === 0 ? "#799bad39" : "#6d91a226";
      outline(ctx, [[-gridSize, t, 0], [gridSize, t, 0]].map(project), color);
      outline(ctx, [[t, -gridSize, 0], [t, gridSize, 0]].map(project), color);
    }
    const ringSize = 72;
    const ring = (radii, z, notch = false) => Array.from({length:ringSize}, (_, i) => {
      const t = i / ringSize * Math.PI * 2, x = radii[0] * Math.cos(t), y = radii[1] * Math.sin(t);
      const cut = notch && y < 0 && Math.abs(x - tank.outlet_x_m) < tank.outlet_width_m / 2;
      return transform([x, y, z - (cut ? tank.outlet_drop_m || 0 : 0)]);
    });
    const skin = (low, high, palette, edge) => {
      for (let i = 0; i < low.length; i++) {
        const j = (i + 1) % low.length;
        const light = clamp(Math.round((Math.sin(i * Math.PI * 2 / low.length - .65) + 1) * (palette.length - 1) / 2), 0, palette.length - 1);
        face([low[i], low[j], high[j], high[i]], palette[light], edge);
      }
    };
    const external = [tank.length_m / 2, tank.width_m / 2];
    const bottomZ = tank.rim_z_m - tank.height_m;
    const outerBottom = ring(scale(external, tank.bottom_scale), bottomZ);
    const belt = ring(scale(external, .90), bottomZ + tank.height_m * .43);
    const shoulder = ring(scale(external, .985), tank.rim_z_m - .28);
    const top = ring(external, tank.rim_z_m, true);
    const inner = ring(tank.rim_radii_m, tank.rim_z_m, true);
    const bath = state.bath;
    const bathPoint = (u, v, dz = 0) => bath ?
      [bath.center_xy[0] + u * bath.radii_m[0], bath.center_xy[1] + v * bath.radii_m[1], bath.z_m + dz] :
      transform([u * tank.inner_radii_m[0], v * tank.inner_radii_m[1], tank.liquid_z_m + dz]);
    const liquid = Array.from({length:ringSize}, (_, i) => bathPoint(Math.cos(i / ringSize * 2 * Math.PI), Math.sin(i / ringSize * 2 * Math.PI)));
    skin(outerBottom, belt, ["#2a3a45", "#374d5a", "#4e6977", "#637c85"], "#68838d40");
    skin(belt, shoulder, ["#2e4251", "#45606d", "#55707c", "#6d8690"], "#7e97a331");
    skin(shoulder, top, ["#465d6a", "#5f7984", "#77919a", "#89a1aa"]);
    skin(top, inner, ["#8b9fa4", "#9eafb3", "#b4c1c2", "#c6cecc"], "#c6d3d64e");
    skin(inner, liquid, ["#505d69", "#61707e", "#74838e", "#8d9ba2"]);
    // Triangulate the liquid surface so that the front wall can occlude it correctly.
    const liquidCenter = bathPoint(0, 0);
    const glow = ctx.createRadialGradient(...project(liquidCenter).slice(0, 2), 1, ...project(liquidCenter).slice(0, 2), w * .29);
    glow.addColorStop(0, "#ffcb76");
    glow.addColorStop(.50, "#f99546");
    glow.addColorStop(1, "#b05226");
    const baseColor = this.recording || this.texture ? "#4d5867" : glow;
    if (!this.texture) {
      for (let i = 0; i < liquid.length; i++) face([liquidCenter, liquid[i], liquid[(i + 1) % liquid.length]], baseColor, baseColor);
    }
    if (this.texture) {
      const tw = this.texture.width, th = this.texture.height;
      const pixelPoint = (x, y) => bathPoint(x * 2 / tw - 1, 1 - y * 2 / th, .018);
      for (let y = 0; y < th; y += 4) for (let x = 0; x < tw; x += 7) {
        const x1 = Math.min(x + 7, tw), y1 = Math.min(y + 4, th);
        face([pixelPoint(x, y), pixelPoint(x1, y), pixelPoint(x1, y1), pixelPoint(x, y1)],
          null, null, -.055, [x, y, x1 - x, y1 - y]);
      }
    }
    for (const patch of this.texture ? [] : this.obs?.slag || []) {
      if (!patch.plane_contour || patch.plane_contour.length < 3) continue;
      const vertices = patch.plane_contour.map(([x, y]) => transform([x, y, tank.liquid_z_m + .025]));
      const center = transform([...patch.plane_xy, tank.liquid_z_m + .025]);
      for (let i = 0; i < vertices.length; i++) {
        face([center, vertices[i], vertices[(i + 1) % vertices.length]], "#ad638e", null, -.045);
      }
    }
    const lip = tank.outlet_x_m, width = tank.outlet_width_m / 2, outY = -tank.rim_radii_m[1];
    const sillZ = tank.rim_z_m - (tank.outlet_drop_m || 0);
    const chuteY = -tank.width_m / 2 - tank.outlet_extension_m;
    face([[lip - width, outY + .045, sillZ + .025], [lip + width, outY + .045, sillZ + .025],
      [lip + width * .89, chuteY, sillZ - .11], [lip - width * .89, chuteY, sillZ - .11]].map(transform), "#768a99", "#c6d4df");

    const cube = (center, axes, dims, palette, edge) => {
      const offsets = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
      const vertices = offsets.map(v => axes.reduce((xyz, axis, i) => add(xyz, scale(axis, v[i] * dims[i] / 2)), center));
      const surfaces = [[0,1,2,3],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]];
      surfaces.forEach((group, i) => face(group.map(k => vertices[k]), palette[i % palette.length], edge));
    };
    const globalAxes = [[1,0,0],[0,1,0],[0,0,1]];
    const steel = ["#263f54", "#36536a", "#4e7189", "#284658", "#496b83", "#6c8da0"];
    const yellow = ["#8f6234", "#a98049", "#d5a867", "#ab8248", "#be9857", "#eac38a"];
    const cyan = ["#236a8b", "#248eb6", "#4fb5d9", "#286a94", "#388db0", "#7bd9ed"];
    const [bx, by] = arm.base_m;
    cube([bx, by, .14], globalAxes, [1.07, 1.12, .28], steel, "#678399");
    cube([bx, by, arm.boom_z_m / 2], globalAxes, [.53, .56, arm.boom_z_m], steel, "#7e99ab79");
    for (const sign of [-1, 1]) {
      const side = tank.pivot_m[0] + sign * tank.trunnion_span_m / 2;
      cube([side, tank.pivot_m[1], tank.pivot_m[2] * .48], globalAxes, [.50, .68, tank.pivot_m[2] * .96], steel);
      cube([side, tank.pivot_m[1], tank.pivot_m[2]], globalAxes, [.82, .66, .58], yellow, "#bba17e");
    }
    const head = state.head_m;
    if (head) {
      const yaw = state.joints.swing_deg * RAD, pitch = state.joints.pitch_deg * RAD;
      const side = [Math.cos(yaw), -Math.sin(yaw), 0];
      const along = [Math.sin(yaw), Math.cos(yaw), 0];
      const pitched = [along[0] * Math.cos(pitch), along[1] * Math.cos(pitch), Math.sin(pitch)];
      const boomTop = arm.boom_z_m, stroke = state.joints.extension_m;
      cube([bx + along[0] * stroke / 2, by + along[1] * stroke / 2, boomTop],
        [side, along, [0, 0, 1]], [.29, stroke, .24], cyan, "#91cce395");
      cube([head[0], head[1], (boomTop + head[2]) / 2], globalAxes, [.155, .16, boomTop - head[2]], cyan, "#8bd8f0");
      cube(head, [side, pitched, cross(side, pitched)], arm.tool_size_m, cyan, "#a8e7ff");
    }
    const binX = tank.pivot_m[0] + lip, binY = tank.pivot_m[1] + chuteY - .25;
    cube([binX, binY, .36], globalAxes, [tank.outlet_width_m * 1.17, .98, .70], steel, "#8396a6");
    if (state.discharged_load > 0) {
      cube([binX, binY, .73], globalAxes, [tank.outlet_width_m * .91, .71, .06],
        ["#803b68", "#9b5081", "#b06d9c"], "#d395be");
    }
    for (const zone of arm.safety_boxes) {
      const center = scale(add(zone.min_m, zone.max_m), .5);
      cube(center, globalAxes, sub(zone.max_m, zone.min_m), ["#8b5c3f35", "#bb76462b", "#cd83443b"], "#ca925e65");
    }
    shape.sort((a, b) => b.depth - a.depth).forEach(drawFace);

    if (this.path?.trajectory) {
      const segments = trajectorySegments(this.path, arm);
      for (const [i, segment] of segments.entries()) {
        const active = ["sweep", "discharge"].includes(segment.phase);
        const points = segment.points.map(project);
        if (active) {
          outline(ctx, points, "#46f3b64c", 6);
          outline(ctx, points, segment.phase === "discharge" ? "#b4ffd7" : "#78f5a6", 2.5);
        } else {
          outline(ctx, points, "#69d5ddaf", 1.4, [5, 5]);
        }
        if (i < segments.length - 1) {
          ctx.beginPath();
          ctx.arc(points.at(-1)[0], points.at(-1)[1], 3.5, 0, Math.PI * 2);
          ctx.fillStyle = active ? "#bdffed" : "#6bbdcb";
          ctx.fill();
        }
      }
    }
    const lipLabel = project(transform([lip, chuteY, sillZ + .12]));
    ctx.font = "11px 'Microsoft YaHei UI', sans-serif";
    ctx.fillStyle = "#14232fe9";
    ctx.fillRect(lipLabel[0] + 4, lipLabel[1] - 14, 46, 16);
    ctx.fillStyle = "#dce9f3";
    ctx.fillText("出渣口", lipLabel[0] + 8, lipLabel[1] - 2);
    const origin = [w - 46, h - 44], axisRoot = project([0, 0, 0]);
    const axes = [[[1,0,0], "X", "#f0a27b"], [[0,1,0], "Y", "#69d8ba"], [[0,0,1], "Z", "#74b6f6"]];
    ctx.font = "10px 'Consolas', monospace";
    for (const [direction, label, color] of axes) {
      const end = project(direction);
      const delta = [end[0] - axisRoot[0], end[1] - axisRoot[1]];
      const length = Math.hypot(...delta);
      const p = add(origin, scale(delta, Math.min(1.4, 21 / (length || 1))));
      outline(ctx, [origin, p], color, 1.6);
      ctx.fillStyle = color;
      ctx.fillText(label, p[0] + 2, p[1] - 1);
    }
  }
}

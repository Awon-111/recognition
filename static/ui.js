import {PHASE} from "./scene.js";

export const NAV = [
  ["live", "实时作业", "monitor"], ["config", "设备配置", "sliders"], ["data", "数据与训练", "database"],
  ["experiment", "仿真实验", "flask"], ["replay", "过程回放", "history"], ["report", "结果报告", "report"],
];

const PATHS = {
  monitor: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M7 21h10M12 17v4m-6-10h3l2-3 2 5 2-2h3"/>',
  sliders: '<path d="M4 7h6m4 0h6M4 17h10m4 0h2"/><circle cx="12" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
  database: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  flask: '<path d="M9 3h6m-5 0v7L5 18a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3L14 10V3M8 15h8"/>',
  history: '<path d="M4 10a8 8 0 1 1 2 8M4 4v6h6m2-3v5l3 2"/>',
  report: '<path d="M6 3h9l4 4v14H5V4a1 1 0 0 1 1-1Zm8 0v5h5M8 17h7M8 13h5"/>',
  time: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3 2"/>',
  target: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4"/><path d="m12 12 7-7"/>',
  motion: '<path d="M5 17 10 8l4 6 5-8"/><circle cx="5" cy="17" r="2"/><circle cx="19" cy="6" r="2"/>',
  pulse: '<path d="M3 12h4l2.5-6 4 12 2.5-6h5"/>',
  camera: '<rect x="3" y="5" width="14" height="14" rx="2"/><path d="m17 10 4-3v10l-4-3"/>',
  cube: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9Zm0 9 8-4.5M12 12 4 7.5M12 12v9"/>',
  play: '<path d="m8 5 12 7-12 7Z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M9 5v14m6-14v14" stroke-width="3"/>',
  step: '<path d="m6 5 10 7-10 7Z" fill="currentColor" stroke="none"/><path d="M19 5v14"/>',
  reset: '<path d="M5 10a7.5 7.5 0 1 1 1.9 7M5 5v5h5"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M5 16v4h14v-4"/>',
  upload: '<path d="M12 16V4m-4 4 4-4 4 4M5 16v4h14v-4"/>',
  eye: '<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/>',
  check: '<path d="m5 12 4 4 10-10"/>',
  alert: '<path d="M12 3 2.6 20h18.8ZM12 9v5m0 3v.2"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  save: '<path d="M5 3h12l4 4v13H3V5a2 2 0 0 1 2-2Zm2 0v6h9V3M7 20v-7h10v7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
};

export const icon = key => `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[key] || ""}</svg>`;
const entities = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"};
export const esc = value => String(value ?? "").replace(/[&<>"']/g, x => entities[x]);
export const num = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : "—";
export const pct = (value, digits = 1) => Number.isFinite(value) ? (value * 100).toFixed(digits) + "%" : "—";
export const seconds = value => num(value, 2) + " s";
export const statusName = {ready: "就绪", hold: "安全保持", passed: "仿真达标", failed: "仿真未达标"};
export const timingNames = {recognition: "识别", planning: "规划", motion: "设备运动", feedback: "等待反馈", holding: "安全保持"};

export function btn(action, label, key, style = "", attrs = "") {
  return `<button type="button" class="btn ${style}" data-action="${action}" ${attrs}>${key ? icon(key) : ""}${esc(label)}</button>`;
}
export const pill = (label, tone = "ghost") => `<span class="pill ${tone}">${esc(label)}</span>`;
export const option = (value, label, selected) => `<option value="${esc(value)}" ${value === selected ? "selected" : ""}>${esc(label)}</option>`;
export const blank = value => `<div class="blank">${esc(value)}</div>`;
export const download = (url, name, label = "导出 JSON") => `<a class="btn" href="${esc(url)}" download="${esc(name)}">${icon("download")}${esc(label)}</a>`;

export function heading(kicker, title, subtitle, actions = "") {
  return `<div class="page-heading"><div><p class="kicker">${kicker}</p><h1>${title}</h1><p class="subtitle">${subtitle}</p></div><div class="actions">${actions}</div></div>`;
}

export function card(title, iconName, body, actions = "", foot = "", cls = "") {
  return `<section class="card ${cls}"><header class="card-head"><div class="titleline">${icon(iconName)}<h2>${title}</h2></div>${actions}</header>${body}${foot ? `<footer class="card-foot">${foot}</footer>` : ""}</section>`;
}

export function metric(label, value, unit, foot, iconName, style = "") {
  return `<div class="metric"><div class="metric-head"><span>${label}</span>${icon(iconName)}</div><div class="metric-value ${style}">${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}</div><span class="metric-foot" title="${esc(foot)}">${esc(foot)}</span></div>`;
}

export function table(headers, rows, scroll = false) {
  return `<div class="table-wrap ${scroll ? "scroll-table" : ""}"><table><thead><tr>${headers.map(h => "<th>" + h + "</th>").join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

export function pairViews({replay = false, mode = "sim"} = {}) {
  const modeControl = replay ? pill("归档同步", "blue") : `<div class="seg" aria-label="相机输入"><button data-action="mode-sim" class="${mode === "sim" ? "active" : ""}">仿真相机</button><button data-action="mode-video" class="${mode === "video" ? "active" : ""}">录像输入</button></div>`;
  const camera = card("相机识别", "camera", `<div class="viewport"><canvas id="camera" aria-label="相机画面和识别覆盖层"></canvas><div class="float-layer left"><span class="stream-tag" id="stream-tag"><i></i>CAM–01</span></div><div class="float-layer left-bottom"><span class="stream-tag" id="camera-stamp">FRAME —</span></div></div><div id="video-controls"></div>`, modeControl,
    `<div class="legend"><span><em class="slag"></em>残渣</span><span><em class="iron"></em>铁水</span><span><em class="tool"></em>机械臂</span><span><em class="rim"></em>罐体</span><span><em class="steam"></em>遮挡／不确定</span></div><label class="check"><input type="checkbox" id="overlay" checked>识别覆盖层</label>`);
  const machine = card(mode === "video" ? "同步三维与扒渣路径" : "三维设备", "cube", `<div class="viewport"><canvas class="orbit" id="machine" aria-label="三维设备场景，可拖动旋转和滚轮缩放"></canvas><div class="float-layer left"><span class="stream-tag" id="scene-tag">360 t · 仿真假设</span></div><div class="float-layer right-top">${btn("view-top", "俯视", null, "small")}${btn("view-reset", "复位", "reset", "small")}</div><div class="float-layer left-bottom"><span class="stream-tag film-key"><em></em>${mode === "video" ? "紫色 · 渣层　绿色 · 回拉与排渣" : "渣层浓度 · 低 → 高"}</span></div></div>`,
    `<span class="tiny muted">拖动旋转 · 滚轮缩放</span>`, `<div id="axis-values" class="axis-values"></div><span id="scene-stamp" class="mono tiny">—</span>`);
  return `<div class="panels">${camera}${machine}</div>`;
}

export function axisValues(state) {
  if (!state) return "姿态未同步";
  const q = state.joints;
  return `<span>伸缩<b>${num(q.extension_m)} m</b></span><span>升降<b>${num(q.lift_m)} m</b></span><span>回转<b>${num(q.swing_deg, 1)}°</b></span><span>俯仰<b>${num(q.pitch_deg, 1)}°</b></span><span>倾角<b>${num(state.tilt_deg, 1)}°</b></span>`;
}

export function phases(active) {
  const steps = ["tilt", "approach", "skim", "sweep", "discharge", "retreat"];
  return steps.map((key, i) => `<span class="phase ${active === key ? "active" : ""}">${i + 1} · ${PHASE[key]}</span>`).join('<span class="arrow">›</span>');
}

export function candidatesTable(candidates, selected, filter, preview, action = "candidate") {
  const rows = candidates.filter(x => filter === "all" || x.valid === (filter === "valid"));
  if (!rows.length) return blank("当前帧暂无符合筛选的候选动作");
  return table(["候选动作", "评分", "预计带出", "耗时", "约束结果"], rows.map(c => `
    <tr class="${preview === c.id || selected === c.id ? "selected" : ""}"><td><button class="btn small flat" data-action="${action}" data-id="${esc(c.id)}">${esc(c.target)} · ${esc(c.label)}</button></td>
      <td class="mono">${num(c.score, 3)}</td><td class="mono">${num(c.metrics.expected_load ?? c.metrics.expected_m2, 3)}${c.metrics.expected_load !== undefined ? " 指数" : c.metrics.expected_m2 !== undefined ? " m²" : ""}</td>
      <td class="mono nowrap">${seconds(c.duration_s)}</td><td title="${esc(c.reasons.join(" · "))}">${c.id === selected ? pill("已选择", "teal") : c.valid ? pill("可行", "blue") : `<span class="value-amber">${esc(c.reasons[0])}</span>`}</td></tr>`), true);
}

export function tracksTable(obs) {
  if (!obs?.slag?.length) return blank(obs?.reason || "当前画面暂无可见残渣");
  return table(["编号", "面积", "视觉类别", "漂移 XY", "可信度"], obs.slag.map(p => {
    const velocity = p.plane_xy ? p.velocity : p.velocity_px_s;
    return `<tr><td class="mono"><strong>${esc(p.id)}</strong></td>
      <td class="mono nowrap" title="${esc(p.pixel_area)} px²">${p.area_m2 == null ? p.pixel_area + " px²" : num(p.area_m2, 3) + " m²"}</td>
      <td>${esc(p.visual_class)}</td><td class="mono nowrap" title="${p.plane_xy ? "m/s" : "预览像素/s"}">${num(velocity[0], 2)}, ${num(velocity[1], 2)}</td>
      <td class="mono">${pct(p.confidence, 0)}</td></tr>`;
  }), true);
}

export function candidateDetail(row) {
  const m = row.metrics;
  return `<p class="secondary">${esc(row.explanation || row.reasons.join(" · "))}</p>
    <div class="kv"><div><small>${m.expected_load === undefined ? "预测去除" : "预计新增裸露铁水"}</small><strong class="mono">${num(m.expected_m2, 4)} m²</strong></div>
      ${m.expected_load === undefined ? "" : `<div><small>预计带出渣量指数</small><strong class="mono">${num(m.expected_load, 4)}</strong></div>`}
      <div><small>可信度</small><strong class="mono">${pct(row.confidence)}</strong></div>
      <div><small>铁损风险指数</small><strong class="mono">${num(m.iron_risk, 3)}</strong></div>
      <div><small>液面扰动 / 后续漂移风险</small><strong class="mono">${num(m.disturbance, 3)} / ${num(m.future_risk, 3)}</strong></div></div>
    <p class="tiny muted">各轴按相邻节点平滑插值。速度列为到达该节点的区段峰值。</p>
    ${table(["相位", "时刻 s", "伸缩 m", "升降 m", "回转 °", "俯仰 °", "峰值速度"], row.trajectory.map(k => `<tr><td>${PHASE[k.phase]}</td><td class="mono">${num(k.t_s)}</td>
      <td class="mono">${num(k.joints.extension_m)}</td><td class="mono">${num(k.joints.lift_m)}</td><td class="mono">${num(k.joints.swing_deg)}</td><td class="mono">${num(k.joints.pitch_deg)}</td>
      <td class="mono nowrap">${Object.values(k.segment_peak_speed).map(v => num(v, 1)).join(" / ")}</td></tr>`), true)}`;
}

export function reportBody(report) {
  if (!report) return blank("请选择已归档的仿真作业");
  const r = report, t = r.thresholds, heat = (r.remaining_density_heatmap || r.remaining_heatmap).flat();
  const max = Math.max(1, ...heat);
  const time = Object.entries(r.timings).map(([key, v]) => `<div class="timing"><span>${timingNames[key]}</span><div class="track"><span style="width:${Math.min(100, v / Math.max(.001, r.elapsed_s) * 100)}%"></span></div><span class="mono">${seconds(v)}</span></div>`).join("");
  const map = `<div class="heatmap" role="img" aria-label="剩余残渣分布，亮色表示该网格残渣更多">${heat.map(v => `<span style="background:${v ? `rgba(211,117,176,${.2 + v / max * .80})` : "#1b2a35"}" title="${v}"></span>`).join("")}</div>`;
  const concentration = r.remaining_mean_density === undefined ? "" : `<div class="kv"><div><small>覆盖率</small><strong class="mono">${pct(r.initial_coverage,2)} → ${pct(r.remaining_coverage,2)}</strong></div>
    <div><small>平均浓度指数</small><strong class="mono">${num(r.initial_mean_density,3)} → ${num(r.remaining_mean_density,3)}</strong></div></div>
    <p class="muted tiny">累计渣量指数下降 ${pct(r.load_removal_rate,2)}${r.first_stroke ? ` · 首轮排渣后仍覆盖 ${pct(r.first_stroke.coverage_after,2)}，浓度降至 ${num(r.first_stroke.density_after,3)}` : ""}</p>`;
  const passed = r.ended ? pill(r.passed ? "配置目标达标" : "配置目标未达标", r.passed ? "teal" : "amber") : pill("作业进行中", "blue");
  return `<div class="metrics">
    ${metric("总作业时间", num(r.elapsed_s, 3), "s", "配置上限 " + t.time_s + " s", "time", r.checks.time ? "value-teal" : "value-amber")}
    ${metric("真值面积去除率", pct(r.removal_rate, 2), "", "配置目标 > " + pct(t.removal_rate, 0), "target", r.checks.area ? "value-teal" : "value-amber")}
    ${metric("执行动作", r.actions, "段", `扒渣 ${r.strokes} 段 · 保持 ${r.holds} 次`, "motion")}
    ${metric("模拟铁损", num(r.iron_risk, 3), "风险指数", `风险上限 ${t.iron_risk} · 扰动 ${num(r.disturbance, 3)}`, "pulse")}
    </div><div class="cols">
      ${card("作业时间构成", "time", `<div class="card-body">${time}<p class="muted tiny">实测计算耗时 · 识别 ${seconds(r.processing_wall_s.recognition)} · 规划 ${seconds(r.processing_wall_s.planning)}</p></div>`, passed)}
      ${card("残余分布", "target", `<div class="card-body">${map}<div class="kv"><div><small>初始覆盖面积</small><strong class="mono">${num(r.initial_m2, 4)} m²</strong></div><div><small>残余覆盖面积</small><strong class="mono">${num(r.remaining_m2, 4)} m²</strong></div></div>${concentration}</div>`, pill(r.remaining_density_heatmap ? "浓度网格" : "罐体平面"))}
    </div>
    <div class="notice ${r.ended && !r.passed ? "amber" : ""}">${esc(r.stop_reason || "当前运行的阶段结果")}<br>${r.ended ? (r.reference_passed ? "固定验收达标 · " : "固定验收未达标 · ") : ""}配置 ${esc(r.config_revision)} · 种子 ${r.seed} · ${esc(r.field_status)} · ${Object.entries(r.sources).map(([k, v]) => esc(({tank:"罐体",arm:"扒渣机",tilt:"倾翻",camera:"相机",slag:"残渣",objectives:"目标"})[k] + " " + v)).join(" · ")}</div>`;
}

export function batchTable(report, strategies) {
  if (!report?.results?.length) return blank("尚无批量评测记录");
  const rows = report.results;
  return table(["评测工况", "策略", "种子", "用时 s", "面积去除", "渣量指数下降", "动作", "120 s / 98%", "详情"], rows.map((r, i) => `<tr>
      <td><strong>${esc(r.case.name)}</strong></td><td>${esc(strategies[r.strategy])}</td><td class="mono">${r.seed}</td>
      <td class="mono">${num(r.elapsed_s, 3)}</td><td class="mono">${pct(r.removal_rate, 2)}</td><td class="mono">${pct(r.load_removal_rate, 2)}</td><td class="mono">${r.actions}</td>
      <td>${pill(r.reference_passed ? "达标" : "未达标", r.reference_passed ? "teal" : "amber")}</td>
      <td>${btn("batch-detail", "查看", null, "small flat", `data-index="${i}"`)}</td></tr>`));
}

export function saveBlob(data, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2) + "\n"], {type: "application/json"}));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function request(url, options = {}) {
  const res = await fetch(url, {cache:"no-store", ...options});
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail) ? detail.map(e => e.loc.slice(1).join(".") + " · " + e.msg).join(" / ") : detail;
    throw new Error(message || "本地服务返回 " + res.status);
  }
  return data;
}

export const post = (url, body) => request(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});

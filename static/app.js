import {CameraView, MachineView, PHASE, initialPose, recordedPose, trajectoryPose} from "./scene.js";
import {
  NAV, axisValues, blank, btn, candidateDetail, candidatesTable, card, esc, heading, icon, metric,
  num, option, pairViews, pct, phases, pill, post, reportBody, request, saveBlob, seconds, statusName, tracksTable,
} from "./ui.js";
import {
  calibrationForm, configPage, dataPage, drawConfigAxis, experimentPage, replayPage, reportPage, trainForm,
} from "./pages.js";

const $ = (selector, root = document) => root.querySelector(selector);
const page = $("#page");
const dialog = $("#dialog");
const faults = {offline:"视频中断", stale:"帧时间戳过期", steam:"严重蒸汽", low:"低可信度"};
const videoBlank = {id:"", playing:false, time:0, clip:null, sample:null, meta:null, plan:null, segments:new Map(),
  preview:null, message:"等待录像输入", token:0, controller:null, lastFrameAt:0};
const ctx = {
  route:"live", boot:null, cfg:null, revision:null, revisions:[], run:null, runConfig:null,
  mode:"sim", mask:true, filter:"all", previewId:null, scene:null, camera:null, shown:null,
  live:{auto:false, busy:false, starting:false, speed:4, animation:null, pauseRequested:false, timer:null},
  video:{...videoBlank}, data:null, trainJob:null, batch:null, batchJob:null, archives:[], singleReport:null,
  draft:null, configGroup:"tank", exp:{caseId:"commissioning", strategy:"balanced", seed:1734, batchStrategies:["balanced"], weights:null},
  replay:{trace:null, time:0, speed:2, playing:false, frame:null, event:null, frames:[], candidate:null},
};

function toast(text, error = false) {
  const item = document.createElement("div");
  item.className = "toast" + (error ? " error" : "");
  item.textContent = text;
  $("#toasts").append(item);
  setTimeout(() => item.remove(), error ? 6400 : 3500);
}

function modal(title, body) {
  dialog.innerHTML = `<div class="dialog-head"><h2 id="dialog-title">${esc(title)}</h2><button class="icon-btn" data-action="close-dialog" aria-label="关闭">${icon("close")}</button></div><div class="dialog-body">${body}</div>`;
  dialog.setAttribute("aria-labelledby", "dialog-title");
  if (!dialog.open) dialog.showModal();
}

const isTerminal = run => run && ["passed", "failed"].includes(run.status);
const videoInfo = () => ctx.data?.videos.find(v => v.id === ctx.video.id);
const currentDecision = () => ctx.mode === "video" ? ctx.video.plan : ctx.run;
const selectedCandidate = () => {
  const decision = currentDecision();
  return decision?.candidates.find(c => c.id === (ctx.previewId || decision.selected_id));
};
const videoPath = () => ctx.video.plan?.timely && selectedCandidate()?.valid ? selectedCandidate() : null;

function drawNavigation() {
  $("#nav").innerHTML = NAV.map(([key, title, ico], i) => `<a class="nav-link ${key === ctx.route ? "active" : ""}" href="#${key}" ${key === ctx.route ? 'aria-current="page"' : ""}>${icon(ico)}${title}<small>0${i + 1}</small></a>`).join("");
  $("#crumb").textContent = NAV.find(([key]) => key === ctx.route)[1];
}

function releaseViews() {
  ctx.camera?.dispose();
  ctx.scene?.dispose();
  ctx.camera = ctx.scene = ctx.shown = null;
}

function attachViews() {
  const camera = $("#camera"), machine = $("#machine"), overlay = $("#overlay");
  if (camera) ctx.camera = new CameraView(camera);
  if (machine) ctx.scene = new MachineView(machine);
  if (overlay) overlay.checked = ctx.mask;
}

function livePage() {
  const simDesc = ctx.run ? `${esc(ctx.run.case.name)} · ${esc(ctx.boot.strategies[ctx.run.strategy])} · 配置 <span class="mono">${esc(ctx.run.config_revision)}</span>` : "同步检查相机识别、设备姿态和下一段动作。";
  const desc = ctx.mode === "video" ? "从原始录像片段生成扒渣路径，在右侧三维查看或演示。" : simDesc;
  const headingActions = '<span class="actions" id="live-actions"></span>';
  const controls = card("作业控制", "motion", '<div class="flowline" id="flowline"></div><div class="card-body" id="live-control"></div>', '<span class="pill ghost" id="phase-state">等待</span>', "", "controls-bar");
  const cand = card("候选动作", "motion", '<div id="candidate-summary"></div><div id="candidate-table"></div>',
    `<div class="seg" id="candidate-filter"><button data-action="candidate-filter" data-id="all">全部</button><button data-action="candidate-filter" data-id="valid">可行</button><button data-action="candidate-filter" data-id="invalid">排除</button></div>`,
    `<span>点击候选可预览轨迹</span><span id="candidate-count">—</span>`);
  const obs = card("残渣跟踪", "target", '<div id="track-table"></div>', '<span id="tracking-state"></span>', "视觉类别未经厚度实测校准");
  return heading("OPERATIONS / 01", "实时作业", desc, headingActions) +
    '<div class="metrics" id="live-metrics"></div>' + pairViews({mode:ctx.mode}) + controls + `<div class="cols wide-left">${cand}${obs}</div>`;
}

function render() {
  if (!ctx.boot) return;
  drawNavigation();
  releaseViews();
  if (ctx.route === "live") page.innerHTML = livePage();
  if (ctx.route === "config") page.innerHTML = configPage(ctx.draft, ctx.configGroup, ctx.revisions, ctx.revision);
  if (ctx.route === "data") page.innerHTML = ctx.data ? dataPage(ctx.data, ctx.trainJob) : '<div class="loading">正在读取数据目录</div>';
  if (ctx.route === "experiment") {
    const data = ctx.exp.weights ? {...ctx, cfg:{...ctx.cfg, objectives:{...ctx.cfg.objectives, weights:ctx.exp.weights}}} : ctx;
    page.innerHTML = experimentPage(data);
  }
  if (ctx.route === "replay") page.innerHTML = replayPage(ctx);
  if (ctx.route === "report") page.innerHTML = reportPage(ctx);
  attachViews();
  if (ctx.route === "live") {
    if (ctx.mode === "video") drawVideoControls();
    updateLive();
  }
  if (ctx.route === "config") updateConfigPreview();
  if (ctx.route === "replay") drawReplay(true);
}

function showSample(sample, cfg, path, record = false) {
  if (!ctx.scene && !ctx.camera) return;
  ctx.shown = {sample, cfg, path, record};
  const camera = ctx.camera;
  const paint = current => {
    if (ctx.camera !== camera) return;
    const obs = current?.observation, reconstruction = current?.reconstruction;
    const pose = record ? recordedPose(cfg, reconstruction) : current?.state || initialPose(cfg);
    ctx.scene?.set(cfg, pose, obs, path, current?.surface || obs?.surface, record);
    const axes = $("#axis-values");
    if (axes) {
      const tool = pose.head_m ? "机械臂位置估计" : obs?.tool?.length ? "机械臂已检出 · 液面位置待定" : "机械臂未检出";
      axes.innerHTML = record ? esc(tool + (reconstruction?.basis === "calibrated" ? " · 平面标定倾角" : " · 倾角示意")) : axisValues(pose);
    }
    const stamp = current?.time_s;
    const frame = $("#camera-stamp"), time = $("#scene-stamp");
    const frameText = obs ? "FRAME " + String(obs.frame).padStart(5, "0") : "相机采样";
    if (frame) frame.textContent = frameText + "  ·  " + seconds(stamp);
    if (time) time.textContent = record && obs ? frameText + " · " + seconds(stamp) : seconds(stamp);
    for (const canvas of [camera?.canvas, ctx.scene?.canvas]) if (canvas) canvas.dataset.frame = obs?.frame ?? "";
    const source = $("#scene-tag");
    if (source) {
      const basis = {calibrated:"平面标定", relative:"相对投影", unavailable:"等待清晰液面"}[reconstruction?.basis];
      source.textContent = record ? "录像同步 · " + (basis || "等待帧") : "360 t · " + cfg.tank.source;
    }
    const tag = $("#stream-tag");
    if (tag) {
      tag.classList.toggle("warn", Boolean(obs && !obs.usable) || (record && !obs));
      tag.innerHTML = "<i></i>" + esc(record ? (ctx.video.id ? "录像 " + ctx.video.id : "录像输入") : obs && !obs.usable ? "虚拟相机 · 保持" : "CAM–01 · 虚拟相机");
    }
  };
  if (camera) camera.set(sample, path, cfg, ctx.mask, record, paint);
  else paint(sample);
}

function updateMetrics() {
  const host = $("#live-metrics");
  if (!host) return;
  const r = ctx.run, a = ctx.live.animation, v = ctx.video, recording = ctx.mode === "video";
  const limit = ctx.runConfig?.objectives.time_limit_s ?? 120;
  const target = pct(ctx.runConfig?.objectives.removal_target ?? .98, 0);
  const name = recording ? (v.plan?.selected_id ? "路径已就绪" : v.playing ? "分段分析" : "录像预览") : a ? (a.paused ? "已暂停" : "执行中") : ctx.live.busy ? "处理反馈" : statusName[r?.status] || "连接中";
  const obs = recording ? v.sample?.observation : r?.current.observation;
  const activeField = a?.frame?.surface || a?.frame?.observation?.surface;
  const coverage = recording ? obs?.surface?.coverage : activeField?.coverage ?? r?.coverage;
  const density = activeField?.mean ?? r?.mean_density;
  const time = recording ? v.time : a?.cursor ?? r?.elapsed_s;
  const cap = recording ? v.message : a ? a.event.action?.label : r?.reason || "等待提交下一段动作";
  host.innerHTML =
    metric(recording ? "录像时间" : "总作业时间", num(time), "s", recording ? "按录像时间戳播放" : "作业时限 " + limit + " s", "time") +
    metric(recording ? "可见表面渣层占比" : "观测渣层覆盖", pct(coverage), "", recording ? "仅统计当前可识别表面" : "验收面积目标 > " + target + " · 以真值核算", "target", "value-teal") +
    metric(recording ? "机械臂识别" : "观测平均浓度", recording ? (obs?.tool?.length ? "已检出" : "未检出") : num(density, 3), "", recording ? `原始 ${num(videoInfo()?.fps,1)} fps${v.meta?.skipped ? " · 跳过迟到帧 " + v.meta.skipped : ""}` : `归一化指数 · 已排渣 ${r?.strokes ?? 0} 轮`, "motion", recording ? "text" : "value-teal") +
    metric("系统状态", name, "", cap, "pulse", "text " + (name.includes("保持") || name.includes("未达标") ? "value-amber" : "value-teal"));
}

function drawLiveActions() {
  const top = $("#live-actions"), content = $("#live-control"), flow = $("#flowline"), state = $("#phase-state");
  if (!top || !content) return;
  const r = ctx.run, busy = ctx.live.busy || ctx.live.starting, a = ctx.live.animation;
  const ended = isTerminal(r);
  const recording = ctx.mode === "video";
  const toggle = ctx.live.auto || (a && !a.paused) ? ["暂停", "pause"] : a?.paused ? ["继续运行","play"] : ["开始运行","play"];
  top.innerHTML = recording ? btn("mode-sim", "返回仿真", "arrow", "primary") :
    btn("run-reset", "重置", "reset", "", busy ? "disabled" : "") +
    btn("mode-video", "录像分段规划", "camera", "") +
    btn("run-toggle", toggle[0], toggle[1], "primary", !r || ended ? "disabled" : "");
  const active = a?.phase;
  flow.innerHTML = recording ? '<span class="phase">截取片段</span><span class="arrow">›</span><span class="phase">图像增强</span><span class="arrow">›</span><span class="phase">YOLO 分割</span><span class="arrow">›</span><span class="phase active">路径规划</span>' : phases(active || (r?.selected_id?.startsWith("TILT") ? "tilt" : undefined));
  state.textContent = recording ? "录像输入" : active ? PHASE[active] : statusName[r?.status] || "等待";
  state.className = "pill " + (!recording && r?.status === "hold" ? "amber" : "teal");
  if (recording) {
    const v = ctx.video, details = v.meta;
    const results = Array.from(v.segments.values()).slice(-8).reverse();
    content.innerHTML = `<div class="notice ${v.plan?.selected_id ? "" : "amber"}">${esc(v.message)}</div>
      ${details ? `<p class="muted tiny space-top">${esc(details.model_name)} · ${esc(details.enhancement)} 增强 · ${details.calibrated ? "液面平面标定" : "相对投影"} · 规划从参考待机姿态开始</p>` : ""}
      ${results.length ? `<div class="actions space-top">${btn("video-export", "导出分段结果", "report", "small")}</div>
        <div class="space-top table-wrap"><table><thead><tr><th>片段</th><th>取帧</th><th>规划截止</th><th>端到端耗时</th><th>结果</th></tr></thead><tbody>
        ${results.map(p => `<tr><td>${p.segment + 1}</td><td>${seconds(p.time_s)}</td><td>${seconds(p.deadline_s)}</td><td>${p.total_ms == null ? "—" : num(p.total_ms, 1) + " ms"}</td><td>${esc(p.selected_id || p.reason)}</td></tr>`).join("")}</tbody></table></div>` : ""}`;
    return;
  }
  const ready = r && !busy && !ended && !a && !ctx.live.auto;
  const progress = Math.min(100, (a?.cursor ?? r?.elapsed_s ?? 0) / (ctx.runConfig?.objectives.time_limit_s ?? 120) * 100);
  const faultOptions = Object.entries(faults).map(([key, label]) => option(key, label, r?.fault)).join("");
  content.innerHTML = `<div class="scrubber">
    ${btn("run-step", "执行一段", "step", "", ready ? "" : "disabled")}
    <label class="check">显示速度 <select id="run-speed" aria-label="仿真显示速度">${[1,2,4,8].map(n => option(n, n + "×", ctx.live.speed)).join("")}</select></label>
    <div class="actions right">
      ${btn("disturb", "扰动残渣", "pulse", "flat", ready ? "" : "disabled")}
      <select id="fault-select" aria-label="相机故障注入" ${ready ? "" : "disabled"}><option value="">正常相机</option>${faultOptions}</select>
      ${ended ? btn("open-live-report", "查看报告", "report", "primary small") : ""}
    </div></div>
    <div class="track space-top" aria-label="仿真计时进度"><span id="live-progress" style="width:${progress}%"></span></div>
    ${r?.reason ? `<p class="tiny ${r.status === "passed" ? "value-teal" : "value-amber"} space-top">${esc(r.reason)}</p>` : ""}
    ${r?.archive_issue ? `<p class="tiny value-amber space-top">${esc(r.archive_issue)}</p>` : ""}`;
}

function drawCandidates() {
  const host = $("#candidate-table");
  if (!host) return;
  const r = currentDecision(), recording = ctx.mode === "video";
  const options = r?.candidates || [];
  const plan = selectedCandidate();
  host.innerHTML = candidatesTable(options, r?.selected_id, ctx.filter, ctx.previewId);
  $("#candidate-count").textContent = options.length + " 条候选 · " + options.filter(c => c.valid).length + " 条可行";
  for (const item of $("#candidate-filter").children) item.classList.toggle("active", item.dataset.id === ctx.filter);
  $("#candidate-summary").innerHTML = plan ? `<div class="card-body"><div class="actions"><strong>${esc(plan.target)} · ${esc(plan.label)}</strong>${pill(plan.id === r.selected_id ? (recording ? "本段最优候选" : "下一段") : "轨迹预览", plan.valid ? "teal" : "amber")}${btn("candidate-detail", "设备轨迹", "motion", "small flat right", `data-id="${esc(plan.id)}"`)}
    ${recording && videoPath() && !ctx.video.playing ? btn("video-motion", "演示路径", "play", "small") : ""}</div>
    <p class="tiny muted space-top">${recording ? seconds(r.time_s) + " 帧 · " : ""}${esc(plan.reasons.join(" · ") || plan.explanation)}</p></div>` : "";
}

function drawTracks() {
  const el = $("#track-table");
  if (!el) return;
  const obs = (ctx.mode === "sim" ? ctx.run?.current : ctx.video.sample)?.observation;
  el.innerHTML = tracksTable(obs);
  $("#tracking-state").innerHTML = obs ? pill(`可信度 ${pct(obs.confidence,0)} · 遮挡 ${pct(obs.steam_fraction,0)}`, obs.usable ? "teal" : "amber") : "";
}

function updateLive() {
  if (ctx.route !== "live") return;
  updateMetrics();
  drawLiveActions();
  drawCandidates();
  drawTracks();
  if (ctx.mode === "video") showSample(ctx.video.sample, ctx.cfg, videoPath(), true);
  else if (ctx.live.animation) drawAnimation(ctx.live.animation, true);
  else showSample(ctx.run?.current, ctx.runConfig || ctx.cfg, selectedCandidate()?.valid ? selectedCandidate() : null);
}

function cancelAuto() {
  ctx.live.auto = false;
  clearTimeout(ctx.live.timer);
  ctx.live.timer = null;
}

function pauseRun() {
  cancelAuto();
  ctx.live.pauseRequested = true;
  if (ctx.live.animation) ctx.live.animation.paused = true;
  updateMetrics();
  drawLiveActions();
}

async function startRun({useExperiment = false, navigate = false} = {}) {
  if (ctx.live.starting || ctx.live.busy) return;
  cancelAuto();
  stopVideo();
  ctx.live.starting = true;
  ctx.live.animation = null;
  ctx.live.pauseRequested = false;
  const prior = ctx.run;
  const config = structuredClone(ctx.cfg);
  const revision = ctx.revision;
  const settings = useExperiment || !prior ? ctx.exp : {caseId:prior.case.id, seed:prior.seed, strategy:prior.strategy};
  try {
    updateLive();
    ctx.run = await post("/api/runs", {case_id:settings.caseId, seed:settings.seed, strategy:settings.strategy, revision});
    ctx.runConfig = config;
    ctx.mode = "sim";
    ctx.previewId = null;
    if (navigate && ctx.route !== "live") location.hash = "live";
    else if (ctx.route === "live") render();
  } finally {
    ctx.live.starting = false;
    if (ctx.route === "live") drawLiveActions();
  }
}

function queueAdvance(delay = 0) {
  clearTimeout(ctx.live.timer);
  if (ctx.live.auto && !isTerminal(ctx.run) && ctx.route === "live" && ctx.mode === "sim") {
    ctx.live.timer = setTimeout(() => advance().catch(failLive), delay);
  }
}

function failLive(error) {
  cancelAuto();
  ctx.live.busy = false;
  updateLive();
  toast(error.message, true);
}

function commit(result) {
  ctx.run = result.run;
  ctx.previewId = null;
  if (isTerminal(ctx.run)) {
    cancelAuto();
    toast(statusName[ctx.run.status] + " · " + (ctx.run.report ? pct(ctx.run.report.removal_rate, 2) : ctx.run.reason));
  }
  updateLive();
}

function phaseAt(event, t) {
  const trajectory = event.action?.trajectory || [];
  const elapsed = t - event.start_s;
  return (trajectory.find(k => k.t_s > elapsed) || trajectory.at(-1))?.phase || "ready";
}

function sampleAt(frames, t) {
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time_s <= t) lo = mid;
    else hi = mid - 1;
  }
  return frames[lo];
}

function drawAnimation(a, force = false) {
  if (ctx.route !== "live" || ctx.mode !== "sim") return;
  const sample = sampleAt(a.samples, a.cursor);
  if (sample !== a.frame || force) {
    a.frame = sample;
    showSample(sample, ctx.runConfig, a.event.action);
  }
  const phase = phaseAt(a.event, a.cursor);
  if (a.phase !== phase || force) {
    a.phase = phase;
    const flow = $("#flowline");
    if (flow) flow.innerHTML = phases(phase);
    const label = $("#phase-state");
    if (label) label.textContent = PHASE[phase];
  }
  const now = performance.now();
  if (force || now - (a.lastMetrics || 0) > 110) {
    a.lastMetrics = now;
    updateMetrics();
    const bar = $("#live-progress");
    if (bar) bar.style.width = Math.min(100, a.cursor / ctx.runConfig.objectives.time_limit_s * 100) + "%";
  }
}

function animationTick(a, now) {
  if (ctx.live.animation !== a) return;
  if (!a.paused) a.cursor += Math.min(.15, (now - a.previous) / 1000) * ctx.live.speed;
  a.previous = now;
  if (a.cursor >= a.event.start_s + a.event.motion_s) {
    ctx.live.animation = null;
    commit(a.result);
    queueAdvance(40);
    return;
  }
  if (!a.paused) drawAnimation(a);
  requestAnimationFrame(t => animationTick(a, t));
}

async function advance() {
  const r = ctx.run;
  if (!r || ctx.live.busy || ctx.live.animation || isTerminal(r)) return;
  ctx.live.busy = true;
  ctx.live.pauseRequested = false;
  drawLiveActions();
  updateMetrics();
  try {
    const result = await post("/api/runs/" + r.id + "/step", {expected_index:r.last_index});
    ctx.live.busy = false;
    const event = result.event;
    if (event?.motion_s > 0 && event.frames.length) {
      const a = {result, event, samples:[event.before, ...event.frames], cursor:event.start_s, previous:performance.now(),
        frame:null, paused:ctx.live.pauseRequested || ctx.route !== "live" || ctx.mode !== "sim"};
      ctx.live.animation = a;
      updateLive();
      requestAnimationFrame(now => animationTick(a, now));
    } else {
      commit(result);
      queueAdvance(Math.max(90, ((event?.end_s || 0) - (event?.start_s || 0)) * 1000 / ctx.live.speed));
    }
  } finally {
    ctx.live.busy = false;
    if (ctx.route === "live") drawLiveActions();
  }
}

function toggleRun() {
  if (ctx.live.auto || (ctx.live.animation && !ctx.live.animation.paused)) {
    pauseRun();
    return;
  }
  ctx.live.auto = true;
  ctx.live.pauseRequested = false;
  if (ctx.live.animation) ctx.live.animation.paused = false;
  else queueAdvance();
  drawLiveActions();
}

async function intervene(kind, body) {
  if (!ctx.run || ctx.live.busy || ctx.live.animation || isTerminal(ctx.run)) return;
  ctx.live.busy = true;
  drawLiveActions();
  try {
    const result = await post("/api/runs/" + ctx.run.id + "/" + kind, body);
    ctx.live.busy = false;
    commit(result);
    toast(kind === "disturb" ? "已根据扰动后的新画面重新规划" : result.run.reason || "已恢复并重新规划");
  } finally {
    ctx.live.busy = false;
    drawLiveActions();
  }
}

async function loadData() {
  ctx.data = await request("/api/data");
  return ctx.data;
}

function stopVideo(message) {
  const v = ctx.video;
  v.controller?.abort();
  v.controller = null;
  v.token++;
  v.playing = false;
  v.preview = null;
  if (message) v.message = message;
}

function selectVideo(id) {
  stopVideo();
  const v = ctx.video;
  v.id = id;
  const info = videoInfo();
  v.time = Math.min(info?.duration_s - 1 / info?.fps || 0, id === "01" ? 285 : id === "00" ? 30 : 0);
  v.clip = info ? {start:v.time, end:Math.min(info.duration_s, v.time + 35)} : null;
  v.sample = v.meta = v.plan = ctx.previewId = null;
  v.segments.clear();
}

function drawVideoControls() {
  const controls = $("#video-controls");
  if (!controls || ctx.mode !== "video") return;
  const v = ctx.video, info = videoInfo();
  const options = (ctx.data?.videos || []).map(x => option(x.id, "录像 " + x.id + " · " + num(x.fps, 0) + " fps", v.id)).join("");
  controls.innerHTML = `<div class="card-body"><div class="scrubber"><select id="video-source" aria-label="选择原始录像">${options}</select>
    <span id="video-play-button">${btn("video-toggle", v.playing ? "暂停" : "播放", v.playing ? "pause" : "play", "small primary", info ? "" : "disabled")}</span>
    <input id="video-position" type="range" min="${v.clip?.start ?? 0}" max="${v.clip?.end ?? 0}" step="any" value="${v.time}" aria-label="片段进度"><span id="video-time" class="mono tiny">${seconds(v.time)}</span>
    </div>${info && v.clip ? `<div class="scrubber clip-inputs space-top">
      <label>起点（秒） <input id="clip-start" type="number" min="0" max="${info.duration_s}" step="any" value="${v.clip.start}"></label>
      <label>结束 / 动作截止（秒） <input id="clip-end" type="number" min="0" max="${info.duration_s}" step="any" value="${v.clip.end}"></label>
      ${btn("video-clip", "分析片段", "target", "small")}
    </div>` : ""}</div>`;
}

function drawVideoFrame() {
  if (ctx.route !== "live" || ctx.mode !== "video") return;
  const v = ctx.video;
  if (!v.preview) showSample(v.sample, ctx.cfg, videoPath(), true);
  const position = $("#video-position");
  if (position && document.activeElement !== position) position.value = v.time;
  const time = $("#video-time");
  if (time) time.textContent = seconds(v.time);
  const now = performance.now();
  if (now - (v.updatedAt || 0) > 180 || !v.playing) {
    v.updatedAt = now;
    updateMetrics();
    drawTracks();
    drawLiveActions();
  }
}

function acceptVideo(row) {
  const v = ctx.video;
  if (row.kind === "frame") {
    const obs = row.observation;
    v.time = obs.captured_s;
    v.sample = {time_s:v.time, observation:obs, reconstruction:row.meta?.reconstruction,
      image_url:"data:image/jpeg;base64," + row.jpeg};
    v.meta = {...row.meta, skipped:row.skipped};
    const skipped = v.meta.skipped_segments;
    if (skipped) {
      for (let segment = skipped[0]; segment < skipped[1]; segment++) {
        if (!v.segments.has(segment)) v.segments.set(segment, {segment,
          deadline_s:Math.min(v.clip.end, v.clip.start + segment + 1), reason:"截止前未取得新帧"});
      }
    }
    const prior = v.plan;
    if (v.meta.plan) {
      v.plan = {...v.meta.plan, segment:v.meta.segment};
      v.segments.set(v.meta.segment, v.plan);
    } else if (v.meta.planning_reason || v.meta.segment !== prior?.segment) {
      v.plan = null;
      if (v.meta.planning_reason && !v.segments.has(v.meta.segment)) {
        v.segments.set(v.meta.segment, {segment:v.meta.segment, time_s:v.time,
          deadline_s:Math.min(v.clip.end, v.clip.start + v.meta.segment + 1), reason:v.meta.planning_reason});
      }
    }
    const stale = row.lag_s > ctx.cfg.camera.max_frame_age_s;
    if (stale) v.plan = null;
    if (v.plan !== prior) {
      ctx.previewId = null;
      drawCandidates();
    }
    v.message = stale ? "录像处理落后，等待新帧" :
      v.meta.planning_reason || (v.plan?.selected_id ? "已在截止前选出可行候选" : v.plan?.reason) || "等待本段规划结果";
    v.lastFrameAt = performance.now();
  } else {
    v.message = row.reason || "等待新帧";
    v.playing = false;
  }
  drawVideoFrame();
}

async function openVideo(start, playing) {
  const info = videoInfo();
  if (!info || !ctx.video.clip) return;
  stopVideo();
  const v = ctx.video;
  const controller = new AbortController();
  const token = v.token;
  v.controller = controller;
  v.playing = playing;
  v.plan = null;
  ctx.previewId = null;
  v.lastFrameAt = performance.now();
  v.message = v.sample ? "等待新帧" : "加载模型并准备分析片段";
  drawVideoControls();
  updateLive();
  const params = new URLSearchParams({start_s:String(start), clip_start_s:String(v.clip.start), end_s:String(v.clip.end),
    revision:ctx.revision, analyze:"true", strategy:ctx.exp.strategy});
  if (!playing) params.set("limit", "1");
  try {
    const response = await fetch("/api/data/videos/" + encodeURIComponent(info.id) + "/stream?" + params, {signal:controller.signal});
    if (!response.ok) throw new Error((await response.json()).detail || "录像输入无法读取");
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let pending = "";
    try {
      while (v.token === token) {
        const {done, value} = await reader.read();
        if (done) break;
        pending += value;
        const rows = pending.split("\n");
        pending = rows.pop();
        for (const line of rows) if (line && v.token === token) acceptVideo(JSON.parse(line));
      }
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    if (error.name !== "AbortError" && v.token === token) {
      v.message = "录像输入中断 · " + error.message;
      toast(v.message, true);
    }
  } finally {
    if (v.token === token) {
      v.playing = false;
      v.controller = null;
      if (ctx.mode === "video" && ctx.route === "live") {
        drawVideoControls();
        drawVideoFrame();
        drawCandidates();
      }
    }
  }
}

async function switchMode(mode) {
  if (ctx.mode === mode) return;
  pauseRun();
  stopVideo();
  ctx.mode = mode;
  if (mode === "video") {
    if (!ctx.data) await loadData();
    if (!ctx.video.id) {
      selectVideo(ctx.data.videos[0]?.id || "");
    }
  }
  render();
  if (ctx.route === "live" && ctx.mode === "video" && !ctx.video.sample && ctx.video.id) void openVideo(ctx.video.time, false);
}

function previewVideoMotion() {
  const v = ctx.video, path = videoPath();
  if (!path || !v.sample || v.playing || !ctx.scene) return;
  const {observation} = v.sample;
  const preview = v.preview = {start:performance.now()};
  function draw(now) {
    if (v.preview !== preview || ctx.route !== "live" || ctx.mode !== "video") return;
    const t = (now - preview.start) / 1000 * 4;
    if (t >= path.duration_s) {
      v.preview = null;
      drawVideoFrame();
      return;
    }
    const pose = trajectoryPose(ctx.cfg, path, t);
    ctx.scene?.set(ctx.cfg, pose, observation, path, observation.surface, true);
    $("#axis-values").textContent = `路径演示 · ${PHASE[pose.phase]} · ${seconds(t)}`;
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
}

function updateConfigPreview() {
  try {
    const pose = initialPose(ctx.draft);
    ctx.scene?.set(ctx.draft, pose, null, null);
    drawConfigAxis(pose);
  } catch {
    // Unsaved geometry is validated on save.
  }
}

function readConfigField(field) {
  const key = field.dataset.configPath;
  const path = key.split(".");
  const parent = path.slice(0,-1).reduce((obj, segment) => obj[segment], ctx.draft);
  let value = field.value;
  try {
    if (field.type === "checkbox") value = field.checked;
    else if (field.dataset.type === "json") value = JSON.parse(value);
    else if (field.type === "number") value = key === "tilt.working_deg" && !field.value ? null : field.valueAsNumber;
  } catch {
    field.setCustomValidity("请输入有效 JSON");
    field.reportValidity();
    return false;
  }
  field.setCustomValidity("");
  parent[path.at(-1)] = value;
  return true;
}

async function useConfig(revision, cfg) {
  stopVideo();
  ctx.video.sample = ctx.video.meta = ctx.video.plan = ctx.previewId = null;
  ctx.video.segments.clear();
  ctx.cfg = cfg || await request("/api/configs/" + revision);
  ctx.revision = revision;
  localStorage.setItem("slag-config", revision);
  ctx.draft = structuredClone(ctx.cfg);
  ctx.exp.weights = null;
}

async function saveConfig(config) {
  const saved = await post("/api/configs", config);
  await useConfig(saved.revision, saved.config);
  ctx.revisions = await request("/api/configs");
  render();
  toast("已保存配置版本 " + saved.revision);
}

async function loadBatch() {
  ctx.batch = await request("/api/batch/latest").catch(() => null);
}

async function watchJob(id, kind) {
  const key = kind === "training" ? "trainJob" : "batchJob";
  const route = kind === "training" ? "data" : "experiment";
  let status;
  do {
    status = await request("/api/jobs/" + id);
    ctx[key] = status;
    if (ctx.route === route && !dialog.open) render();
    if (status.status === "done") {
      if (kind === "training") await loadData();
      else await loadBatch();
      if (ctx.route === route || ctx.route === "report") render();
      toast(kind === "training" ? "训练与独立录像验证已完成" : "批量评测已完成");
    } else if (status.status === "failed") {
      toast(status.message, true);
    } else {
      await new Promise(resolve => setTimeout(resolve, 900));
    }
  } while (["queued","running"].includes(status.status));
}

async function upload(type, input, videoId, furnaceId) {
  const file = $(input)?.files[0];
  if (!file) throw new Error("请选择需要导入的文件");
  const form = new FormData();
  form.set("file", file);
  if (videoId) form.set("video_id", videoId);
  if (furnaceId?.trim()) form.set("furnace_id", furnaceId.trim());
  await request("/api/data/" + type, {method:"POST", body:form});
  await loadData();
  if (ctx.route === "data") render();
  toast("已导入 " + file.name);
}

async function getArchives() {
  ctx.archives = await request("/api/runs");
}

async function openTrace(id) {
  const r = ctx.replay;
  r.playing = false;
  r.trace = id ? await request("/api/runs/" + id + "/trace") : null;
  r.time = 0;
  r.frame = r.event = r.candidate = null;
  r.frames = r.trace ? [r.trace.initial, ...r.trace.events.flatMap(e => [...e.frames, e.after])].sort((a,b) => a.time_s - b.time_s) : [];
  if (ctx.route === "replay") render();
}

function replayEvent(r) {
  const events = r.trace.events;
  for (let i = events.length - 1; i >= 0; i--) if (events[i].start_s <= r.time) return events[i];
  return null;
}

function drawReplay(force = false) {
  const r = ctx.replay;
  if (ctx.route !== "replay" || !r.trace) return;
  const frame = sampleAt(r.frames, r.time), event = replayEvent(r);
  const candidate = r.candidate && (event === r.event || force) && event?.decision.find(c => c.id === r.candidate);
  if (frame !== r.frame || event !== r.event || force) {
    r.frame = frame;
    const path = candidate?.valid ? candidate : event?.action;
    showSample(frame, r.trace.config, path);
  }
  $("#replay-clock").textContent = seconds(r.time);
  $("#replay-time").value = r.time;
  $("#replay-toggle").innerHTML = icon(r.playing ? "pause" : "play") + (r.playing ? "暂停" : "播放");
  if (event === r.event && !force) return;
  r.event = event;
  if (!force) r.candidate = null;
  for (const row of page.querySelectorAll("[data-event]")) row.classList.toggle("selected", Number(row.dataset.event) === event?.index);
  $("#event-badge").innerHTML = pill(event ? `事件 ${event.index}` : "起始帧", "blue");
  if (!event) {
    $("#event-detail").innerHTML = '<p class="muted">起始识别帧。拖动时间轴或选中事件查看详情。</p>';
    return;
  }
  const action = event.action, act = event.actual, before = event.before.observation, after = event.after.observation;
  $("#event-detail").innerHTML = `<p>${esc(action ? action.target + " · " + action.label : event.note || "等待新帧")}</p>
    <p class="tiny muted space-top">帧 ${before.frame} → ${after.frame} · ${num(event.start_s)}–${num(event.end_s)} s · 置信度 ${pct(before.confidence,0)} → ${pct(after.confidence,0)}</p>
    ${act ? `<div class="kv"><div><small>预计覆盖面积下降</small><strong>${num(action.metrics.expected_m2,4)} m²</strong></div><div><small>真值覆盖面积下降</small><strong>${num(act.removed_m2,4)} m²</strong></div>
      ${act.density_before === undefined ? "" : `<div><small>平均浓度</small><strong>${num(act.density_before,3)} → ${num(act.density_after,3)}</strong></div><div><small>覆盖率</small><strong>${pct(act.coverage_before,2)} → ${pct(act.coverage_after,2)}</strong></div>`}</div>${btn("replay-action-detail","设备轨迹","motion","small")}` : ""}
    ${event.decision.length ? `<details class="space-top" ${candidate ? "open" : ""}><summary>本轮 ${event.decision.length} 条候选</summary>${candidatesTable(event.decision, action?.id, "all", candidate?.id, "replay-candidate")}</details>` : ""}`;
}

function playReplay() {
  const r = ctx.replay;
  if (!r.trace) return;
  r.playing = !r.playing;
  if (!r.playing) return drawReplay();
  if (r.time >= r.trace.summary.elapsed_s) r.time = 0;
  const trace = r.trace;
  let previous = performance.now();
  const tick = now => {
    if (!r.playing || r.trace !== trace || ctx.route !== "replay") return;
    r.time = Math.min(trace.summary.elapsed_s, r.time + (now - previous) / 1000 * r.speed);
    previous = now;
    if (r.time >= trace.summary.elapsed_s) r.playing = false;
    drawReplay();
    if (r.playing) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function routeChanged() {
  const name = location.hash.slice(1);
  const route = NAV.some(row => row[0] === name) ? name : "live";
  if (ctx.route !== route) {
    if (ctx.route === "live") {
      pauseRun();
      stopVideo("录像已暂停");
    }
    if (ctx.route === "replay") ctx.replay.playing = false;
    ctx.route = route;
    window.scrollTo(0, 0);
  }
  $("#sidebar").classList.remove("open");
  if (!ctx.boot) return;
  render();
  if (route === "live" && !ctx.run) {
    await startRun();
  } else if (route === "data") {
    await loadData();
  } else if (route === "experiment") {
    await loadBatch();
  } else if (route === "replay" || route === "report") {
    await getArchives();
    if (route === "replay" && !ctx.replay.trace && ctx.archives.length && ctx.route === route) await openTrace(ctx.archives[0].id);
    if (route === "report") await loadBatch();
  }
  if (ctx.route === route && route !== "live") render();
}

async function handleAction(button) {
  const action = button.dataset.action, id = button.dataset.id;
  const index = Number(button.dataset.index);
  switch (action) {
    case "help":
      modal("使用说明", `<p>熔线使用单套 360 吨级参考设备和参数化仿真。每次只提交一段经过约束检查的设备轨迹，下一段根据执行后的新相机帧生成。</p>
        <ol class="spaced-list"><li>在实时作业单步或连续运行，点击候选查看打分与设备轨迹。</li><li>在仿真实验载入工况、调整权重和比较策略。</li><li>在数据与训练导入抽帧标注，按完整录像验证模型。</li><li>在过程回放与结果报告查看同步记录和真值验收结果。</li></ol>
        <p class="muted">现场待验证。模拟铁损使用风险指数，视觉类别未经厚度实测校准。</p>`);
      break;
    case "close-dialog": dialog.close(); break;
    case "view-reset": ctx.scene?.reset(); break;
    case "view-top": ctx.scene?.top(); break;
    case "mode-sim": await switchMode("sim"); break;
    case "mode-video": await switchMode("video"); break;
    case "run-reset": await startRun(); break;
    case "run-toggle": toggleRun(); break;
    case "run-step": await advance(); break;
    case "disturb": await intervene("disturb"); break;
    case "candidate-filter":
      ctx.filter = id;
      drawCandidates();
      break;
    case "candidate":
      ctx.previewId = id;
      drawCandidates();
      if (ctx.mode === "video") {
        ctx.video.preview = null;
        drawVideoFrame();
      } else if (!ctx.live.animation) {
        const c = selectedCandidate();
        showSample(ctx.run.current, ctx.runConfig, c?.valid ? c : null);
      }
      break;
    case "candidate-detail": {
      const row = currentDecision()?.candidates.find(c => c.id === id);
      if (row) modal(row.target + " · " + row.label, candidateDetail(row));
      break;
    }
    case "open-live-report":
      ctx.singleReport = ctx.run.report;
      location.hash = "report";
      break;
    case "video-toggle": {
      const v = ctx.video, info = videoInfo();
      if (v.playing) {
        void openVideo(v.time, false);
      } else if (info) {
        const next = v.time + (v.sample ? 1 / info.fps : 0);
        const t = next >= v.clip.end - 1e-7 ? v.clip.start : next;
        void openVideo(t, true);
      }
      break;
    }
    case "video-clip": {
      const start = $("#clip-start").valueAsNumber, end = $("#clip-end").valueAsNumber;
      const info = videoInfo();
      if (!info || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end > info.duration_s || end <= start ||
          Math.ceil(start * info.fps - 1e-7) >= Math.ceil(end * info.fps - 1e-7)) {
        throw new Error("请填写录像范围内包含采样帧的片段");
      }
      const v = ctx.video;
      v.clip = {start, end};
      v.time = start;
      v.sample = v.meta = null;
      v.segments.clear();
      void openVideo(start, true);
      break;
    }
    case "video-motion": previewVideoMotion(); break;
    case "video-export": {
      const v = ctx.video;
      saveBlob({video_id:v.id, clip:v.clip, model_version:v.meta?.model_version,
        strategy:ctx.exp.strategy, segments:Array.from(v.segments.values())}, `录像-${v.id}-分段规划.json`);
      break;
    }
    case "config-tab":
      if ([...page.querySelectorAll("[data-config-path]")].every(readConfigField)) {
        ctx.configGroup = id;
        render();
      }
      break;
    case "config-save":
    case "config-copy":
    case "config-export":
      if (![...page.querySelectorAll("[data-config-path]")].every(readConfigField)) break;
      if (action === "config-export") saveBlob(ctx.draft, "slag-config.json");
      else {
        const copy = structuredClone(ctx.draft);
        if (action === "config-copy") copy.name = copy.name.slice(0, 73) + " · 副本";
        await saveConfig(copy);
      }
      break;
    case "config-import": $("#config-file").click(); break;
    case "calibrate":
      if (!ctx.data) await loadData();
      modal("录像平面标定", calibrationForm(ctx.data, ctx.draft.tilt.initial_deg));
      break;
    case "calibrate-solve": {
      const rows = $("#cal-points").value.trim().split(/\r?\n/).filter(Boolean).map(line => line.trim().split(/[\s,，]+/).map(Number));
      if (rows.length < 4 || rows.some(row => row.length !== 4 || row.some(value => !Number.isFinite(value)))) throw new Error("每行输入四个有效数值，至少四行");
      const result = await post("/api/calibration/solve", {video_id:$("#cal-video").value, revision:ctx.revision, tilt_deg:$("#cal-tilt").valueAsNumber, pixel:rows.map(r => r.slice(0,2)), plane_m:rows.map(r => r.slice(2))});
      ctx.draft.camera.video_plane = result.calibration;
      dialog.close();
      render();
      toast("标定已填入草稿 · 拟合 RMS " + num(result.calibration.rms_m, 4) + " m" + (result.ready_for_review ? "" : " · 超出回放投影阈值"));
      break;
    }
    case "upload-video": await upload("videos", "#video-upload"); break;
    case "upload-labels": await upload("labels", "#label-upload", $("#label-video").value, $("#furnace-id").value); break;
    case "train-open": modal("新建训练任务", trainForm(ctx.data)); break;
    case "train-submit": {
      const validationIds = [...dialog.querySelectorAll('input[name="val-set"]:checked')].map(input => input.value);
      const {id} = await post("/api/data/train", {validation_ids:validationIds, seed:$("#train-seed").valueAsNumber});
      dialog.close();
      void watchJob(id, "training").catch(error => toast(error.message,true));
      break;
    }
    case "pick-case": {
      const selection = ctx.boot.scenarios.find(s => s.id === id);
      ctx.exp.caseId = id;
      ctx.exp.seed = selection.seed;
      render();
      break;
    }
    case "load-experiment":
      ctx.exp.seed = $("#experiment-seed").valueAsNumber;
      await startRun({useExperiment:true,navigate:true});
      break;
    case "save-weights": {
      const draft = structuredClone(ctx.cfg);
      draft.objectives.weights = Object.fromEntries([...page.querySelectorAll("[data-weight]")].map(input => [input.dataset.weight, input.valueAsNumber]));
      await saveConfig(draft);
      break;
    }
    case "batch-run": {
      const {id} = await post("/api/batch", {strategies:ctx.exp.batchStrategies, revision:ctx.revision});
      void watchJob(id, "batch").catch(error => toast(error.message,true));
      break;
    }
    case "batch-detail": {
      const row = ctx.batch.results[index];
      if (row) modal(row.case.name + " · " + ctx.boot.strategies[row.strategy], reportBody(row));
      break;
    }
    case "replay-toggle": playReplay(); break;
    case "replay-rewind":
      ctx.replay.playing = false;
      ctx.replay.time = 0;
      ctx.replay.candidate = null;
      drawReplay(true);
      break;
    case "replay-event": {
      const e = ctx.replay.trace.events.find(row => row.index === index);
      ctx.replay.playing = false;
      ctx.replay.time = e.start_s;
      ctx.replay.candidate = null;
      drawReplay(true);
      break;
    }
    case "replay-action-detail":
      if (ctx.replay.event?.action) modal("回放 · 设备轨迹", candidateDetail(ctx.replay.event.action));
      break;
    case "replay-candidate": {
      ctx.replay.candidate = id;
      const row = ctx.replay.event?.decision.find(c => c.id === id);
      drawReplay(true);
      if (row) modal("回放 · " + row.target + " · " + row.label, candidateDetail(row));
      break;
    }
  }
}

document.addEventListener("click", event => {
  const button = event.target.closest("[data-action]");
  if (!button || button.disabled) return;
  handleAction(button).catch(error => toast(error.message, true));
});

document.addEventListener("change", async event => {
  const input = event.target;
  try {
    if (input.dataset.configPath && readConfigField(input)) updateConfigPreview();
    if (input.id === "config-revision") {
      await useConfig(input.value);
      render();
    }
    if (input.id === "config-file" && input.files[0]) await saveConfig(JSON.parse(await input.files[0].text()));
    if (input.id === "run-speed") ctx.live.speed = Number(input.value);
    if (input.id === "fault-select") await intervene("fault", {fault:input.value || null});
    if (input.id === "overlay") {
      ctx.mask = input.checked;
      if (ctx.shown) showSample(ctx.shown.sample, ctx.shown.cfg, ctx.shown.path, ctx.shown.record);
    }
    if (input.id === "video-source") {
      selectVideo(input.value);
      void openVideo(ctx.video.time, false);
    }
    if (input.id === "video-position") void openVideo(Number(input.value), ctx.video.playing);
    if (input.id === "experiment-seed") ctx.exp.seed = input.valueAsNumber;
    if (input.id === "experiment-strategy") ctx.exp.strategy = input.value;
    if (input.dataset.weight) ctx.exp.weights = Object.fromEntries([...page.querySelectorAll("[data-weight]")].map(field => [field.dataset.weight, field.valueAsNumber]));
    if (input.name === "batch-strategy") ctx.exp.batchStrategies = [...page.querySelectorAll('input[name="batch-strategy"]:checked')].map(field => field.value);
    if (input.id === "replay-run") await openTrace(input.value);
    if (input.id === "replay-speed") ctx.replay.speed = Number(input.value);
    if (input.id === "report-run") {
      ctx.singleReport = input.value ? await request("/api/runs/" + input.value + "/report") : null;
      if (ctx.route === "report") render();
    }
  } catch (error) {
    toast(error.message, true);
  }
});

document.addEventListener("input", event => {
  if (event.target.id === "replay-time") {
    ctx.replay.playing = false;
    ctx.replay.time = Number(event.target.value);
    drawReplay();
  }
  if (event.target.id === "video-position") $("#video-time").textContent = seconds(Number(event.target.value));
});

$("#menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
window.addEventListener("hashchange", () => routeChanged().catch(error => toast(error.message, true)));
setInterval(() => {
  const v = ctx.video;
  if (v.playing && ctx.mode === "video" && ctx.route === "live" && performance.now() - v.lastFrameAt > ctx.cfg.camera.max_frame_age_s * 1000) {
    v.message = "相机帧过期 · 等待新帧";
    v.plan = ctx.previewId = null;
    drawCandidates();
    drawVideoFrame();
  }
}, 400);

async function init() {
  const [boot, revisions] = await Promise.all([request("/api/bootstrap"), request("/api/configs")]);
  ctx.boot = boot;
  ctx.revisions = revisions;
  const remembered = localStorage.getItem("slag-config");
  const active = revisions.find(r => r.revision === remembered);
  await useConfig(active?.revision || boot.default_revision, active && !active.built_in ? null : boot.config);
  await routeChanged();
}

init().catch(error => {
  page.innerHTML = `<div class="notice amber">${esc(error.message)}<br>请确认本地仿真服务已启动后刷新页面。</div>`;
});

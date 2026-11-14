import {
  axisValues, batchTable, blank, btn, card, download, esc, heading, icon, metric, num, option,
  pairViews, pct, pill, reportBody, seconds, table,
} from "./ui.js";

export const GROUPS = {tank:"罐体", arm:"扒渣机", tilt:"倾翻机构", camera:"工业相机", slag:"残渣仿真", objectives:"优化目标"};
const SOURCE = ["仿真假设", "企业提供", "现场标定"];
const AXIS_LABEL = {extension:"伸缩", lift:"升降", swing:"回转", pitch:"俯仰"};
const LABEL = {slag:"残渣", iron:"铁水", tool:"机械臂", mouth:"罐口", steam:"蒸汽", unknown:"无法判断"};
const WEIGHTS = {removal:"带出渣量", iron:"铁损风险", disturbance:"液面扰动", time:"时间", future:"后续漂移"};
const FIELDS = {
  tank: {
    length_m:"外形长度 (m)", width_m:"外形宽度 (m)", height_m:"外形高度 (m)", trunnion_span_m:"耳轴中心距 (m)",
    pivot_m:"倾转轴中心 [X, Y, Z] (m)", inner_radii_m:"液面半轴 [X, Y] (m)", rim_radii_m:"罐口内半轴 [X, Y] (m)",
    bottom_scale:"内腔底部缩放系数", rim_z_m:"罐口高度 · 罐体坐标 (m)", liquid_z_m:"液面高度 · 罐体坐标 (m)",
    outlet_x_m:"出渣边中心 X (m)", outlet_width_m:"出渣边宽度 (m)", outlet_extension_m:"出渣边伸出长度 (m)",
    outlet_drop_m:"出渣槽相对罐口的降低量 (m)",
  },
  arm: {
    base_m:"设备基点 [X, Y, Z] (m)", boom_z_m:"伸缩横梁高度 (m)", tool_size_m:"扒渣头 [宽, 长, 厚] (m)",
    collision_margin_m:"碰撞余量 (m)", skim_depth_m:"扒渣板底缘贴面量 (m)", safety_boxes:"设备坐标禁入盒 (JSON)",
  },
  tilt: {initial_deg:"初始倾角 (°)", min_deg:"最小倾角 (°)", max_deg:"最大倾角 (°)", speed_deg_s:"倾转角速度 (°/s)",
    step_deg:"单次调整角度 (°)", working_deg:"作业倾角 (°，空白时自动选择)", allow_planner:"允许规划器提出倾角调整"},
  camera: {
    resolution:"分辨率 [宽, 高] (px)", fps:"虚拟相机帧率 (fps)", position_m:"安装位置 [X, Y, Z] (m)",
    look_at_m:"视线指向 [X, Y, Z] (m)", intrinsics:"内参 [fx, fy, cx, cy] (px)", distortion:"畸变 [k1, k2, p1, p2, k3]",
    device_to_camera:"设备到相机变换 (4×4 JSON 或 null)", video_plane:"录像平面标定 (JSON 或 null)",
    max_frame_age_s:"最大帧龄 (s)", steam_limit:"遮挡阈值 (0–1)", min_confidence:"最低可信度 (0–1)",
    feedback_s:"设备反馈延迟 (s)", noise:"虚拟相机噪声强度",
  },
  slag: {distribution:"初始分布", grid_step_m:"面积离散步长 (m)", area_scale:"高浓度区面积缩放",
    drift_m_s:"漂移基准速度 (m/s)", aggregation:"聚合程度 (0–1)", disturbance_shift_m:"扫后扰动位移尺度 (m)",
    contact_efficiency:"贴面收集效率 (0–1)", initial_film:"初始基础渣层浓度 (0–1)",
    coverage_threshold:"覆盖面积统计阈值 (0–1)", skim_fraction:"单轮最高收集份额 (0–1)"},
  objectives: {removal_target:"面积去除率下限 (0–1，严格大于)", time_limit_s:"时间上限 (s)", iron_limit:"累计铁损风险上限",
    recognition_budget_s:"识别计时基线 (s)", planning_budget_s:"规划计时基线 (s)", hold_retry_s:"保持后重试间隔 (s)"},
};

function control(path, label, value, choices = null) {
  const field = `data-config-path="${esc(path)}"`;
  const area = ["arm.safety_boxes", "camera.device_to_camera", "camera.video_plane"].includes(path);
  let input;
  if (choices) input = `<select ${field}>${choices.map(x => option(x, x, value)).join("")}</select>`;
  else if (typeof value === "boolean") return `<label class="check full"><input type="checkbox" ${field} ${value ? "checked" : ""}>${esc(label)}</label>`;
  else if (area) input = `<textarea ${field} data-type="json" class="mono" spellcheck="false">${esc(JSON.stringify(value, null, 2))}</textarea>`;
  else if (Array.isArray(value)) input = `<input type="text" ${field} data-type="json" class="mono" value="${esc(JSON.stringify(value))}">`;
  else input = `<input type="${typeof value === "number" || path === "tilt.working_deg" ? "number" : "text"}" ${field} step="any" value="${esc(value)}">`;
  return `<label class="field-label ${area ? "full" : ""}"><span>${esc(label)}</span>${input}</label>`;
}

export function configPage(draft, group, revisions, revision) {
  const data = draft[group];
  const tab = Object.entries(GROUPS).map(([key, title]) => btn("config-tab", title, null, "small " + (key === group ? "active" : ""), `data-id="${key}"`)).join("");
  let fields = control(group + ".source", "参数来源", data.source, SOURCE);
  for (const [key, title] of Object.entries(FIELDS[group])) {
    fields += control(group + "." + key, title, data[key], key === "distribution" ? ["工况定义", "均匀", "聚集"] : null);
  }
  if (group === "arm") {
    for (const [axis, label] of Object.entries(AXIS_LABEL)) {
      const unit = axis === "extension" || axis === "lift" ? "m" : "°";
      fields += `<h3 class="group-title full">${label}轴</h3>`;
      for (const [key, title] of Object.entries({min:"行程下界", max:"行程上界", initial:"初始位置", speed:"速度上限"})) {
        fields += control("arm." + axis + "." + key, title + " (" + unit + (key === "speed" ? "/s" : "") + ")", data[axis][key]);
      }
    }
  }
  if (group === "objectives") {
    fields += '<h3 class="group-title full">收益与代价权重</h3>';
    for (const [key, title] of Object.entries(WEIGHTS)) fields += control("objectives.weights." + key, title, data.weights[key]);
  }
  const revisionSelect = `<select id="config-revision" aria-label="已保存配置">${revisions.map(row => option(row.revision, row.name + " · " + row.revision, revision)).join("")}</select>`;
  const preview = card("设备几何预览", "cube", '<div class="viewport compact"><canvas id="machine" class="orbit" aria-label="当前配置的三维设备预览"></canvas></div>', btn("view-reset", "复位", "reset", "small"), '<div class="axis-values" id="axis-values"></div>');
  return heading("CONFIGURATION / 02", "设备配置", "单套 360 吨级参考设备，所有参数按来源分组保存。", btn("config-import", "导入配置", "upload") + btn("config-copy", "复制配置", "plus") + btn("config-save", "保存版本", "save", "primary")) +
    '<input type="file" id="config-file" hidden accept=".json,application/json">' +
    `<div class="cols wide-left"><div>
      <div class="config-tabs">${tab}</div>
      ${card(GROUPS[group] + "参数", "sliders", `<div class="card-body form-grid">${fields}</div>`, group === "camera" ? btn("calibrate", "录像标定", "target", "small") : pill(data.source, data.source === "仿真假设" ? "amber" : "blue"))}
    </div><div>
      ${card("当前配置", "report", `<div class="card-body stack-form"><label class="field-label">已保存版本${revisionSelect}</label>
      ${control("name", "配置名称", draft.name)}<div class="notice">版本 ${esc(revision)}<br>保存后的版本可在仿真实验中载入。参数改动后可在下方预览设备几何。</div><div class="actions">${btn("config-export", "下载配置", "download", "small")}</div></div>`)}
      <div class="space-top">${preview}</div>
      <div class="notice amber space-top">现场待验证。内腔、扒渣头、速度与标定数据均以各组所选来源为准。</div>
    </div></div>`;
}

function datasetRows(data) {
  return data.datasets.map(d => `<tr><td><strong>录像 ${esc(d.id)}</strong><div class="tiny muted">炉次 ${esc(d.furnace_id || "未知")}</div></td><td class="mono">${d.frames}</td><td class="mono">${d.aligned} / ${d.frames}</td><td>
    ${Object.entries(d.labels).map(([key, count]) => `${esc(LABEL[key] || key)} ${count}`).join(" · ")}
    <div class="tiny muted">带 group_id 的残渣轮廓 ${d.slag_track_annotations}</div>
    </td><td title="${esc(d.issues.join(" · "))}">${esc(d.issues[0] || "检查通过")}</td></tr>`);
}

export function dataPage(data, job) {
  const models = data.models;
  const current = models.find(m => m.version === data.active_model);
  const samples = data.datasets.reduce((n, d) => n + d.frames, 0);
  const existing = new Set(data.datasets.map(d => d.id));
  const candidates = data.videos.filter(v => !existing.has(v.id));
  const classRows = current?.validation.flatMap(v => Object.entries(v.classes).map(([key, info]) => `<tr><td>录像 ${esc(v.video_id)}</td><td>${LABEL[key]}</td>
      <td class="mono">${info.support.toLocaleString("zh-CN")}</td><td class="mono">${pct(info.iou, 2)}</td><td class="mono">${pct(info.recall, 2)}</td></tr>`)) || [];
  const latest = current?.validation[0], prior = current?.comparison?.validation[0];
  const comparisons = prior && latest ? ["slag", "iron", "tool"].map(key => {
    const before = prior.classes[key]?.iou, after = latest.classes[key]?.iou;
    const delta = (after - before) * 100;
    return `<tr><td><strong>${LABEL[key]}</strong></td><td class="mono">${pct(before, 2)}</td><td class="mono">${pct(after, 2)}</td><td class="mono ${delta >= 0 ? "value-teal" : "value-amber"}">${delta > 0 ? "+" : ""}${num(delta, 2)} 个百分点</td></tr>`;
  }) : [];
  const modelRows = models.map(m => `<tr><td><strong>${esc(m.name)}</strong><div class="tiny mono muted">${esc(m.version)}</div></td><td>${esc(m.training_ids.join("、"))}</td><td>${esc(m.validation_ids.join("、"))}</td><td class="mono">${m.training_frames}</td><td>${pill("使用中", "teal")}</td></tr>`);
  const task = job ? `<div class="notice ${job.status === "failed" ? "amber" : ""}">${["queued","running"].includes(job.status) ? '<span class="spinner"></span> ' : ""}${esc(job.message)}${job.result ? " · 版本 " + esc(job.result.version) : ""}</div>` : "";
  return heading("VISION DATA / 03", "数据与训练", "导入抽帧照片和 LabelMe 标注，按整段录像划分训练与验证。", btn("train-open", "新建训练任务", "play", "primary", job && ["running","queued"].includes(job.status) ? "disabled" : "")) +
    `<div class="metrics">
      ${metric("录像来源", data.videos.length, "段", "原始录像 / 原始帧率", "camera")}
      ${metric("人工标注抽帧", samples, "张", `${data.datasets.length} 组 LabelMe 标注`, "database")}
      ${metric("模型版本", models.length, "个", current ? "使用中 " + current.version : "等待训练", "pulse")}
      ${metric("残渣 IoU", pct(current?.validation[0]?.classes.slag?.iou, 1), "", "首个验证录像的已标注像素", "target", "value-teal")}
    </div>
    <div class="cols wide-left">
      ${card("标注检查", "database", data.datasets.length ? table(["来源", "抽帧数", "时间对齐", "人工标注类别", "待补项"], datasetRows(data)) : blank("等待导入标注"), pill("同一录像整体划分", "blue"))}
      ${card("本地文件导入", "upload", `<div class="card-body stack-form">
      <label class="field-label">原始录像<input type="file" id="video-upload" accept=".avi,.mp4,.mov,.mkv,.webm"></label>
      <div class="actions">${btn("upload-video", "导入录像", "upload", "small")}</div>
      <label class="field-label">标注对应录像<select id="label-video">${candidates.length ? candidates.map(v => option(v.id, v.name)).join("") : '<option value="">请先导入待标注录像</option>'}</select></label>
      <label class="field-label">炉次编号（可选）<input type="text" id="furnace-id" maxlength="60" placeholder="同炉次的录像使用同一编号"></label>
      <label class="field-label">图片与 LabelMe JSON 的 ZIP<input type="file" id="label-upload" accept=".zip"></label>
      <div class="actions">${btn("upload-labels", "检查并导入标注", "upload", "small", candidates.length ? "" : "disabled")}</div>
      </div>`)}
    </div>
    <div class="cols wide-left">
      ${card("识别模型", "report", `${task ? `<div class="card-body">${task}</div>` : ""}${modelRows.length ? table(["模型", "训练录像", "验证录像", "训练帧", "状态"], modelRows) : blank("等待训练")}`, pill("YOLO 语义分割"))}
      ${card("独立录像验证", "target", classRows.length ? table(["录像", "类别", "支持像素", "IoU", "召回"], classRows, true) : blank("暂无验证数据"), "", "依据人工标注像素计算。无支持样本的类别显示 —。")}
    </div>
    ${comparisons.length ? card("实际录像掩码对比", "target", table(["标注类别", "旧版回放 IoU", "当前回放 IoU", "变化"], comparisons), pill("同一独立录像", "blue"), "旧版和当前版均按同一批已标注预览像素计分，模糊像素可标为无法判断。", "section-gap") : ""}
    <div class="notice">已知炉次整体划分。回放的渣块编号由在线跟踪生成，跨帧实例准确率待独立验证。缺少罐口和蒸汽标注时，罐口由可见液面轮廓估计，蒸汽显示为疑似遮挡。视觉类别未经厚度实测校准。</div>`;
}

export function experimentPage(ctx) {
  const {boot, exp, revision, cfg, batch, batchJob} = ctx;
  const selected = boot.scenarios.find(s => s.id === exp.caseId);
  const cases = boot.scenarios.map(s => `<button class="case ${s.id === exp.caseId ? "active" : ""}" data-action="pick-case" data-id="${esc(s.id)}">
      <div class="case-top"><strong>${esc(s.name)}</strong>${pill(s.split === "development" ? "调试" : "验收", s.split === "development" ? "blue" : "ghost")}</div><p>${esc(s.description)}</p>
    </button>`).join("");
  const weights = Object.entries(cfg.objectives.weights).map(([key, value]) => `<label class="field-label">${WEIGHTS[key]}<input type="number" min="0" max="20" step="0.001" data-weight="${key}" value="${esc(value)}"></label>`).join("");
  const checks = Object.entries(boot.strategies).map(([key, title]) => `<label class="check"><input type="checkbox" name="batch-strategy" value="${key}" ${exp.batchStrategies.includes(key) ? "checked" : ""}>${esc(title)}</label>`).join("");
  const job = batchJob ? `<div class="notice ${batchJob.status === "failed" ? "amber" : ""}">${["queued","running"].includes(batchJob.status) ? '<span class="spinner"></span> ' : ""}${esc(batchJob.message)}</div>` : "";
  const counts = batch?.results?.length;
  return heading("SIMULATION LAB / 04", "仿真实验", "载入工况、调整优化权重，或对固定的六组验收工况批量比较。", btn("load-experiment", "载入实时作业", "arrow", "primary")) +
    `<div class="cols wide-left">
    ${card("工况库", "flask", `<div class="card-body case-grid">${cases}</div>`, pill("调试与验收分离", "blue"))}
    ${card("作业参数", "sliders", `<div class="card-body stack-form"><div class="form-grid">
      <label class="field-label">随机种子<input type="number" id="experiment-seed" min="0" max="4294967295" value="${esc(exp.seed)}"></label>
      <label class="field-label">规划策略<select id="experiment-strategy">${Object.entries(boot.strategies).map(([k,v]) => option(k,v,exp.strategy)).join("")}</select></label>
      </div><div class="notice">${esc(selected?.description)}<br>配置 ${esc(revision)} · 用时 ≤ ${cfg.objectives.time_limit_s} s · 去除率 > ${pct(cfg.objectives.removal_target, 0)}</div>
      <h3>优化权重</h3><div class="form-grid">${weights}</div><div class="actions">${btn("save-weights", "保存权重到新版本", "save", "small")}</div>
      </div>`)}
    </div>
    ${card("固定工况批量评测", "report", `<div class="card-body stack-form"><div class="actions">${checks}
      ${btn("batch-run", "运行所选策略", "play", "primary small", batchJob && ["queued","running"].includes(batchJob.status) ? "disabled" : "")}
      </div>${job}<p class="tiny muted">每个策略使用六个固定种子和同一配置版本。统计同时检查 120 s 和面积去除率 > 98%。</p></div>${batchTable(batch, boot.strategies)}`,
      counts ? pill(`${batch.results.filter(row => row.reference_passed).length} / ${counts} 达标`, "teal") : "", counts ? download("/api/batch/latest", "slag-acceptance.json") : "")}`;
}

export function archivesOptions(rows, selected) {
  return rows.map(r => option(r.id, r.case.name + " · " + r.id.slice(0, 7) + " · " + seconds(r.elapsed_s), selected)).join("");
}

export function replayPage(ctx) {
  const trace = ctx.replay.trace;
  return heading("PROCESS TRACE / 05", "过程回放", "沿时间轴对照识别帧、候选动作、执行轨迹和实际面积变化。",
    trace ? download("/api/runs/" + trace.summary.id + "/trace", "slag-trace-" + trace.summary.id.slice(0, 7) + ".json", "导出过程") : "") +
    `<div class="card section-gap"><div class="card-body stack-form"><div class="scrubber">
      <label class="field-label">作业归档<select id="replay-run"><option value="">选择作业</option>${archivesOptions(ctx.archives, trace?.summary.id)}</select></label>
      ${trace ? `<div class="right tiny muted">配置 ${esc(trace.summary.config_revision)} · 种子 ${trace.summary.seed}</div>` : ""}
      </div>
      ${trace ? `<div class="scrubber">${btn("replay-toggle", "播放", "play", "primary small", 'id="replay-toggle"')}${btn("replay-rewind", "回到起点", "reset", "small")}<input aria-label="回放时间" id="replay-time" type="range" min="0" max="${trace.summary.elapsed_s}" value="${ctx.replay.time}" step="0.01"><span id="replay-clock" class="mono time-text"></span><select id="replay-speed" aria-label="回放速度">${[1,2,4,8].map(n => option(n, n + "×", ctx.replay.speed)).join("")}</select></div>` : ""}
    </div></div>` +
    (trace ? pairViews({replay:true}) + `<div class="cols wide-left">
      ${card("作业时间线", "history", `<div id="trace-events">${trace.events.length ? table(["序号", "时间 s", "阶段", "候选选择", "真实去除"], trace.events.map(e => `<tr data-event="${e.index}">
      <td>${btn("replay-event", String(e.index).padStart(2,"0"), null, "small flat", `data-index="${e.index}"`)}</td><td class="mono">${num(e.start_s)}</td><td>${({motion:"执行",hold:"保持",operator:"干预"})[e.kind]}</td>
      <td>${esc(e.action ? e.action.target + " · " + e.action.label : e.note || "等待")}</td><td class="mono">${e.actual ? num(e.actual.removed_m2, 4) + " m²" : "—"}</td></tr>`), true) : blank("作业已归档首帧，等待执行事件")}</div>`)}
      ${card("当前事件", "motion", '<div id="event-detail" class="card-body"></div>', '<span id="event-badge"></span>')}
      </div>` : `<div class="card">${blank("开始一次仿真或单步执行后，画面和轨迹会自动写入作业归档。")}</div>`);
}

export function reportPage(ctx) {
  const r = ctx.singleReport;
  const count = ctx.batch?.results?.length || 0;
  return heading("ACCEPTANCE / 06", "结果报告", "根据仿真场景真值面积核算去除率，逐项记录未达标原因。", r ?
    download("/api/runs/" + r.id + "/report", "slag-run-" + r.id.slice(0, 7) + ".json", "导出作业报告") : "") +
    `<div class="card section-gap"><div class="card-body scrubber">
      <label class="field-label">单次作业<select id="report-run"><option value="">仅查看批量评测</option>${archivesOptions(ctx.archives, r?.id)}</select></label>
      <span class="right tiny muted">固定验收线 ≤ 120 s · 面积去除率严格大于 98%</span></div></div>` +
    (r ? `<div class="section-gap">${reportBody(r)}</div>` : "") +
    card("固定工况评测结果", "report", batchTable(ctx.batch, ctx.boot.strategies),
      count ? pill(`${ctx.batch.results.filter(row => row.reference_passed).length} / ${count} 达标`, "teal") : "",
      count ? `<span>配置 ${esc(ctx.batch.config_revision)} · 现场待验证</span>${download("/api/batch/latest", "slag-acceptance.json", "导出批量报告")}` : "");
}

export function drawConfigAxis(state) {
  const footer = document.querySelector("#axis-values");
  if (footer) footer.innerHTML = axisValues(state);
}

export function trainForm(data) {
  const validation = data.models.find(model => model.version === data.active_model)?.validation_ids ?? [data.datasets[0]?.id];
  const items = data.datasets.map(row => `<label class="check"><input type="checkbox" name="val-set" value="${esc(row.id)}" ${validation.includes(row.id) ? "checked" : ""}>录像 ${esc(row.id)} · ${row.frames} 张抽帧</label>`).join("");
  return `<div class="stack-form"><p>勾选的完整录像用于独立验证，其余标注组用于训练。已知同炉次录像需一起划分。训练和验证完成后替换当前 YOLO 模型。</p><div class="actions">${items}</div>
    <label class="field-label">训练随机种子<input type="number" id="train-seed" min="0" max="4294967295" value="360"></label>
    <div class="actions end">${btn("train-submit", "开始训练", "play", "primary", data.datasets.length < 2 ? "disabled" : "")}</div></div>`;
}

export function calibrationForm(data, tilt) {
  return `<div class="stack-form"><p>填入原始分辨率下的像素点和对应的罐体液面坐标。至少四组不共线对应点，液面坐标单位为米。</p>
    <div class="form-grid"><label class="field-label">录像<select id="cal-video">${data.videos.map(v => option(v.id, v.name)).join("")}</select></label>
    <label class="field-label">对应倾角 (°)<input type="number" id="cal-tilt" step=".1" value="${tilt}"></label></div>
    <label class="field-label">每行 u, v, X, Y<textarea id="cal-points" class="mono" rows="5" placeholder="像素 u, 像素 v, 液面 X, 液面 Y"></textarea></label>
    <p class="muted tiny">使用当前保存版本的相机内参与畸变参数。求解后填回配置，可在设备配置中保存新版本。</p>
    <div class="actions end">${btn("calibrate-solve", "求解平面变换", "target", "primary")}</div></div>`;
}

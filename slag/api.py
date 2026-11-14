"""Local HTTP API for the workbench, replay, configuration and dataset jobs."""

import asyncio
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from math import floor
from pathlib import Path
from threading import Lock
from time import perf_counter, time
from typing import Literal
from uuid import uuid4
from zipfile import BadZipFile

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from .benchmark import evaluate
from .config import ROOT, RUNTIME, Config, Strict, Vec2, default_config, load_config, save_config, scenarios, write_json
from .engine import Run
from .motion import AXES, State
from .planner import STRATEGIES
from .recording import Replay, calibrate, plan_recording
from .training import active_model, furnace_for, inspect_dataset, label_files, load_model, model_versions, train, video_files, video_info

app = FastAPI(title="熔线 · 铁水罐扒渣路径规划平台", version="1.2.0")
RUNS = RUNTIME / "runs"
Strategy = Literal["balanced", "coverage", "gentle"]
Fault = Literal["offline", "stale", "steam", "low"]
VIDEO_PLAN_INTERVAL_S = 1.


class RunSpec(Strict):
    case_id: str = "commissioning"
    seed: int | None = Field(default=None, ge=0, le=4294967295)
    strategy: Strategy = "balanced"
    revision: str | None = None


class FaultSpec(Strict):
    fault: Fault | None


class StepSpec(Strict):
    expected_index: int = Field(ge=0)


class BatchSpec(Strict):
    strategies: list[Strategy] = Field(min_length=1, max_length=3)
    revision: str | None = None


class TrainSpec(Strict):
    validation_ids: list[str] = Field(min_length=1, max_length=20)
    seed: int = Field(default=360, ge=0, le=4294967295)


class CalibrationSpec(Strict):
    video_id: str
    revision: str | None = None
    pixel: list[Vec2] = Field(min_length=4, max_length=40)
    plane_m: list[Vec2] = Field(min_length=4, max_length=40)
    tilt_deg: float = Field(ge=-20, le=30)


class Joints(Strict):
    extension_m: float
    lift_m: float
    swing_deg: float
    pitch_deg: float


class DeviceTelemetry(Strict):
    timestamp_s: float
    revision: str | None = None
    joints: Joints
    tilt_deg: float


@dataclass
class Handle:
    run: Run
    lock: Lock = field(default_factory=Lock)


handles = {}
handles_lock = Lock()
jobs = {}
jobs_lock = Lock()
worker = ThreadPoolExecutor(max_workers=2, thread_name_prefix="slag-job")
telemetry_lock = Lock()
label_publish_lock = Lock()
telemetry: DeviceTelemetry | None = None


def get_cfg(revision):
    try:
        return load_config(revision)
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "配置版本不存在或格式无效") from exc


def get_video(video_id):
    path = video_files().get(video_id)
    if path is None:
        raise HTTPException(404, "录像不存在")
    return path


def get_handle(run_id):
    with handles_lock:
        h = handles.get(run_id)
    if h is None:
        raise HTTPException(404, "运行会话已结束或服务已重启，可在过程回放打开归档")
    return h


def run_directory(run_id):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise HTTPException(404, "作业不存在")
    return RUNS / run_id


def _start_job(kind, target):
    key = uuid4().hex
    with jobs_lock:
        if any(job["kind"] == kind and job["status"] in ("queued", "running") for job in jobs.values()):
            raise HTTPException(409, "同类任务正在执行")
        jobs[key] = {"id": key, "kind": kind, "status": "queued", "message": "等待执行", "result": None}

    def publish(message):
        with jobs_lock:
            jobs[key].update(status="running", message=message)

    def execute():
        publish("开始执行")
        try:
            result = target(publish)
        except Exception as exc:
            with jobs_lock:
                jobs[key].update(status="failed", message=str(exc) or "任务执行失败")
        else:
            with jobs_lock:
                jobs[key].update(status="done", message="完成", result=result)

    worker.submit(execute)
    return {"id": key}


@app.get("/api/bootstrap")
def bootstrap():
    cfg = default_config()
    return {"default_revision": cfg.revision, "config": cfg, "scenarios": list(scenarios().values()),
            "strategies": STRATEGIES, "active_model": active_model(), "field_status": "现场待验证"}


@app.get("/api/configs")
def configs():
    cfg = default_config()
    items = [{"revision": cfg.revision, "name": cfg.name, "built_in": True}]
    for path in sorted((RUNTIME / "configs").glob("*.json")):
        saved = load_config(path.stem)
        if saved.revision != cfg.revision:
            items.append({"revision": saved.revision, "name": saved.name, "built_in": False})
    return items


@app.get("/api/configs/{revision}")
def config_detail(revision: str):
    return get_cfg(revision)


@app.post("/api/configs")
def config_create(cfg: Config):
    return {"revision": save_config(cfg), "config": cfg}


@app.post("/api/calibration/solve")
def solve(spec: CalibrationSpec):
    cfg = get_cfg(spec.revision)
    try:
        result = calibrate(cfg, video_info(get_video(spec.video_id)), spec.pixel, spec.plane_m, spec.tilt_deg)
    except (ValueError, cv2.error) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"calibration": result, "ready_for_review": result.rms_m <= .10}


@app.get("/api/field/status")
def field_status():
    with telemetry_lock:
        sample = telemetry
    if sample is None:
        return {"status": "现场待验证", "reason": "尚未收到设备状态", "state": None, "controller": "仅仿真执行器"}
    cfg = get_cfg(sample.revision)
    age = time() - sample.timestamp_s
    state = State(tuple(getattr(sample.joints, key) for key in AXES), sample.tilt_deg)
    return {
        "status": "现场待验证", "controller": "仅仿真执行器", "state": state.view(cfg),
        "age_s": age, "reason": "设备状态过期" if age >= cfg.camera.max_frame_age_s else "已收到状态样本，等待现场联调",
    }


@app.post("/api/field/status")
def field_sample(sample: DeviceTelemetry):
    global telemetry
    cfg = get_cfg(sample.revision)
    now = time()
    if not -.05 <= now - sample.timestamp_s < cfg.camera.max_frame_age_s:
        raise HTTPException(422, "设备状态时间戳过期或超前")
    if any(not a.min <= getattr(sample.joints, key) <= a.max for key, a in zip(AXES, cfg.arm.axes)):
        raise HTTPException(422, "设备状态超出轴行程")
    if not cfg.tilt.min_deg <= sample.tilt_deg <= cfg.tilt.max_deg:
        raise HTTPException(422, "设备状态倾角超范围")
    with telemetry_lock:
        if telemetry and sample.timestamp_s <= telemetry.timestamp_s:
            raise HTTPException(409, "设备状态时间戳未递增")
        telemetry = sample
    return field_status()


@app.post("/api/runs", status_code=201)
def new_run(spec: RunSpec):
    cfg = get_cfg(spec.revision)
    case = scenarios().get(spec.case_id)
    if not case:
        raise HTTPException(422, "工况不存在")
    run = Run(cfg, case, case.seed if spec.seed is None else spec.seed, spec.strategy, RUNS)
    with handles_lock:
        handles[run.id] = Handle(run)
    return run.view()


@app.get("/api/runs")
def list_runs():
    entries = sorted(RUNS.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]
    return [json.loads(path.read_text(encoding="utf-8")) for path in entries]


@app.get("/api/runs/{run_id}")
def view_run(run_id: str):
    h = get_handle(run_id)
    with h.lock:
        return h.run.view()


@app.post("/api/runs/{run_id}/step")
def advance(run_id: str, spec: StepSpec):
    h = get_handle(run_id)
    with h.lock:
        if spec.expected_index != len(h.run.records):
            raise HTTPException(409, "作业步骤已经更新，请刷新会话后继续")
        event = h.run.advance()
        return {"run": h.run.view(), "event": event}


def _intervene(run_id, **options):
    h = get_handle(run_id)
    with h.lock:
        try:
            event = h.run.intervene(**options)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run": h.run.view(), "event": event}


@app.post("/api/runs/{run_id}/disturb")
def disturb(run_id: str):
    return _intervene(run_id, disturb=True)


@app.post("/api/runs/{run_id}/fault")
def fault(run_id: str, spec: FaultSpec):
    return _intervene(run_id, fault=spec.fault)


@app.get("/api/runs/{run_id}/trace")
def run_trace(run_id: str):
    path = run_directory(run_id) / "trace.json"
    if not path.is_file():
        raise HTTPException(404, "作业过程不存在")
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str):
    path = run_directory(run_id) / "summary.json"
    if not path.is_file():
        raise HTTPException(404, "作业报告不存在")
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/runs/{run_id}/frames/{name}")
def run_frame(run_id: str, name: str):
    if not re.fullmatch(r"\d{5,9}\.jpg", name):
        raise HTTPException(404, "画面不存在")
    path = run_directory(run_id) / "frames" / name
    if not path.is_file():
        raise HTTPException(404, "画面不存在")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400, immutable"})


@app.post("/api/batch", status_code=202)
def batch(spec: BatchSpec):
    cfg = get_cfg(spec.revision)
    strategies = list(dict.fromkeys(spec.strategies))

    def target(notify):
        report = evaluate(cfg, strategies, notify)
        write_json(RUNTIME / "reports/latest.json", report)
        return {"rows": len(report["results"]), "revision": cfg.revision}

    return _start_job("batch", target)


@app.get("/api/batch/latest")
def batch_latest():
    path = RUNTIME / "reports/latest.json"
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("surface_model") != "continuous-film":
        path = ROOT / "reports/acceptance.json"
    if not path.is_file():
        raise HTTPException(404, "尚未生成批量评测")
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return job.copy()


@lru_cache(maxsize=128)
def _video_metadata(path, mtime_ns):
    return video_info(path)


@lru_cache(maxsize=128)
def _label_metadata(path, mtime_ns, video):
    return inspect_dataset(path, dict(video))


@app.get("/api/data")
def catalog():
    videos = [_video_metadata(path, path.stat().st_mtime_ns) for path in sorted(video_files().values())]
    by_id = {v["id"]: v for v in videos}
    datasets = []
    for key, path in sorted(label_files().items()):
        info = by_id.get(key, {})
        align = tuple((key, info[key]) for key in ("fps", "frames") if key in info)
        datasets.append(_label_metadata(path, path.stat().st_mtime_ns, align) | {"furnace_id": furnace_for(path)})
    models = model_versions()
    return {"videos": videos, "datasets": datasets, "models": models,
            "active_model": models[0]["version"] if models else None}


def _copy_upload(file, path, max_bytes):
    total = 0
    with path.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(413, "上传文件超过体积上限")
            out.write(chunk)


@app.post("/api/data/videos", status_code=201)
def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".avi", ".mp4", ".mov", ".mkv", ".webm"):
        raise HTTPException(422, "请选择 AVI、MP4、MOV、MKV 或 WebM 录像")
    folder = RUNTIME / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    ident = uuid4().hex[:12]
    temporary = folder / f".upload-{ident}{ext}"
    try:
        _copy_upload(file, temporary, 1_100_000_000)
        info = video_info(temporary)
        name = f"{ident}{ext}"
        temporary.replace(folder / name)
        return info | {"id": ident, "name": name}
    except (ValueError, cv2.error) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)
        file.file.close()


@app.post("/api/data/labels", status_code=201)
def upload_labels(video_id: str = Form(...), furnace_id: str | None = Form(None, max_length=60), file: UploadFile = File(...)):
    vid = video_info(get_video(video_id))
    if video_id in label_files():
        raise HTTPException(409, "该录像已有标注组，请保留原始版本并导入另一段录像")
    folder = RUNTIME / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".upload-{uuid4().hex}.tmp"
    destination = folder / f"{video_id}.zip"
    try:
        _copy_upload(file, temporary, 500_000_000)
        result = inspect_dataset(temporary, vid, check_images=True)
        group = furnace_id.strip() if furnace_id and furnace_id.strip() else None
        with label_publish_lock:
            if video_id in label_files():
                raise HTTPException(409, "该录像已有标注组")
            write_json(destination.with_suffix(".meta.json"), {"furnace_id": group})
            temporary.replace(destination)
        return result | {"id": video_id, "name": destination.name, "furnace_id": group}
    except (BadZipFile, ValueError, KeyError, cv2.error) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)
        file.file.close()


@app.post("/api/data/train", status_code=202)
def train_model(spec: TrainSpec):
    datasets = label_files()
    if not set(spec.validation_ids) < datasets.keys():
        raise HTTPException(422, "请选择独立验证录像并保留至少一组训练数据")
    return _start_job("training", lambda notify: train(spec.validation_ids, spec.seed, notify=notify))


@app.get("/api/data/videos/{video_id}/stream")
async def stream_video(video_id: str, start_s: float = Query(0, ge=0, le=1000000),
                       clip_start_s: float = Query(0, ge=0, le=1000000),
                       end_s: float | None = Query(None, gt=0, le=1000000),
                       analyze: bool = False, strategy: Strategy = "balanced",
                       revision: str | None = None, version: str | None = None,
                       limit: int | None = Query(None, ge=1, le=2000)):
    path = get_video(video_id)
    cfg = get_cfg(revision)
    try:
        def open_replay():
            info = _video_metadata(path, path.stat().st_mtime_ns)
            return Replay(path, cfg, load_model(version), start_s, clip_start_s=clip_start_s,
                          end_s=end_s, analyze=analyze, info=info)
        replay = await run_in_threadpool(open_replay)
    except (ValueError, cv2.error) as exc:
        raise HTTPException(422, str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise HTTPException(503, f"YOLO 模型暂不可用 · {exc}") from exc

    async def frames():
        started = first_stamp = first_index = None
        planned_segment = previous_segment = None
        sent = 0
        skipped = 0
        try:
            while limit is None or sent < limit:
                if limit is None and started is not None:
                    due = first_index + int((perf_counter() - started) * replay.fps)
                    if due > replay.index + 1:
                        skipped += await run_in_threadpool(replay.catch_up, due)
                frame_started = perf_counter()
                sample = await run_in_threadpool(replay.read)
                if sample is None:
                    yield json.dumps({"kind": "eof", "reason": "片段播放结束" if end_s is not None else "录像已结束，执行器保持"}, ensure_ascii=False) + "\n"
                    break
                ts = sample.observation.captured_s
                if first_stamp is None:
                    first_stamp = ts
                    first_index = sample.observation.frame
                    started = frame_started
                if analyze:
                    segment = max(0, floor((ts - replay.clip_start_s + 1e-7) / VIDEO_PLAN_INTERVAL_S))
                    segment_end = replay.clip_start_s + (segment + 1) * VIDEO_PLAN_INTERVAL_S
                    # A seek starts a fresh planning interval, bounded by the selected action deadline.
                    action_deadline = min(replay.end_s, ts + VIDEO_PLAN_INTERVAL_S if limit == 1 else segment_end)
                    sample.details["segment"] = segment
                    if previous_segment is not None and segment > previous_segment + 1:
                        sample.details["skipped_segments"] = [previous_segment + 1, segment]
                    previous_segment = segment
                    if sample.planning_surface is None:
                        planned_segment = None
                    elif segment != planned_segment:
                        result = await run_in_threadpool(plan_recording, sample, cfg, strategy,
                                                        frame_started=frame_started,
                                                        source_started_s=max(ts, first_stamp + frame_started - started),
                                                        deadline_s=action_deadline)
                        sample.details["plan"] = result
                        planned_segment = segment
                deadline = started + ts - first_stamp
                if deadline > perf_counter():
                    await asyncio.sleep(deadline - perf_counter())
                ok, encoded = cv2.imencode(".jpg", sample.image, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok:
                    raise ValueError("录像预览编码失败")
                frame = {
                    "kind": "frame", "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "observation": sample.observation.view(), "meta": sample.details,
                    "fps": replay.fps, "lag_s": max(0., perf_counter() - deadline), "skipped": skipped,
                }
                yield json.dumps(frame, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                sent += 1
        except (ValueError, RuntimeError, cv2.error) as exc:
            yield json.dumps({"kind": "error", "reason": str(exc)}, ensure_ascii=False) + "\n"
        finally:
            replay.close()

    return StreamingResponse(frames(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/")
def home():
    return FileResponse(ROOT / "static/index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

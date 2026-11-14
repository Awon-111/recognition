"""Original-rate camera replay and optional image-to-plane calibration."""

from dataclasses import dataclass, replace
from math import ceil, floor, isfinite
from time import perf_counter

import cv2
import numpy as np

from .config import PlaneCalibration
from .motion import State, bath_frame, initial_state
from .perception import Observation, Tracker, mask_regions, patch_components, polygon_points
from .planner import plan
from .surface import MAX_VALUE, UNKNOWN as UNKNOWN_PIXEL, SurfaceField, project_field
from .training import preview_gray, video_info
from .vision import IRON, MOUTH, SLAG, STEAM, TOOL, UNKNOWN


def _intrinsics(cfg, size):
    w, h = size
    src_w, src_h = cfg.camera.resolution
    fx, fy, cx, cy = cfg.camera.intrinsics
    return np.array([[fx * w / src_w, 0, cx * w / src_w],
                     [0, fy * h / src_h, cy * h / src_h], [0, 0, 1.]])


def calibrate(cfg, info, pixel, plane, tilt):
    if not 4 <= len(pixel) == len(plane) <= 40:
        raise ValueError("至少提供四对同序的图像像素与设备液面坐标")
    src, dst = np.float32(pixel), np.float32(plane)
    size = tuple(info["size"])
    if np.any(src < 0) or np.any(src >= size):
        raise ValueError("像素对应点须位于原始图像范围")
    if np.any(np.abs(dst) > 20):
        raise ValueError("设备液面坐标须以米填写")
    if cfg.camera.distortion != (0.,) * 5:
        k = _intrinsics(cfg, size)
        src = cv2.undistortPoints(src.reshape(-1, 1, 2), k, np.asarray(cfg.camera.distortion), P=k).reshape(-1, 2)
    if cv2.contourArea(cv2.convexHull(src)) < 100 or cv2.contourArea(cv2.convexHull(dst)) < .02:
        raise ValueError("标定点分布范围过小或共线")
    matrix, _ = cv2.findHomography(src, dst)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise ValueError("不能由该组对应点建立有效平面变换")
    result = cv2.perspectiveTransform(src[None, ...], matrix)[0]
    rms = float(np.sqrt(np.mean(np.sum((result - dst) ** 2, axis=1))))
    return PlaneCalibration(video_id=info["id"], image_size=size, image_to_plane=matrix.tolist(), rms_m=rms, tilt_deg=tilt)


def apparent_plane(region, radii):
    x, y, w, h = cv2.boundingRect(np.uint8(region))
    if w < region.shape[1] * .22 or h < region.shape[0] * .19 or region.sum() < region.size * .055:
        return None
    center_x, center_y = x + (w - 1) / 2, y + (h - 1) / 2
    sx, sy = 2 * radii[0] / w, 2 * radii[1] / h
    return np.array([[sx, 0., -sx * center_x], [0., -sy, sy * center_y], [0., 0., 1.]])


def video_surface(gray, labels, mapping, radii):
    pixels = np.full(gray.shape, UNKNOWN_PIXEL, dtype=np.uint8)
    slag = labels == SLAG
    pixels[labels == IRON] = 0
    # This is a display tone from appearance. Camera exposure and temperature also affect it.
    tone = .25 + .65 * (1 - gray[slag].astype(np.float32) / 255)
    pixels[slag] = np.rint(tone * MAX_VALUE).astype(np.uint8)
    return project_field(pixels, mapping, radii, .13, "appearance")


def enhance_frame(frame):
    """Raise local contrast while keeping the original exposure and pixel geometry."""
    h, w = frame.shape[:2]
    scale = min(1., 640 / max(w, h))
    if scale < 1.:
        frame = cv2.resize(frame, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    contrast = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(light)
    lab[:, :, 0] = cv2.addWeighted(light, .70, contrast, .30, 0.)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def recorded_observation(frame, cfg, model, tracker, index, stamp, video_id, analyze=False):
    begin = perf_counter()
    calibration = cfg.camera.video_plane
    size = frame.shape[1], frame.shape[0]
    mapped = bool(calibration and calibration.video_id == video_id and calibration.image_size == size and calibration.rms_m <= .10)
    if mapped and cfg.camera.distortion != (0.,) * 5:
        k = _intrinsics(cfg, size)
        frame = cv2.undistort(frame, k, np.asarray(cfg.camera.distortion))
    preview, gray = preview_gray(frame)
    h, w = gray.shape
    matrix = np.array(calibration.image_to_plane) @ np.diag((size[0] / w, size[1] / h, 1)) if mapped else None
    infer_image = enhance_frame(frame) if analyze or model.manifest.get("input_enhancement") == "CLAHE" else frame
    if analyze:
        preview, _ = preview_gray(infer_image)
    infer_at = perf_counter()
    predicted = model.segment(infer_image, gray)
    inferred = perf_counter()
    region, labels, certainty = predicted.region, predicted.labels, predicted.confidence
    available = model.manifest["classes"]
    slag, iron, tool = labels == SLAG, labels == IRON, labels == TOOL
    steam = labels == STEAM
    unknown = (labels == UNKNOWN) & region
    if np.any(region):
        overall_confidence = float(certainty[region].mean())
    else:
        overall_confidence = 0.
    tool_shapes = polygon_points(tool, 60, 1.35, 1)
    tilt = calibration.tilt_deg if mapped else cfg.tilt.working_deg
    if tilt is None:
        tilt = cfg.tilt.initial_deg
    radii = cfg.tank.inner_radii_m
    appearance_mapping = matrix if mapped else apparent_plane(region, radii)
    surface = video_surface(gray, labels, appearance_mapping, radii)
    tip = predicted.arm_tip if tool_shapes else None
    head = None
    if appearance_mapping is not None and tip is not None:
        xy = cv2.perspectiveTransform(np.float32([[tip]]), appearance_mapping)[0, 0]
        if np.all(np.isfinite(xy)) and np.linalg.norm(xy / radii) < 1.25:
            head = xy.tolist()
    obs = Observation(
        index, stamp, (w, h), mouth=polygon_points(labels == MOUTH if available["mouth"] else region, 90, 2, 2),
        liquid=polygon_points(region, 140, 1.8, 1), tool=tool_shapes,
        slag_regions=mask_regions(slag, 14, 1.25), iron_regions=mask_regions(iron, 14, 1.25),
        tool_center_px=tip, surface=surface,
        steam=polygon_points(steam, 42, 1.8, 8), unknown=polygon_points(unknown, 70, 1.8, 8),
        steam_fraction=float(np.count_nonzero(steam) / max(1, np.count_nonzero(region))),
        confidence=overall_confidence, source="原始录像 · " + model.manifest["name"],
    )
    obs.slag = tracker.update(
        patch_components(slag, gray, matrix, certainty, minimum=22, brightness=101), stamp
    )
    if obs.steam_fraction >= cfg.camera.steam_limit:
        obs.reason = "疑似蒸汽或曝光遮挡过重"
    elif obs.confidence < cfg.camera.min_confidence:
        obs.reason = "识别可信度不足"
    elif not mapped:
        obs.reason = "录像尚未完成有效平面标定"
    else:
        obs.reason = "录像回放 · 仿真执行器保持"
    # Recorded frames do not provide a closed-loop response to a submitted machine command.
    reconstruction = {
        "basis": ("calibrated" if mapped else "relative") if surface is not None else "unavailable",
        "tilt_deg": tilt, "bath": bath_frame(cfg.tank, tilt).view(), "tool_xy": head,
    }
    details = {"calibrated": mapped, "mouth_estimated": not available["mouth"],
                          "steam_estimated": not available["steam"], "model_version": model.manifest["version"],
                          "model_name": model.manifest["name"],
                          "reconstruction": reconstruction}
    sample = Sample(preview, obs, details)
    if analyze:
        details["enhancement"] = "CLAHE"
        if obs.steam_fraction >= cfg.camera.steam_limit or obs.confidence < cfg.camera.min_confidence:
            details["planning_reason"] = obs.reason
        elif surface is None or surface.mean is None:
            details["planning_reason"] = "未获得可用于规划的液面投影"
        else:
            semantic = np.full_like(gray, UNKNOWN_PIXEL)
            semantic[slag] = MAX_VALUE
            semantic[iron] = 0
            sample.planning_surface = project_field(semantic, appearance_mapping, radii, .5, "semantic")
        details["timing_ms"] = {
            "preprocess": (infer_at - begin) * 1000,
            "yolo": (inferred - infer_at) * 1000,
            "projection": (perf_counter() - inferred) * 1000,
        }
    return sample


@dataclass
class Sample:
    image: np.ndarray
    observation: Observation
    details: dict
    planning_surface: SurfaceField | None = None


def plan_recording(sample, cfg, strategy, *, frame_started, source_started_s, deadline_s):
    """Plan against recorded segmentation in the configured reference machine."""
    planning_started = perf_counter()
    obs = replace(sample.observation, usable=True, surface=sample.planning_surface)
    tilt = sample.details["reconstruction"]["tilt_deg"]
    if not cfg.tilt.min_deg <= tilt <= cfg.tilt.max_deg:
        candidates, selected, reason = [], None, "录像标定倾角超出配置的设备范围"
    else:
        state = State(initial_state(cfg).joints, tilt)
        budget = cfg.objectives.time_limit_s - cfg.objectives.planning_budget_s
        candidates, selected, reason = plan(obs, state, cfg, budget, 0., strategy)
    views = [candidate.view(cfg) for candidate in candidates]
    complete = perf_counter()
    ready_at = source_started_s + complete - frame_started
    timely = ready_at < deadline_s
    return {
        "frame": obs.frame, "time_s": obs.captured_s,
        "selected_id": selected.id if selected and timely else None,
        "reason": reason if timely else "超过本段规划截止时间",
        "candidates": views, "timely": timely, "deadline_s": deadline_s, "ready_at_s": ready_at,
        "total_ms": (complete - frame_started) * 1000, "planning_ms": (complete - planning_started) * 1000,
    }


class Replay:
    def __init__(self, path, cfg, model, start_s, *, clip_start_s=0., end_s=None, analyze=False, info=None):
        self.info = info if info is not None else video_info(path)
        self.cfg, self.model = cfg, model
        self.fps = self.info["fps"]
        end_s = self.info["duration_s"] if end_s is None else end_s
        first = ceil(clip_start_s * self.fps - 1e-7)
        self.stop_index = min(self.info["frames"], ceil(end_s * self.fps - 1e-7))
        if end_s <= clip_start_s or end_s > self.info["duration_s"] + 1e-7 or first >= self.stop_index:
            raise ValueError("片段起止时间超出录像范围或未包含完整采样帧")
        self.index = min(self.stop_index - 1, max(first, floor(start_s * self.fps + 1e-7)))
        self.clip_start_s, self.end_s = clip_start_s, end_s
        self.analyze = analyze
        self.cap = cv2.VideoCapture(str(path))
        self.tracker = Tracker(ttl_s=1.5)
        self.last_stamp = None
        ok, first = self.cap.read()
        if not ok:
            self.close()
            raise ValueError("无法读取录像首帧")
        self.pts_property = getattr(cv2, "CAP_PROP_PTS", None)
        self.origin_pts = self.cap.get(self.pts_property) if self.pts_property is not None else 0.
        self.pending = first if self.index == 0 else None
        if self.index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.index)

    def read(self):
        if self.index >= self.stop_index:
            return None
        if self.pending is not None:
            image, self.pending = self.pending, None
        else:
            ok, image = self.cap.read()
            if not ok:
                return None
        pts = self.cap.get(self.pts_property) if self.pts_property is not None else float("nan")
        stamp = (pts - self.origin_pts) / self.fps
        if not isfinite(stamp) or stamp < 0 or (self.last_stamp is not None and stamp <= self.last_stamp):
            stamp = max(self.index / self.fps, (self.last_stamp or 0) + (1 / self.fps if self.last_stamp is not None else 0.))
        self.last_stamp = stamp
        sample = recorded_observation(
            image, self.cfg, self.model, self.tracker, self.index, stamp, self.info["id"], self.analyze
        )
        self.index += 1
        return sample

    def catch_up(self, requested_index):
        """Decode past stale source frames without doing segmentation on them."""
        goal = min(self.stop_index - 1, requested_index)
        before = self.index
        if self.pending is not None and self.index < goal:
            self.pending = None
            self.index += 1
        while self.index < goal and self.cap.grab():
            self.index += 1
        return self.index - before

    def close(self):
        self.cap.release()

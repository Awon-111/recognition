"""One-motion-at-a-time coordinator with independent acceptance accounting."""

from collections import Counter
from math import floor
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

import cv2

from .config import write_json
from .motion import Motion, State, hard_reasons
from .perception import Tracker, synthetic_observation, synthetic_surface
from .physics import World
from .planner import plan, tilt_candidate


class Controller(Protocol):
    """A site adapter implements this contract after integration and validation."""

    def read_state(self) -> State: ...

    def submit(self, motion: Motion, capture=None) -> None: ...

    def hold(self, reason: str) -> None: ...


class SimController:
    def __init__(self, world):
        self.world = world
        self.held_reason = ""

    def read_state(self):
        return self.world.state

    def submit(self, motion, capture=None):
        self.held_reason = ""
        self.world.execute(motion, capture)

    def hold(self, reason):
        self.held_reason = reason


class Run:
    def __init__(self, cfg, case, seed, strategy, output_root: Path | None = None):
        self.id = uuid4().hex
        self.cfg = cfg
        self.case = case
        self.seed = seed
        self.strategy = strategy
        self.directory = output_root / self.id if output_root else None
        self.world = World(cfg, case, seed)
        self.controller = SimController(self.world)
        self.tracker = Tracker()
        self.initial_audit = self.world.audit()
        self.elapsed = 0.
        self.timings = dict(recognition=0., planning=0., motion=0., feedback=0., holding=0.)
        self.processing_wall = dict(recognition=0., planning=0.)
        self.actions = 0
        self.predicted_iron_used = 0.
        self.holds = 0
        self.no_path = 0
        self.empty = 0
        self.records = []
        self.status = "ready"
        self.reason = ""
        self.fault = None
        self.selected = None
        self.candidates = []
        self.tilt_done = False
        self.shift_done = False
        self.frame_counter = 0
        self.archive_issue = ""
        self.hold_counts = Counter()
        self._observe()
        self.initial = self.current
        self.initial_observed = self.observation.area
        self.initial_surface_mean = self.observation.surface.mean if self.observation.surface else None
        self._prepare()
        self._save()

    @property
    def terminal(self):
        return self.status in ("passed", "failed")

    def _time(self, bucket, seconds, advance=True):
        self.elapsed += seconds
        self.timings[bucket] += seconds
        if advance:
            self.world.wait(seconds)

    def _frame(self, image):
        if not self.directory:
            return None
        self.frame_counter += 1
        frames = self.directory / "frames"
        name = f"{self.frame_counter:05d}.jpg"
        try:
            frames.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise OSError("JPEG 编码失败")
            (frames / name).write_bytes(encoded.tobytes())
        except (OSError, cv2.error):
            self.archive_issue = "部分相机帧归档失败，请检查运行目录和可用磁盘空间"
            return None
        return f"/api/runs/{self.id}/frames/{name}"

    def _next_frame(self, bucket="feedback", minimum=0.):
        fps = self.cfg.camera.fps
        next_frame = (floor((self.elapsed + minimum) * fps + 1e-8) + 1) / fps
        self._time(bucket, max(0., next_frame - self.elapsed))

    def _observe(self):
        begin = perf_counter()
        clock = self.elapsed
        index = floor(clock * self.cfg.camera.fps + 1e-8)
        stamp = clock if self.fault != "stale" else clock - self.cfg.camera.max_frame_age_s - .25
        state = self.controller.read_state()
        image = self.world.image(clock, self.fault)
        projection = self.world.camera.image_to_plane(state.tilt_deg)
        # Bad frames never change the tracker used to extrapolate future poses.
        tracker = Tracker() if self.fault else self.tracker
        obs = synthetic_observation(image, projection, self.cfg, tracker, index, stamp, self.fault)
        measured = perf_counter() - begin
        self.processing_wall["recognition"] += measured
        self._time("recognition", max(self.cfg.objectives.recognition_budget_s, measured))
        self.observation = obs
        self.current = {
            "time_s": clock, "state": state.view(self.cfg) | {"discharged_load": self.world.collected_load},
            "observation": obs.view(), "image_url": self._frame(image),
        }

    def _reserve(self):
        cfg = self.cfg
        return cfg.camera.feedback_s + 1 / cfg.camera.fps + cfg.objectives.recognition_budget_s

    def _safe_reason(self):
        if not self.observation.usable:
            return self.observation.reason
        if self.elapsed - self.observation.captured_s >= self.cfg.camera.max_frame_age_s:
            return "帧时间戳过期，等待新帧"
        return ""

    def _prepare(self):
        self.selected, self.candidates = None, []
        safety = self._safe_reason()
        if safety:
            self._hold(safety)
        else:
            t = perf_counter()
            state = self.controller.read_state()
            remaining = self.cfg.objectives.time_limit_s - self.elapsed - self._reserve() - self.cfg.objectives.planning_budget_s
            due = self.case.tilt_after is not None and self.world.tool_cycles >= self.case.tilt_after and not self.tilt_done
            scheduled = tilt_candidate(self.cfg, state, remaining, self.observation.confidence,
                                       "工况触发倾角改变，完成后重新识别。") if due else None
            if scheduled:
                scheduled.id = "TILT-S"
                self.candidates = [scheduled]
                self.selected = None if scheduled.reasons else scheduled
                reason = "工况倾翻无法执行 · " + "；".join(scheduled.reasons)
            else:
                self.candidates, self.selected, reason = plan(
                    self.observation, state, self.cfg, remaining, self.predicted_iron_used, self.strategy
                )
                if due:
                    self.tilt_done = True
            measured = perf_counter() - t
            self.processing_wall["planning"] += measured
            self._time("planning", max(self.cfg.objectives.planning_budget_s, measured))
            safety = self._safe_reason()
            if safety:
                self.selected = None
                self._hold(safety)
            elif self.selected:
                self.status, self.reason = "ready", ""
                self.no_path = self.empty = 0
            else:
                self.empty = self.empty + 1 if reason == "未见可确认的残渣" else 0
                self.no_path += 1
                self._hold(reason)
                if self.empty >= 3:
                    self._finish("连续 3 帧未见可确认的残渣")
                elif self.no_path >= 4:
                    self._finish(reason)
        if not self.terminal and self.elapsed >= self.cfg.objectives.time_limit_s:
            self._finish("达到时限" + (f" · {self.reason}" if self.reason else ""))
        elif not self.terminal and self.selected is None:
            minimum = self.cfg.objectives.hold_retry_s + 1 / self.cfg.camera.fps
            minimum += self.cfg.objectives.recognition_budget_s
            if self.elapsed + minimum > self.cfg.objectives.time_limit_s:
                self._finish("剩余时间不足以完成下一次识别 · " + self.reason)

    def _hold(self, reason):
        self.status, self.reason = "hold", reason
        self.controller.hold(reason)

    def _finish(self, reason):
        self.reason = reason
        self.selected = None
        self.status = "passed" if self.report()["passed"] else "failed"
        self.controller.hold(reason)

    def _capture_motion(self, t, state):
        stamp = self.elapsed + t
        image = self.world.image(stamp)
        projection = self.world.camera.image_to_plane(state.tilt_deg)
        surface = synthetic_surface(image, projection, self.cfg)
        return {
            "time_s": stamp, "state": state.view(self.cfg) | {"discharged_load": self.world.collected_load},
            "image_url": self._frame(image), "observation": None,
            "surface": surface.view() if surface else None,
        }

    def advance(self):
        if self.terminal:
            return None
        entry = self.current
        decision = self.candidate_view()
        selected = self.selected
        frames = []
        checks = self._safe_reason()
        if checks:
            self._hold(checks)
            selected = None
        if selected:
            before_check = perf_counter()
            budget = self.cfg.objectives.time_limit_s - self.elapsed - self._reserve()
            reasons = hard_reasons(selected.motion, self.cfg, budget)
            if tuple(selected.motion.joints[0]) != self.controller.read_state().joints:
                reasons.append("执行前设备状态已改变")
            if selected.motion.tilts[0] != self.controller.read_state().tilt_deg:
                reasons.append("执行前倾角已改变")
            measured = perf_counter() - before_check
            self.processing_wall["planning"] += measured
            self._time("planning", max(.008, measured))
            safety = self._safe_reason()
            if safety:
                reasons.append(safety)
            if selected.motion.duration > self.cfg.objectives.time_limit_s - self.elapsed - self._reserve():
                reasons.append("动作与反馈将超过剩余时间")
            if reasons:
                self._hold("；".join(dict.fromkeys(reasons)))
                selected = None
        kind = "motion" if selected else "hold"
        start = self.elapsed
        note = self.reason
        motion_s = 0.
        measured = None
        if selected:
            before = self.world.audit()
            strokes_before = self.world.tool_cycles
            iron_before, disturbance_before = self.world.iron, self.world.disturbance
            capture = (lambda t, state: frames.append(self._capture_motion(t, state))) if self.directory else None
            self.controller.submit(selected.motion, capture)
            motion_s = selected.motion.duration
            self._time("motion", motion_s, advance=False)
            self.actions += 1
            self.predicted_iron_used += selected.metrics["iron_risk"]
            if selected.id == "TILT-S":
                self.tilt_done = True
            if self.case.shift_after and self.world.tool_cycles == self.case.shift_after and not self.shift_done:
                self.world.disturb()
                self.shift_done = True
                note = "工况施加扫后剪切扰动"
            after = self.world.audit()
            measured = {
                "removed_m2": before.area_m2 - after.area_m2,
                "removed_load": before.load - after.load,
                "coverage_before": before.coverage, "coverage_after": after.coverage,
                "density_before": before.mean_density, "density_after": after.mean_density,
                "discharged": self.world.tool_cycles > strokes_before,
                "iron": self.world.iron - iron_before, "disturbance": self.world.disturbance - disturbance_before,
            }
            self._next_frame(minimum=self.cfg.camera.feedback_s)
        else:
            self.holds += 1
            self.hold_counts[note or "等待新观测"] += 1
            self._next_frame("holding", self.cfg.objectives.hold_retry_s)
        self._observe()
        if measured is not None:
            measured["observed_delta_m2"] = sum(p["area_m2"] or 0. for p in entry["observation"]["slag"]) - self.observation.area
        self._prepare()
        event = {
            "index": len(self.records) + 1, "kind": kind, "note": note,
            "start_s": start, "motion_s": motion_s, "end_s": self.elapsed,
            "before": entry, "after": self.current, "frames": frames, "decision": decision,
            "action": selected.view(self.cfg) if selected else None, "actual": measured,
        }
        self.records.append(event)
        self._save()
        return event

    def intervene(self, fault=None, disturb=False):
        if self.terminal:
            raise ValueError("本次作业已结束，请重置后再操作")
        entry = self.current
        start = self.elapsed
        if disturb:
            self.world.disturb()
            note = "人为扰动残渣，废弃旧候选"
        else:
            self.fault = fault
            note = "清除故障，等待新帧" if not fault else "注入相机故障，废弃旧候选"
        self.no_path = self.empty = 0
        self._next_frame()
        self._observe()
        self._prepare()
        event = {
            "index": len(self.records) + 1, "kind": "operator", "note": note, "start_s": start,
            "motion_s": 0., "end_s": self.elapsed, "before": entry, "after": self.current,
            "frames": [], "decision": self.candidate_view(), "action": None, "actual": None,
        }
        self.records.append(event)
        self._save()
        return event

    def candidate_view(self):
        return [c.view(self.cfg) for c in self.candidates]

    def report(self):
        audit = self.world.audit()
        initial = self.initial_audit.area_m2
        rate = 1 - audit.area_m2 / initial if initial > 0 else 0.
        o = self.cfg.objectives
        area_ok = rate > o.removal_target
        time_ok = self.elapsed <= o.time_limit_s
        iron_ok = self.world.iron <= o.iron_limit
        first_stroke = next((event["actual"] for event in self.records
                             if event["actual"] and event["actual"]["discharged"]), None)
        return {
            "id": self.id, "case": self.case.model_dump(mode="json"), "seed": self.seed,
            "strategy": self.strategy, "config_revision": self.cfg.revision,
            "sources": {key: getattr(self.cfg, key).source for key in ("tank", "arm", "tilt", "camera", "slag", "objectives")},
            "status": self.status, "stop_reason": self.reason, "ended": self.terminal,
            "elapsed_s": self.elapsed, "initial_m2": initial, "remaining_m2": audit.area_m2,
            "removal_rate": rate, "actions": self.actions, "strokes": self.world.tool_cycles,
            "holds": self.holds, "hold_reasons": dict(self.hold_counts), "iron_risk": self.world.iron,
            "disturbance": self.world.disturbance, "timings": self.timings.copy(),
            "processing_wall_s": self.processing_wall.copy(), "remaining_heatmap": audit.heatmap,
            "initial_coverage": self.initial_audit.coverage, "remaining_coverage": audit.coverage,
            "initial_mean_density": self.initial_audit.mean_density, "remaining_mean_density": audit.mean_density,
            "load_removal_rate": 1 - audit.load / self.initial_audit.load if self.initial_audit.load > 0 else 0.,
            "remaining_density_heatmap": audit.density_heatmap, "first_stroke": first_stroke,
            "surface_model": "continuous-film",
            "thresholds": {"time_s": o.time_limit_s, "removal_rate": o.removal_target, "iron_risk": o.iron_limit},
            "checks": {"area": area_ok, "time": time_ok, "iron": iron_ok},
            "passed": area_ok and time_ok and iron_ok, "reference_passed": self.elapsed <= 120 and rate > .98,
            "field_status": "现场待验证", "archive_issue": self.archive_issue,
        }

    def view(self):
        rate = 1 - self.observation.area / self.initial_observed if self.initial_observed and self.observation.usable else None
        surface = self.observation.surface if self.observation.usable else None
        mean = surface.mean if surface else None
        reduction = 1 - mean / self.initial_surface_mean if mean is not None and self.initial_surface_mean else None
        return {
            "id": self.id, "status": self.status, "reason": self.reason, "fault": self.fault,
            "elapsed_s": self.elapsed, "timings": self.timings.copy(), "actions": self.actions,
            "strokes": self.world.tool_cycles, "coverage": surface.coverage if surface else None,
            "mean_density": mean, "density_reduction": reduction,
            "holds": self.holds, "observed_rate": rate, "iron_risk": self.predicted_iron_used,
            "case": self.case.model_dump(mode="json"), "seed": self.seed, "strategy": self.strategy,
            "config_revision": self.cfg.revision, "current": self.current, "candidates": self.candidate_view(),
            "selected_id": self.selected.id if self.selected else None, "last_index": len(self.records),
            "report": self.report() if self.terminal else None, "archive_issue": self.archive_issue,
        }

    def trace(self):
        return {
            "summary": self.report(), "config": self.cfg.model_dump(mode="json"),
            "initial": self.initial, "events": self.records,
        }

    def _save(self):
        if self.directory:
            try:
                write_json(self.directory / "trace.json", self.trace())
                write_json(self.directory / "summary.json", self.report())
            except OSError:
                self.archive_issue = "作业归档写入失败，请检查运行目录和可用磁盘空间"

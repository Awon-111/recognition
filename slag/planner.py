"""Receding-horizon planner. Inputs are observations, device state, and limits."""

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from .config import Config
from .motion import (MAX_SKIM_FREEBOARD, MIN_FREEBOARD, Motion, State, bath_frame, discharge_point,
                     freeboard, hard_reasons, head_xyz, inverse, outlet_sill, parameterize)
from .perception import Observation
from .surface import MAX_VALUE, OUTSIDE, UNKNOWN, grid_from_plane

STRATEGIES = {"balanced": "综合收益", "coverage": "优先覆盖", "gentle": "低扰动"}
VARIANTS = (
    ("直拉 · 中出渣", 0., 0., 0., 1.),
    ("弧拉 · 左出渣", -.16, -.72, .005, .88),
    ("弧拉 · 右出渣", .16, .72, .005, .88),
    ("缓拉 · 浅贴面", 0., 0., -.005, .69),
)


def _lanes(radii):
    a, b = radii
    return [(f"带道 {i + 1}", (a * x, b * y))
            for i, (x, y) in enumerate(((-.63, .54), (-.37, .71), (0., .77), (.37, .71), (.63, .54),
                                        (-.61, -.05), (0., -.14), (.61, -.05)))]


@dataclass
class Candidate:
    id: str
    label: str
    target: str
    motion: Motion
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    score: float | None = None
    confidence: float = 0.
    explanation: str = ""

    def view(self, cfg):
        return {
            "id": self.id, "label": self.label, "target": self.target, "duration_s": self.motion.duration,
            "valid": not self.reasons, "reasons": self.reasons, "metrics": self.metrics,
            "score": self.score, "confidence": self.confidence, "explanation": self.explanation,
            "trajectory": self.motion.view(cfg),
        }


def _clamp_ellipse(xy, radii, limit):
    fraction = np.linalg.norm(np.divide(xy, radii))
    return np.asarray(xy) * min(1., limit / max(.0001, fraction))


def _waypoints(cfg, state, target_xy, variant):
    _, arc, outlet_offset, depth_delta, speed = variant
    tank, arm = cfg.tank, cfg.arm
    radii = tank.inner_radii_m
    tilt = state.tilt_deg
    bath = bath_frame(tank, tilt)
    start_xy = _clamp_ellipse(target_xy, radii, .78)
    half_width, _, half_depth = np.array(arm.tool_size_m) / 2
    available = max(0., tank.outlet_width_m / 2 - half_width - arm.collision_margin_m - .07)
    slot_x = tank.outlet_x_m + available * outlet_offset
    end_x = bath.reference((tank.pivot_m[0] + slot_x, bath.center_xy[1]))[0]
    end_xy = np.array((end_x, -.76 * radii[1]))
    middle = .53 * start_xy + .47 * end_xy + (arc, 0)
    depth = np.clip(arm.skim_depth_m + depth_delta, .003, .029)
    start, mid, end = bath.world([start_xy, middle, end_xy], half_depth - depth)
    angle = np.deg2rad(tilt)
    rim_max = tank.pivot_m[2] + abs(np.sin(angle)) * tank.width_m / 2 + np.cos(angle) * tank.rim_z_m
    safe_z = float(max(rim_max + half_depth + arm.collision_margin_m + .11, arm.base_m[2] + arm.lift.initial))
    lip = outlet_sill(tank, tilt, slot_x)
    wing = half_width + arm.collision_margin_m
    lip_z = max(outlet_sill(tank, tilt, slot_x + sign * wing)[2] for sign in (-1, 1))
    discharge_z = lip_z + half_depth + arm.collision_margin_m + .035
    exit_position = discharge_point(tank, tilt, slot_x)
    gate = (lip[0], max(end[1] - .04, lip[1] + .19), discharge_z)
    outside = (exit_position[0], exit_position[1] - .045, discharge_z)
    sweep = [("sweep", inverse(mid, 0., arm), speed)] if arc else []
    return [
        ("approach", inverse((*start[:2], safe_z), 0., arm), 1.),
        ("skim", inverse(start, 0., arm), .88),
        *sweep, ("sweep", inverse(end, 0., arm), speed),
        ("discharge", inverse(gate, 0., arm), .82),
        ("discharge", inverse(outside, 0., arm), .94),
        ("retreat", tuple(axis.initial for axis in arm.axes), 1.),
    ]


class Forecast:
    """Short-horizon transport based on the camera's observed concentration field."""

    def __init__(self, observation, cfg, strategy):
        self.cfg, self.strategy = cfg, strategy
        self.radii = np.array(cfg.tank.inner_radii_m)
        self.values = observation.surface.values
        self.semantic = observation.surface.kind == "semantic"
        self.shape = self.values.shape
        self.mapping = grid_from_plane(self.radii, self.shape[::-1])
        self.cell_area = 4 * float(np.prod(self.radii)) / self.values.size
        velocities = [patch.velocity for patch in observation.slag if patch.plane_xy is not None]
        self.drift = np.median(velocities, axis=0) if velocities else np.zeros(2)
        self.buckets = {}

    def cells(self, points):
        return np.int32(np.rint(np.asarray(points) @ self.mapping[:2, :2].T + self.mapping[:2, 2]))

    def material(self, seconds):
        bucket = min(14, max(1, round(seconds)))
        if bucket not in self.buckets:
            dx, dy = self.mapping[:2, :2] @ (self.drift * bucket)
            shift = np.array([[1., 0., dx], [0., 1., dy]])
            self.buckets[bucket] = cv2.warpAffine(self.values, shift, self.shape[::-1],
                                                  flags=cv2.INTER_NEAREST, borderValue=UNKNOWN)
        return self.buckets[bucket]

    def assess(self, motion, variant):
        _, _, _, depth_delta, speed = variant
        cfg = self.cfg
        depth = np.clip(cfg.arm.skim_depth_m + depth_delta, .003, .029)
        sweep_indices = [i for i, phase in enumerate(motion.phases) if phase == "sweep"]
        first, last = sweep_indices[0], sweep_indices[-1]
        start_s, end_s = motion.times[first - 1], motion.times[last]
        samples = np.linspace(start_s, end_s, 32)
        q, _, _ = motion.evaluate(samples)
        bath = bath_frame(cfg.tank, motion.tilts[0])
        head = head_xyz(q, cfg.arm)[:, :2]
        yaw = np.deg2rad(q[:, 2])
        side = np.column_stack((np.cos(yaw), -np.sin(yaw)))
        forward = np.column_stack((np.sin(yaw), np.cos(yaw)))
        hw, hl = np.asarray(cfg.arm.tool_size_m[:2]) / 2
        swath = np.zeros(self.shape, np.uint8)
        for xy, lateral, along in zip(head, side * hw, forward * hl):
            corners = [xy - lateral - along, xy + lateral - along, xy + lateral + along, xy - lateral + along]
            cv2.fillConvexPoly(swath, self.cells(bath.reference(corners)), 1)
        swath = (swath > 0) & (self.values != OUTSIDE)
        observed = self.material((start_s + end_s) / 2)[swath]
        known = observed < OUTSIDE
        density = observed[known] / MAX_VALUE
        area = self.cell_area
        efficiency = cfg.slag.skim_fraction * cfg.slag.contact_efficiency * min(1., .88 + 6 * depth)
        threshold = cfg.slag.coverage_threshold
        expected = float(density.sum() * area * efficiency)
        clear = float(np.count_nonzero((density >= threshold) & (density * (1 - efficiency) < threshold)) * area)
        if self.semantic:
            clear = expected
        clean = float(np.count_nonzero(density < threshold) * area)
        iron = clean * (.045 + 1.3 * depth)
        agitation = float(swath.sum() * area * speed ** 2 * (.06 + depth * .5))
        shift_penalty = cfg.slag.disturbance_shift_m * agitation
        w = cfg.objectives.weights
        metrics = {
            "expected_m2": clear, "expected_load": expected, "iron_risk": iron, "disturbance": agitation,
            "future_risk": shift_penalty, "sweep_area_m2": float(swath.sum() * area),
            "known_fraction": float(known.sum() / max(1, len(known))),
        }
        if self.strategy == "coverage":
            score = (w.removal * (expected + clear * .35) - .20 * w.iron * iron - .22 * w.disturbance * agitation
                     - .18 * w.time * motion.duration - .20 * w.future * shift_penalty)
        else:
            gentle = 2.7 if self.strategy == "gentle" else 1.
            score = (w.removal * expected - gentle * (w.iron * iron + w.disturbance * agitation + w.future * shift_penalty)
                     - w.time * motion.duration)
        return metrics, float(score)


def tilt_candidate(cfg, state, remaining, confidence, reason, goal=None):
    goal = min(cfg.tilt.max_deg, state.tilt_deg + cfg.tilt.step_deg) if goal is None else goal
    if abs(goal - state.tilt_deg) <= 1e-6:
        return None
    motion = parameterize(cfg, state, [], tilt=goal)
    return Candidate(
        "TILT", "倾罐就位", "罐体", motion, hard_reasons(motion, cfg, remaining),
        {"expected_m2": 0., "expected_load": 0., "iron_risk": 0., "disturbance": 0., "future_risk": 0.},
        0., confidence, reason,
    )


@lru_cache(maxsize=12)
def _reference_lane_checks(config_json, state, remaining_s):
    cfg = Config.model_validate_json(config_json)
    checks = {}
    for target, xy in _lanes(cfg.tank.inner_radii_m):
        for index, variant in enumerate(VARIANTS, start=1):
            motion = parameterize(cfg, state, _waypoints(cfg, state, xy, variant))
            checks[(target, index)] = motion, tuple(hard_reasons(motion, cfg, remaining_s))
    return checks


def plan(observation: Observation, state: State, cfg, remaining_s, iron_used, strategy):
    if not observation.usable:
        return [], None, observation.reason
    gap = freeboard(cfg.tank, state.tilt_deg)
    goal = cfg.tilt.working_deg
    needs_tilt = state.tilt_deg <= .1 or gap > MAX_SKIM_FREEBOARD or gap < MIN_FREEBOARD
    needs_tilt |= goal is not None and state.tilt_deg < goal - 1e-6
    if needs_tilt:
        if goal is None:
            goal = next((float(a) for a in np.linspace(max(.2, cfg.tilt.min_deg), cfg.tilt.max_deg, 80)
                         if a > .1 and .02 <= freeboard(cfg.tank, a) <= .095), None)
        if not cfg.tilt.allow_planner or goal is None:
            return [], None, "需先倾罐至可扒渣角度"
        ready = tilt_candidate(cfg, state, remaining_s, observation.confidence,
                               "先降低出渣口，再依据新帧规划沿液面向外的回拉轨迹。", goal=goal)
        if ready is None or freeboard(cfg.tank, goal) > MAX_SKIM_FREEBOARD:
            return [], None, "配置的工作倾角无法把出渣口降至贴面排渣位置"
        return [ready], None if ready.reasons else ready, "；".join(ready.reasons)
    surface = observation.surface
    if surface is None or surface.mean is None:
        return [], None, "渣层浓度图不可用，等待新观测"
    if not np.any((surface.values < OUTSIDE) & (surface.values >= cfg.slag.coverage_threshold * MAX_VALUE)):
        return [], None, "未见可确认的残渣"
    forecast = Forecast(observation, cfg, strategy)
    radii = cfg.tank.inner_radii_m
    b = radii[1]
    targets = _lanes(radii)
    checked = _reference_lane_checks(cfg.model_dump_json(), state, remaining_s) if forecast.semantic else {}
    patches = [p for p in observation.slag if p.plane_xy and p.confidence >= cfg.camera.min_confidence
               and .006 < (p.area_m2 or 0) < 1.]
    for p in sorted(patches, key=lambda p: -(p.area_m2 or 0))[:3]:
        point = np.asarray(p.plane_xy) + np.asarray(p.velocity) * 2. + (0, .21)
        targets.append((p.id, (point[0], max(-.58 * b, point[1]))))
    candidates = []
    for target, xy in targets:
        for index, variant in enumerate(VARIANTS, start=1):
            cached = checked.get((target, index))
            if cached is None:
                motion = parameterize(cfg, state, _waypoints(cfg, state, xy, variant))
                reasons = hard_reasons(motion, cfg, remaining_s)
            else:
                motion, recorded_reasons = cached
                reasons = list(recorded_reasons)
            candidate = Candidate(f"{target}-{index}", variant[0], target, motion, reasons, confidence=observation.confidence)
            if not candidate.reasons:
                candidate.metrics, candidate.score = forecast.assess(motion, variant)
                expected, iron = candidate.metrics["expected_load"], candidate.metrics["iron_risk"]
                if candidate.metrics["known_fraction"] < .45:
                    candidate.reasons.append("带道遮挡过多")
                if expected <= .00045:
                    candidate.reasons.append("短时推演未覆盖可信残渣")
                if iron_used + iron > cfg.objectives.iron_limit:
                    candidate.reasons.append("累计铁损风险预计超过上限")
                candidate.explanation = f"沿液面回拉至罐外。预计带出渣量指数 {expected:.3f}，裸露铁水增加 {candidate.metrics['expected_m2']:.3f} m²。"
                if forecast.semantic:
                    candidate.metrics.pop("expected_load")
                    candidate.explanation = f"沿液面回拉至罐外。参考罐体上预计清理残渣面积 {expected:.3f} m²。"
            candidates.append(candidate)
    candidates.sort(key=lambda item: (bool(item.reasons), -(item.score if item.score is not None else -1e6), item.id))
    selected = next((item for item in candidates if not item.reasons), None)
    if selected:
        return candidates, selected, ""
    reasons = list(dict.fromkeys(reason for row in candidates for reason in row.reasons))
    return candidates, None, "无可行动作 · " + "；".join(reasons[:4])

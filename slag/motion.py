"""Four-axis machine model, time parameterization and collision checks."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import cos, radians, sin, sqrt

import numpy as np

from .config import Config, Vec4

AXES = ("extension_m", "lift_m", "swing_deg", "pitch_deg")
PHASES = {
    "ready": "待机", "approach": "高位伸入", "dip": "入渣", "skim": "贴面",
    "sweep": "回拉扒渣", "discharge": "越口排渣", "lift": "抬起", "retreat": "退出", "tilt": "倾罐",
}
CORNER_SIGNS = np.array(list(product((-1, 1), repeat=3)))
MIN_FREEBOARD = .012
MAX_SKIM_FREEBOARD = .13


def rotation(tilt):
    angle = np.deg2rad(tilt)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def to_world(local, tank, tilt):
    return np.asarray(local) @ rotation(tilt).T + tank.pivot_m


def to_local(world, tank, tilt):
    return (np.asarray(world) - tank.pivot_m) @ rotation(tilt)


def ellipse(radii, z, n=90):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radii[0] * np.cos(angle), radii[1] * np.sin(angle), np.full(n, z)))


@dataclass(frozen=True)
class Bath:
    center_xy: tuple[float, float]
    radii: tuple[float, float]
    reference_radii: tuple[float, float]
    z: float

    def world(self, xy, height=0.):
        xy = np.asarray(xy)
        physical = np.asarray(self.center_xy) + xy * np.divide(self.radii, self.reference_radii)
        return np.concatenate((physical, np.broadcast_to(self.z + height, (*physical.shape[:-1], 1))), axis=-1)

    def reference(self, world_xy):
        return (np.asarray(world_xy) - self.center_xy) * np.divide(self.reference_radii, self.radii)

    def view(self):
        return {"center_xy": self.center_xy, "radii_m": self.radii, "z_m": self.z}


@lru_cache(maxsize=512)
def _horizontal_bath(inner, rim, pivot, liquid_z, rim_z, bottom_z, bottom_scale, tilt):
    a = radians(tilt)
    c, s = cos(a), sin(a)

    def radius(axis, height):
        rise = max(0., min(1., (height - bottom_z) / (liquid_z - bottom_z)))
        flare = max(0., min(1., (height - liquid_z) / (rim_z - liquid_z)))
        return inner[axis] * (bottom_scale + (1 - bottom_scale) * rise) + (rim[axis] - inner[axis]) * flare

    bounds = []
    for sign in (-1, 1):
        lo, hi = 0., rim[1] / c + .1
        for _ in range(18):
            extent = (lo + hi) / 2
            if extent * c < radius(1, liquid_z - sign * extent * s):
                lo = extent
            else:
                hi = extent
        bounds.append(sign * (lo + hi) / 2)
    shift = sum(bounds) / 2
    return Bath((pivot[0], pivot[1] - liquid_z * s + shift),
                (radius(0, liquid_z - shift * s), (bounds[1] - bounds[0]) / 2), inner,
                pivot[2] + liquid_z * c)


def bath_frame(tank, tilt):
    return _horizontal_bath(tank.inner_radii_m, tank.rim_radii_m, tank.pivot_m, tank.liquid_z_m,
                            tank.rim_z_m, tank.rim_z_m - tank.height_m, tank.bottom_scale, float(tilt))


def outlet_sill(tank, tilt, x=None):
    x = tank.outlet_x_m if x is None else x
    y = -tank.rim_radii_m[1] * sqrt(max(0., 1 - (x / tank.rim_radii_m[0]) ** 2))
    return to_world((x, y, tank.rim_z_m - tank.outlet_drop_m), tank, tilt)


def freeboard(tank, tilt):
    return float(outlet_sill(tank, tilt)[2] - bath_frame(tank, tilt).z)


def discharge_point(tank, tilt, x):
    return to_world((x, -tank.width_m / 2 - tank.outlet_extension_m, tank.rim_z_m - tank.outlet_drop_m), tank, tilt)


@dataclass(frozen=True)
class State:
    joints: Vec4
    tilt_deg: float

    def view(self, cfg):
        return {
            "joints": dict(zip(AXES, self.joints)), "tilt_deg": self.tilt_deg,
            "head_m": head_xyz(self.joints, cfg.arm).tolist(),
            "bath": bath_frame(cfg.tank, self.tilt_deg).view(),
        }


def initial_state(cfg):
    return State(tuple(axis.initial for axis in cfg.arm.axes), cfg.tilt.initial_deg)


def head_xyz(joints, arm):
    q = np.asarray(joints)
    yaw = np.deg2rad(q[..., 2])
    return np.stack((q[..., 0] * np.sin(yaw), q[..., 0] * np.cos(yaw), q[..., 1]), axis=-1) + arm.base_m


def inverse(xyz, pitch, arm):
    x, y, z = np.asarray(xyz) - arm.base_m
    return float(np.hypot(x, y)), float(z), float(np.rad2deg(np.arctan2(x, y))), float(pitch)


def tool_corners(joints, arm, padded=True):
    q = np.atleast_2d(joints)
    yaw, pitch = np.deg2rad(q[:, 2]), np.deg2rad(q[:, 3])
    lateral = np.column_stack((np.cos(yaw), -np.sin(yaw), np.zeros(len(q))))
    forward = np.column_stack((np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)))
    up = np.cross(lateral, forward)
    padding = arm.collision_margin_m if padded else 0.
    offsets = CORNER_SIGNS * (np.asarray(arm.tool_size_m) / 2 + padding)
    axes = np.stack((lateral, forward, up), axis=1)
    return head_xyz(q, arm)[:, None, :] + offsets @ axes


@dataclass
class Motion:
    times: np.ndarray
    joints: np.ndarray
    tilts: np.ndarray
    phases: list[str]

    @property
    def duration(self):
        return float(self.times[-1])

    @property
    def end(self):
        return State(tuple(self.joints[-1]), float(self.tilts[-1]))

    def evaluate(self, times):
        t = np.clip(np.atleast_1d(times), 0, self.duration)
        i = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        durations = self.times[i + 1] - self.times[i]
        u = (t - self.times[i]) / durations
        blend = u * u * (3 - 2 * u)
        q = self.joints[i] + (self.joints[i + 1] - self.joints[i]) * blend[:, None]
        tilt = self.tilts[i] + (self.tilts[i + 1] - self.tilts[i]) * blend
        return q, tilt, i + 1

    def sample_times(self):
        pieces = [np.array([0.0])]
        for index, dt in enumerate(np.diff(self.times)):
            delta = np.abs(self.joints[index + 1] - self.joints[index])
            resolution = np.maximum(delta / (.032, .032, .60, .80), 0).max()
            n = int(np.ceil(max(1, dt / .12, resolution, abs(self.tilts[index + 1] - self.tilts[index]) / .35)))
            pieces.append(np.linspace(self.times[index], self.times[index + 1], n + 1)[1:])
        return np.concatenate(pieces)

    def view(self, cfg):
        speeds = 1.5 * np.diff(self.joints, axis=0) / np.diff(self.times)[:, None]
        tilt_speeds = 1.5 * np.diff(self.tilts) / np.diff(self.times)
        return [
            {
                "t_s": float(t), "phase": self.phases[i],
                **State(tuple(q), float(tilt)).view(cfg),
                "segment_peak_speed": dict(zip(AXES, (abs(speeds[i - 1]) if i else (0,) * 4))),
                "segment_peak_tilt_deg_s": abs(float(tilt_speeds[i - 1])) if i else 0,
            }
            for i, (t, q, tilt) in enumerate(zip(self.times, self.joints, self.tilts))
        ]


def parameterize(cfg: Config, state: State, waypoints, tilt=None):
    joints = [state.joints]
    tilts = [state.tilt_deg]
    phases = ["ready"]
    times = [0.0]
    axis_speeds = np.array([axis.speed for axis in cfg.arm.axes])
    for phase, q, fraction in waypoints:
        dt = float(np.max(np.abs(np.subtract(q, joints[-1])) / (axis_speeds * fraction)))
        # Smoothstep has a peak derivative of 1.5, so speed stays inside each axis limit.
        times.append(times[-1] + max(.04, 1.5 * dt))
        joints.append(q)
        tilts.append(tilts[-1])
        phases.append(phase)
    if tilt is not None:
        times.append(times[-1] + max(.04, 1.5 * abs(tilt - tilts[-1]) / cfg.tilt.speed_deg_s))
        joints.append(joints[-1])
        tilts.append(tilt)
        phases.append("tilt")
    return Motion(np.array(times), np.array(joints), np.array(tilts), phases)


def _tank_collision(world_points, tilt, cfg):
    tank = cfg.tank
    v = world_points - tank.pivot_m
    c = np.cos(np.deg2rad(tilt))[:, None]
    s = np.sin(np.deg2rad(tilt))[:, None]
    x, y, z = v[:, :, 0], c * v[:, :, 1] + s * v[:, :, 2], -s * v[:, :, 1] + c * v[:, :, 2]
    bottom = tank.rim_z_m - tank.height_m
    inner = np.array(tank.inner_radii_m)
    rim = np.array(tank.rim_radii_m)
    fill = np.clip((z - bottom) / (tank.liquid_z_m - bottom), 0, 1)
    flare = np.clip((z - tank.liquid_z_m) / (tank.rim_z_m - tank.liquid_z_m), 0, 1)
    radii = inner * (tank.bottom_scale + (1 - tank.bottom_scale) * fill[..., None]) + (rim - inner) * flare[..., None]
    outside = (x / radii[..., 0]) ** 2 + (y / radii[..., 1]) ** 2 >= 1
    slot = (np.abs(x - tank.outlet_x_m) < tank.outlet_width_m / 2) & (y < 0)
    wall_top = tank.rim_z_m - tank.outlet_drop_m * slot
    return np.any((z < wall_top) & (outside | (z <= bottom)))


def _segment_intersects(a, b, lower, upper):
    d = b - a
    parallel = np.abs(d) < 1e-10
    div = np.where(parallel, 1, d)
    enter, leave = (lower - a) / div, (upper - a) / div
    lo = np.where(parallel, -np.inf, np.minimum(enter, leave))
    hi = np.where(parallel, np.inf, np.maximum(enter, leave))
    outside_parallel = np.any(parallel & ((a < lower) | (a > upper)), axis=1)
    return np.any(~outside_parallel & (np.maximum(0, lo.max(axis=1)) <= np.minimum(1, hi.min(axis=1))))


def hard_reasons(motion, cfg, remaining_s):
    reasons = []
    if motion.duration > remaining_s:
        reasons.append("动作与反馈将超过剩余时间")
    for i, axis in enumerate(cfg.arm.axes):
        if motion.joints[:, i].min() < axis.min - 1e-8 or motion.joints[:, i].max() > axis.max + 1e-8:
            reasons.append(f"{('伸缩', '升降', '回转', '俯仰')[i]}超行程")
    if motion.tilts.min() < cfg.tilt.min_deg or motion.tilts.max() > cfg.tilt.max_deg:
        reasons.append("倾角超范围")
    if reasons:
        return reasons
    q, tilt, phase = motion.evaluate(motion.sample_times())
    if any(freeboard(cfg.tank, angle) < MIN_FREEBOARD for angle in np.unique(tilt)):
        reasons.append("倾翻使出渣口的铁水安全余量不足")
    head = head_xyz(q, cfg.arm)
    corners = tool_corners(q, cfg.arm)
    boom_end = head.copy()
    boom_end[:, 2] = cfg.arm.boom_z_m
    boom_start = np.broadcast_to((*cfg.arm.base_m[:2], cfg.arm.boom_z_m), boom_end.shape)
    column_width = cfg.arm.collision_margin_m + .075
    column = head[:, None, :] + (boom_end - head)[:, None, :] * np.linspace(.08, 1, 12)[None, :, None]
    offsets = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]) * column_width
    column_edges = (column[:, :, None, :] + offsets).reshape(len(head), -1, 3)
    if _tank_collision(np.concatenate((corners, column_edges), axis=1), tilt, cfg):
        reasons.append("工具或升降杆与罐口／内壁安全包络相交")
    low, high = corners.min(axis=1), corners.max(axis=1)
    for box in cfg.arm.safety_boxes:
        bmin, bmax = np.array(box.min_m), np.array(box.max_m)
        head_hits = np.any(np.all((high >= bmin) & (low <= bmax), axis=1))
        rod_hits = _segment_intersects(head, boom_end, bmin - column_width, bmax + column_width)
        boom_pad = .11 + cfg.arm.collision_margin_m
        boom_hits = _segment_intersects(boom_start, boom_end, bmin - boom_pad, bmax + boom_pad)
        if head_hits or rod_hits or boom_hits:
            reasons.append(f"进入禁入区「{box.name}」")
    if np.min(corners[..., 2]) < 0:
        reasons.append("工具低于地面安全边界")
    if "sweep" in motion.phases:
        bath = bath_frame(cfg.tank, tilt[0])
        sweep = np.array([motion.phases[i] == "sweep" for i in phase])
        edge_z = tool_corners(q[sweep], cfg.arm, padded=False)[..., 2].min(axis=1)
        if tilt[0] <= .1 or freeboard(cfg.tank, tilt[0]) > MAX_SKIM_FREEBOARD:
            reasons.append("罐体须先倾翻至可扒渣位置")
        if np.any((edge_z < bath.z - .0305) | (edge_z > bath.z + .025)):
            reasons.append("回拉期间扒渣板未保持在允许的液面接触带")
        if "discharge" not in motion.phases:
            reasons.append("轨迹须从出渣口排渣")
        else:
            exit_index = max(i for i, name in enumerate(motion.phases) if name == "discharge")
            exit_pose = head_xyz(motion.joints[exit_index], cfg.arm)
            local = to_local(exit_pose, cfg.tank, tilt[0])
            exit_y = discharge_point(cfg.tank, tilt[0], local[0])[1]
            if exit_pose[1] > exit_y + .025 or abs(local[0] - cfg.tank.outlet_x_m) >= cfg.tank.outlet_width_m / 2:
                reasons.append("扒渣板未沿出渣口到达罐外")
    return reasons

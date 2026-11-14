"""Private surface simulation. Only rendered frames cross into perception."""

from dataclasses import dataclass
from math import ceil, sqrt

import cv2
import numpy as np

from .motion import State, bath_frame, ellipse, head_xyz, initial_state, to_local, to_world, tool_corners
from .perception import VirtualCamera
from .surface import VIRTUAL_CONTRAST, VIRTUAL_GREEN, grid_from_plane


@dataclass(frozen=True)
class Audit:
    area_m2: float
    coverage: float
    mean_density: float
    load: float
    heatmap: list
    density_heatmap: list


class World:
    def __init__(self, cfg, scenario, seed):
        self.cfg = cfg
        self.case = scenario
        self.seed = seed
        self.state = initial_state(cfg)
        self.camera = VirtualCamera(cfg)
        self.tool_cycles = 0
        self.collected_load = 0.
        self.iron = 0.
        self.disturbance = 0.
        self._rng = np.random.default_rng(seed)
        self._angle = 0.
        self._density = self._initial_surface()
        self._backgrounds = {}
        self._screens = {}
        self._noise_grid = self._rng.normal(0, 1, (43, 58)).astype(np.float32)

    def _initial_surface(self):
        settings = self.cfg.slag
        radii = np.asarray(self.cfg.tank.inner_radii_m)
        stride = settings.grid_step_m
        size = tuple(np.ceil(2 * radii / stride).astype(int))
        w, h = size
        u, v = np.meshgrid((np.arange(w) + .5) * 2 / w - 1, 1 - (np.arange(h) + .5) * 2 / h)
        self._uv = u, v
        radius2 = u * u + v * v
        self._disk = radius2 <= 1
        self._flow_gain = 1.16 - .48 * radius2
        self._unit_cells = np.column_stack((u[self._disk], v[self._disk]))
        self._grid = grid_from_plane(radii, size)
        self._cell_area = 4 * float(np.prod(radii)) / (w * h)
        xx, yy = u * radii[0], v * radii[1]
        layout = {"均匀": "scattered", "聚集": "clustered"}.get(settings.distribution, self.case.layout)
        centers = []
        radii_scale = self.case.radius_m * sqrt(settings.area_scale)
        for i in range(self.case.clusters):
            for _ in range(100):
                if layout == "clustered":
                    anchor = np.array((-.55, .35)) if i < self.case.clusters * .68 else np.array((.68, -.63))
                    center = anchor + self._rng.normal(0, .44 - .25 * settings.aggregation, size=2)
                elif layout == "band":
                    angle = self._rng.uniform(-np.pi, np.pi)
                    center = radii * self._rng.uniform(.36, .70) * (np.cos(angle), np.sin(angle))
                else:
                    center = self._rng.uniform(-.8, .8, size=2) * radii
                close = any(np.linalg.norm(center - other) < radii_scale * .95 for other in centers)
                if np.linalg.norm(center / radii) < .79 and (layout == "clustered" or not close):
                    break
            centers.append(center)
        phase = self._rng.uniform(0, 2 * np.pi, 3)
        texture = .042 * np.sin(6.4 * xx + 2.6 * yy + phase[0])
        texture += .038 * np.cos(8.4 * yy - 3.6 * xx + phase[1])
        texture += .022 * np.sin(16 * xx + 15 * yy + phase[2])
        density = settings.initial_film + texture
        for i, center in enumerate(centers):
            angle = self._rng.uniform(0, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            dx, dy = xx - center[0], yy - center[1]
            a, b = radii_scale * self._rng.uniform(.68, 1.45, size=2)
            ell = ((dx * c + dy * s) / a) ** 2 + ((dy * c - dx * s) / b) ** 2
            density += (.12 + .10 * settings.aggregation) * np.exp(-.54 * ell)
        # Small naturally exposed windows give the dense starting film a visible boundary.
        eye = ((u + .81) / .14) ** 2 + ((v - .36) / .17) ** 2
        density *= np.clip(eye - .48, 0., 1.)
        return np.float32(np.where(self._disk, np.clip(density, 0., 1.), 0.))

    def _positions(self):
        u, v = self._unit_cells.T
        angle = self._angle * self._flow_gain[self._disk]
        c, s = np.cos(angle), np.sin(angle)
        return np.column_stack((u * c - v * s, u * s + v * c))

    def _rotate_image(self):
        u, v = self._uv
        angle = self._angle * self._flow_gain
        c, s = np.cos(angle), np.sin(angle)
        h, w = self._density.shape
        map_x = np.float32((u * c + v * s + 1) * w / 2 - .5)
        map_y = np.float32((1 - (v * c - u * s)) * h / 2 - .5)
        return cv2.remap(self._density, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _mix_surface(self):
        # Symmetric exchanges across each in-bath edge preserve total modeled concentration.
        for axis in (0, 1):
            density = np.moveaxis(self._density, axis, 0)
            disk = np.moveaxis(self._disk, axis, 0)
            exchange = .18 * (density[1:] - density[:-1]) * (disk[1:] & disk[:-1])
            density[:-1] += exchange
            density[1:] -= exchange

    def wait(self, dt):
        radius = sqrt(np.prod(self.cfg.tank.inner_radii_m))
        tilt_gain = 1 + .9 * np.sin(np.deg2rad(self.state.tilt_deg))
        speed = self.cfg.slag.drift_m_s * self.case.drift_factor * tilt_gain
        self._angle += dt * speed / radius

    def disturb(self):
        self._angle += .50 + self.cfg.slag.disturbance_shift_m
        self.disturbance += .62

    def audit(self):
        values = self._density[self._disk]
        covered = values >= self.cfg.slag.coverage_threshold
        x, y = self._positions().T
        bin_ranges = dict(bins=(18, 22), range=((-1, 1), (-1, 1)))
        grid = np.histogram2d(y[covered], x[covered], **bin_ranges)[0]
        density_map = np.histogram2d(y, x, weights=values, **bin_ranges)[0]
        return Audit(float(covered.sum() * self._cell_area), float(covered.mean()),
                     float(values.mean()), float(values.sum() * self._cell_area),
                     grid[::-1].astype(int).tolist(), density_map[::-1].round(3).tolist())

    def execute(self, motion, capture=None):
        cfg = self.cfg
        times = motion.sample_times()
        captures = set()
        if capture is not None:
            sample = np.r_[np.linspace(0, motion.duration, ceil(motion.duration * 4) + 1)[1:], motion.times[1:]]
            captures = set(np.minimum(len(times) - 1, np.searchsorted(times, sample)))
        joints, tilts, _ = motion.evaluate(times)
        positions = head_xyz(joints, cfg.arm)
        density = self._density[self._disk].copy()
        hit = np.zeros_like(density)
        iron = 0.
        agitation = 0.
        sweep_indices = [i for i, phase in enumerate(motion.phases) if phase == "sweep"]
        discharge_indices = [i for i, phase in enumerate(motion.phases) if phase == "discharge"]
        sweep_start = motion.times[sweep_indices[0] - 1] if sweep_indices else float("inf")
        sweep_end = motion.times[sweep_indices[-1]] if sweep_indices else -1.
        discharge_end = motion.times[discharge_indices[-1]] if discharge_indices else -1.
        half_width, half_length, half_depth = np.asarray(cfg.arm.tool_size_m) / 2
        for i in range(1, len(times)):
            self.state = State(tuple(joints[i]), float(tilts[i]))
            dt = times[i] - times[i - 1]
            self.wait(dt)
            if sweep_start < times[i] <= sweep_end:
                bath = bath_frame(cfg.tank, self.state.tilt_deg)
                edge_depth = bath.z + half_depth - positions[i, 2]
                if -.025 <= edge_depth <= .0305:
                    yaw = np.deg2rad(joints[i, 2])
                    c, s = np.cos(yaw), np.sin(yaw)
                    delta = self._positions() * bath.radii + bath.center_xy - positions[i, :2]
                    dx, dy = delta.T
                    mask = (np.abs(dx * c - dy * s) <= half_width) & (np.abs(dx * s + dy * c) <= half_length)
                    efficacy = cfg.slag.skim_fraction * cfg.slag.contact_efficiency
                    hit[mask] = np.maximum(hit[mask], efficacy * np.clip(.88 + 6 * edge_depth, .7, 1.))
                    length = float(np.linalg.norm(positions[i, :2] - positions[i - 1, :2]))
                    footprint = cfg.arm.tool_size_m[0] * length
                    exposed = float(np.mean(density[mask] < cfg.slag.coverage_threshold)) if mask.any() else 0.
                    iron += footprint * exposed * (.045 + 1.3 * max(0., edge_depth))
                    agitation += footprint * (length / dt) * (.06 + .5 * max(0., edge_depth))
            if abs(times[i] - discharge_end) < 1e-9 and sweep_indices:
                # Collection is credited when the load reaches the exterior side of the lip.
                self.collected_load += float(np.sum(density * hit) * self._cell_area)
                self._density[self._disk] = density * (1 - hit)
                self._mix_surface()
                self.tool_cycles += 1
                self._angle += cfg.slag.disturbance_shift_m * (1 + agitation) / sqrt(np.prod(cfg.tank.inner_radii_m))
            if i in captures:
                capture(float(times[i]), self.state)
        self.state = motion.end
        self.iron += float(iron)
        self.disturbance += float(agitation)

    def _base_image(self):
        tilt = round(self.state.tilt_deg, 6)
        cached = self._backgrounds.get(tilt)
        if cached is not None:
            return cached
        cfg, camera = self.cfg, self.camera
        tank = cfg.tank
        w, h = cfg.camera.resolution
        yy, xx = self._pixel_grid()
        vignette = np.maximum(0, 1 - ((xx / w - .5) ** 2 + (yy / h - .48) ** 2) * 2)
        image = np.stack((12 + 16 * vignette, 10 + 12 * vignette, 9 + 12 * vignette), axis=-1).astype(np.uint8)
        for y in range(16, h, max(12, h // 17)):
            cv2.line(image, (0, y), (w, y), (24, 22, 20), 1)
        for x in range(0, w, max(16, w // 22)):
            cv2.line(image, (x, 0), (x, h), (24, 22, 20), 1)

        def poly(radii, z):
            rim = ellipse(radii, z)
            slot = (np.abs(rim[:, 0] - tank.outlet_x_m) < tank.outlet_width_m / 2) & (rim[:, 1] < 0)
            rim[slot, 2] -= tank.outlet_drop_m
            return np.round(camera.project(to_world(rim, tank, tilt))).astype(np.int32)

        outside = poly((tank.length_m / 2, tank.width_m / 2), tank.rim_z_m)
        rim = poly(tank.rim_radii_m, tank.rim_z_m)
        bath = bath_frame(tank, tilt)
        surface_xyz = ellipse(bath.radii, bath.z)
        surface_xyz[:, :2] += bath.center_xy
        surface = np.int32(np.rint(camera.project(surface_xyz)))
        cv2.fillPoly(image, [outside], (59, 61, 69))
        cv2.polylines(image, [outside], True, (101, 114, 124), 2, cv2.LINE_AA)
        cv2.fillPoly(image, [rim], (73, 82, 93))
        cv2.polylines(image, [rim], True, (105, 112, 123), 2, cv2.LINE_AA)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [surface], 1)
        mask = mask > 0
        glow = (np.sin(xx / 25 + yy / 31) + np.cos(yy / 17 - xx / 38)) * 6 + 114
        image[mask] = np.column_stack((glow[mask] * .21, glow[mask], 210 + glow[mask] * .30))
        if len(self._backgrounds) >= 4:
            self._backgrounds.clear()
        projection = self.camera.image_to_plane(tilt)
        grid_to_image = np.linalg.inv(self._grid @ projection) if projection is not None else None
        self._backgrounds[tilt] = image, mask, grid_to_image
        return image, mask, grid_to_image

    def _pixel_grid(self):
        if "grid" not in self._screens:
            w, h = self.cfg.camera.resolution
            self._screens["grid"] = np.indices((h, w), dtype=np.float32)
        return self._screens["grid"]

    def image(self, time_s, fault=None):
        background, mask, grid_to_image = self._base_image()
        image = background.copy()
        if fault == "offline":
            return np.zeros_like(image)
        cfg = self.cfg
        tank = cfg.tank
        tilt = self.state.tilt_deg
        w, h = cfg.camera.resolution
        yy, xx = self._pixel_grid()
        if grid_to_image is not None:
            density = cv2.warpPerspective(self._rotate_image(), grid_to_image, (w, h), flags=cv2.INTER_LINEAR)
            film = mask & (density >= cfg.slag.coverage_threshold)
            values = density[film]
            image[film] = np.column_stack((
                118 + 3.5 * np.sin(xx[film] * .28 + yy[film] * .22),
                VIRTUAL_GREEN - VIRTUAL_CONTRAST * values,
                187 - 91 * values + 3 * np.cos(yy[film] * .39 - xx[film] * .16),
            ))

        head = head_xyz(self.state.joints, cfg.arm)
        boom_end = (*head[:2], cfg.arm.boom_z_m)
        boom_start = (*cfg.arm.base_m[:2], cfg.arm.boom_z_m)
        p0, p1, p2 = self.camera.project([boom_start, boom_end, head]).round().astype(int)
        cv2.line(image, tuple(p0), tuple(p1), (148, 120, 44), max(5, w // 94), cv2.LINE_AA)
        cv2.line(image, tuple(p1), tuple(p2), (164, 134, 49), max(3, w // 140), cv2.LINE_AA)
        corner_px = np.int32(self.camera.project(tool_corners(self.state.joints, cfg.arm, padded=False)[0]).round())
        head_poly = cv2.convexHull(corner_px)
        cv2.fillConvexPoly(image, head_poly, (164, 134, 50), cv2.LINE_AA)
        cv2.polylines(image, [head_poly], True, (222, 194, 106), 1, cv2.LINE_AA)

        fraction = self.case.steam
        if self.case.steam_window_s and not self.case.steam_window_s[0] <= time_s < self.case.steam_window_s[1]:
            fraction = .05
        if fault == "steam":
            fraction = .64
        if fraction and np.count_nonzero(mask) > 100:
            field = np.sin(xx / (w / 3.8) + time_s * .13) + .58 * np.cos(yy / (h / 12.) - time_s * .18)
            veil = mask & (field >= np.quantile(field[mask], 1 - fraction))
            edge = cv2.GaussianBlur(np.float32(veil), (21, 21), 0)
            blend = np.minimum(.45, .56 * edge)[..., None]
            image = np.uint8(image * (1 - blend) + np.array((145, 155, 164)) * blend)
            tone = 149 + 12 * np.sin(xx[veil] / 22 + yy[veil] / 34 + time_s * .3)
            image[veil] = np.column_stack((tone, tone + 5, tone + 10))
        if cfg.camera.noise:
            stamp = int(time_s * cfg.camera.fps)
            noise = cv2.resize(np.roll(self._noise_grid, stamp % 57, axis=1), (w, h), interpolation=cv2.INTER_LINEAR)
            image = np.uint8(np.clip(image.astype(np.float32) + cfg.camera.noise * noise[..., None], 0, 255))
        return image

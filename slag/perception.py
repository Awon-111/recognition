"""Image-only observations, pixel/plane geometry, and per-stream tracking."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .motion import bath_frame
from .surface import MAX_VALUE, UNKNOWN, VIRTUAL_CONTRAST, VIRTUAL_GREEN, SurfaceField, project_field

cv2.setNumThreads(1)


@dataclass
class Patch:
    center_px: tuple[float, float]
    contour_px: list
    pixel_area: int
    plane_xy: tuple[float, float] | None
    plane_contour: list | None
    area_m2: float | None
    visual_class: str
    confidence: float
    texture_score: float
    id: str = ""
    velocity: tuple[float, float] = (0., 0.)
    velocity_px_s: tuple[float, float] = (0., 0.)


@dataclass
class Observation:
    frame: int
    captured_s: float
    size: tuple[int, int]
    slag: list[Patch] = field(default_factory=list)
    mouth: list = field(default_factory=list)
    liquid: list = field(default_factory=list)
    iron: list = field(default_factory=list)
    slag_regions: list | None = None
    iron_regions: list | None = None
    tool: list = field(default_factory=list)
    tool_center_px: tuple | None = None
    steam: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
    steam_fraction: float = 0.
    confidence: float = 0.
    usable: bool = False
    reason: str = ""
    source: str = "虚拟相机"
    surface: SurfaceField | None = None

    def view(self):
        return {
            "frame": self.frame, "captured_s": self.captured_s, "size": self.size,
            "slag": [vars(patch) for patch in self.slag], "mouth": self.mouth, "liquid": self.liquid,
            "iron": self.iron, "surface": self.surface.view() if self.surface else None,
            "slag_regions": self.slag_regions, "iron_regions": self.iron_regions,
            "tool": self.tool, "tool_center_px": self.tool_center_px,
            "steam": self.steam, "unknown": self.unknown, "steam_fraction": self.steam_fraction,
            "confidence": self.confidence, "usable": self.usable, "reason": self.reason,
            "source": self.source,
        }

    @property
    def area(self):
        return sum(patch.area_m2 or 0. for patch in self.slag)


class Tracker:
    def __init__(self, ttl_s=28.):
        self.tracks = {}
        self.counter = 0
        self.ttl_s = ttl_s

    def update(self, patches, timestamp):
        self.tracks = {
            key: item for key, item in self.tracks.items()
            if 0 <= timestamp - item[0] <= self.ttl_s
        }
        pairs = []
        if patches and self.tracks:
            keys = list(self.tracks)
            old = [item[1] for item in self.tracks.values()]
            dt = np.maximum(.03, timestamp - np.array([item[0] for item in self.tracks.values()]))
            prediction_time = np.minimum(dt, 5.)[:, None]
            predicted_px = np.array([p.center_px for p in old]) + np.array([p.velocity_px_s for p in old]) * prediction_time
            current_px = np.array([p.center_px for p in patches])
            distances = np.linalg.norm(predicted_px[:, None, :] - current_px[None, :, :], axis=2)
            gate = np.minimum(48, 12 + 58 * dt)
            old_world = np.array([p.plane_xy is not None for p in old])
            new_world = np.array([p.plane_xy is not None for p in patches])
            possible = distances <= gate[:, None]
            if old_world.any() and new_world.any():
                rows, cols = np.flatnonzero(old_world), np.flatnonzero(new_world)
                predicted = np.array([old[i].plane_xy for i in rows])
                predicted += np.array([old[i].velocity for i in rows]) * prediction_time[rows]
                current = np.array([patches[i].plane_xy for i in cols])
                world_distances = np.linalg.norm(predicted[:, None, :] - current[None, :, :], axis=2)
                world_gate = np.minimum(.72, .19 + .065 * dt[rows])
                ix = np.ix_(rows, cols)
                distances[ix] = world_distances
                possible[ix] = world_distances <= world_gate[:, None]
            pairs = [(distances[row, col], keys[row], int(col)) for row, col in zip(*np.nonzero(possible))]
        matched_tracks, matched_patches = set(), set()
        for _, key, index in sorted(pairs):
            if key in matched_tracks or index in matched_patches:
                continue
            old_ts, old = self.tracks[key]
            patch = patches[index]
            dt = max(.03, timestamp - old_ts)
            if old.plane_xy is not None and patch.plane_xy is not None:
                v = np.subtract(patch.plane_xy, old.plane_xy) / dt
                patch.velocity = tuple(.55 * np.asarray(old.velocity) + .45 * np.clip(v, -.38, .38))
            v_px = np.subtract(patch.center_px, old.center_px) / dt
            patch.velocity_px_s = tuple(.52 * np.asarray(old.velocity_px_s) + .48 * np.clip(v_px, -240, 240))
            area_change = min(1., abs(float(np.log((patch.pixel_area + 1) / (old.pixel_area + 1)))))
            patch.texture_score = .70 * patch.texture_score + .30 * old.texture_score - .025 * area_change
            patch.visual_class = texture_class(patch.texture_score)
            patch.id = key
            matched_tracks.add(key)
            matched_patches.add(index)
        for patch in patches:
            if not patch.id:
                self.counter += 1
                patch.id = f"S{self.counter:03d}"
            self.tracks[patch.id] = (timestamp, patch)
        return patches


class VirtualCamera:
    def __init__(self, cfg):
        self.cfg = cfg
        fx, fy, cx, cy = cfg.camera.intrinsics
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        if cfg.camera.device_to_camera:
            self.extrinsic = np.asarray(cfg.camera.device_to_camera)[:3]
        else:
            eye = np.asarray(cfg.camera.position_m)
            forward = np.subtract(cfg.camera.look_at_m, eye)
            forward /= np.linalg.norm(forward)
            right = np.array((1., 0, 0)) - forward[0] * forward
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            rotation = np.array([right, down, forward])
            self.extrinsic = np.column_stack((rotation, -rotation @ eye))

    def project(self, world):
        camera = np.asarray(world) @ self.extrinsic[:, :3].T + self.extrinsic[:, 3]
        pix = camera @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = np.clip(pix[..., :2] / pix[..., 2:3], -10000, 10000)
        return np.where((pix[..., 2] > .07)[..., None], pixels, -10000)

    def image_to_plane(self, tilt):
        bath = bath_frame(self.cfg.tank, tilt)
        world = bath.world([[0., 0.], [1., 0.], [0., 1.]])
        basis = np.column_stack((world[1] - world[0], world[2] - world[0], world[0]))
        homography = self.K @ (self.extrinsic[:, :3] @ basis + np.column_stack((np.zeros((3, 2)), self.extrinsic[:, 3])))
        if abs(np.linalg.det(homography)) < 1e-10:
            return None
        return np.linalg.inv(homography)


def polygon_points(mask, minimum=8., eps=.8, max_count=32):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return [
        cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist()
        for c in contours[:max_count] if cv2.contourArea(c) >= minimum
    ]


def mask_regions(mask, minimum=12., eps=1., max_count=90):
    contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    tree = hierarchy[0]
    roots = [i for i, links in enumerate(tree) if links[3] < 0 and cv2.contourArea(contours[i]) >= minimum]
    regions = []
    for i in sorted(roots, key=lambda index: -cv2.contourArea(contours[index]))[:max_count]:
        rings = [cv2.approxPolyDP(contours[i], eps, True).reshape(-1, 2).tolist()]
        child = tree[i][2]
        while child >= 0:
            if cv2.contourArea(contours[child]) >= minimum:
                rings.append(cv2.approxPolyDP(contours[child], eps, True).reshape(-1, 2).tolist())
            child = tree[child][0]
        regions.append(rings)
    return regions


def texture_class(score):
    return "团聚纹理" if score >= .5 else "薄膜纹理"


def patch_components(mask, gray, matrix=None, confidence_map=None, minimum=6, brightness=101):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    patches = []
    for i in sorted(range(1, count), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:70]:
        x, y, width, height, pixels = map(int, stats[i])
        if pixels < minimum:
            break
        component = labels[y:y + height, x:x + width] == i
        contours, _ = cv2.findContours(np.uint8(component), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = cv2.approxPolyDP(max(contours, key=cv2.contourArea), .65, True).reshape(-1, 2)
        if len(polygon) < 3:
            polygon = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        polygon = polygon + (x, y)
        px_center = tuple(centers[i])
        values = gray[y:y + height, x:x + width][component]
        curve = polygon.astype(np.float32)
        perimeter = cv2.arcLength(curve, True)
        compactness = min(1., 4 * np.pi * cv2.contourArea(curve) / max(1., perimeter ** 2))
        visual = .5 + (brightness - values.mean()) / 120 + (values.std() - 13) / 100 + (.60 - compactness) * .13
        texture_score = float(np.clip(visual, 0., 1.))
        confidence = .86 if confidence_map is None else float(confidence_map[y:y + height, x:x + width][component].mean())
        local_center, local_poly, area = None, None, None
        if matrix is not None:
            mapped = cv2.perspectiveTransform(np.float32([polygon]), matrix)[0]
            center = cv2.perspectiveTransform(np.float32([[px_center]]), matrix)[0, 0]
            if np.all(np.isfinite(mapped)) and np.max(np.abs(mapped)) < 40:
                local_center = tuple(map(float, center))
                local_poly = mapped.tolist()
                # Contour area underestimates small pixel islands. The correction converts the
                # contour's pixel-centre convention to pixel coverage.
                correction = min(3.0, pixels / max(1., cv2.contourArea(np.float32(polygon))))
                area = float(cv2.contourArea(mapped) * correction)
        patches.append(Patch(px_center, polygon.tolist(), pixels, local_center, local_poly, area,
                             texture_class(texture_score), confidence, texture_score))
    return patches


def virtual_masks(frame):
    b, g, r = (channel.astype(np.int16) for channel in cv2.split(frame))
    steam = (b > 117) & (g > 119) & (r > 119) & (r - b < 31) & (np.abs(g - b) < 23)
    tool = (b > 92) & (b - r > 37) & (g - r > 20)
    liquid = (r > 172) & (g - b > 33) & (r - g > 38)
    slag = (r > 75) & (r < 194) & (b - g > 11) & (r - g > 30) & ~tool
    rim = (b >= 52) & (b <= 113) & (r - b >= 9) & (r - b <= 33) & (g - b < 18)
    return steam, tool, liquid, slag, rim


def synthetic_surface(frame, matrix, cfg, masks=None):
    steam, tool, iron, slag, _ = virtual_masks(frame) if masks is None else masks
    pixels = np.full(frame.shape[:2], UNKNOWN, dtype=np.uint8)
    pixels[iron] = 0
    density = (VIRTUAL_GREEN - frame[..., 1][slag].astype(np.float32)) / VIRTUAL_CONTRAST
    pixels[slag] = np.rint(np.clip(density, cfg.slag.coverage_threshold, 1.) * MAX_VALUE).astype(np.uint8)
    pixels[steam | tool] = UNKNOWN
    return project_field(pixels, matrix, cfg.tank.inner_radii_m, cfg.slag.coverage_threshold, "concentration")


def synthetic_observation(image, matrix, cfg, tracker, index, stamp, fault=None):
    h, w = image.shape[:2]
    obs = Observation(index, stamp, (w, h))
    if fault == "offline":
        obs.reason = "视频流中断，等待新帧"
        return obs
    frame = cv2.medianBlur(image, 3)
    masks = virtual_masks(frame)
    steam, tool, liquid, slag, rim = masks
    warm = liquid | slag
    surface = cv2.morphologyEx(np.uint8(warm | steam), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footprint = np.zeros((h, w), np.uint8)
    if contours:
        cv2.drawContours(footprint, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 1, -1)
    footprint_area = int(np.count_nonzero(footprint))
    obs.steam_fraction = float(np.count_nonzero(steam & (footprint > 0)) / max(1, footprint_area))
    obs.mouth = polygon_points(rim, 50, 2.2, 2)
    obs.liquid = polygon_points(footprint, 60, 2.2, 1)
    obs.iron = polygon_points(liquid, 15, 1.3, 16)
    obs.slag_regions = mask_regions(slag & (footprint > 0), 25, 1.4)
    obs.iron_regions = mask_regions(liquid, 20, 1.4)
    obs.tool = polygon_points(tool, 10, 1.4, 3)
    if obs.tool:
        obs.tool_center_px = tuple(np.mean(obs.tool[0], axis=0))
    obs.steam = polygon_points(steam, 22, 2.0, 8)
    obs.surface = synthetic_surface(frame, matrix, cfg, masks)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    obs.slag = patch_components(slag & (footprint > 0), gray, matrix, brightness=54)
    obs.confidence = min(.97, .94 * (1 - .92 * obs.steam_fraction)) if footprint_area > w * h * .06 else .28
    if fault == "low":
        obs.confidence = min(obs.confidence, .21)
    if fault == "stale":
        obs.reason = "帧时间戳过期，等待新帧"
    elif obs.steam_fraction >= cfg.camera.steam_limit:
        obs.reason = "蒸汽遮挡超过阈值"
    elif obs.confidence < cfg.camera.min_confidence or matrix is None:
        obs.reason = "识别可信度不足或投影无效"
    obs.usable = not obs.reason
    obs.slag = (tracker if obs.usable else Tracker()).update(obs.slag, stamp)
    return obs

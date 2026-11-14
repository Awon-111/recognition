"""Versioned, validated configuration shared by the API and the evaluator."""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
Source = Literal["仿真假设", "企业提供", "现场标定"]
Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Tank(Strict):
    source: Source
    length_m: float = Field(gt=1, le=12)
    width_m: float = Field(gt=1, le=12)
    height_m: float = Field(gt=1, le=8)
    trunnion_span_m: float = Field(gt=0, le=15)
    pivot_m: Vec3
    inner_radii_m: Vec2
    rim_radii_m: Vec2
    bottom_scale: float = Field(gt=0.2, lt=1)
    rim_z_m: float
    liquid_z_m: float
    outlet_x_m: float
    outlet_width_m: float = Field(gt=0)
    outlet_extension_m: float = Field(gt=0, le=1.5)
    outlet_drop_m: float = Field(default=0., ge=0, le=.60)

    @model_validator(mode="after")
    def geometry(self):
        for inner, rim, outside in zip(self.inner_radii_m, self.rim_radii_m, (self.length_m / 2, self.width_m / 2)):
            if not 0.3 < inner <= rim < outside:
                raise ValueError("内腔半径须小于外壁半径，液面半径须不大于罐口半径")
        if not self.rim_z_m - self.height_m < self.liquid_z_m < self.rim_z_m - .12:
            raise ValueError("液面须在内腔中且低于罐口至少 0.12 m")
        if abs(self.outlet_x_m) + self.outlet_width_m / 2 >= self.rim_radii_m[0]:
            raise ValueError("出渣边须处于罐口宽度内")
        if self.rim_z_m - self.outlet_drop_m <= self.liquid_z_m:
            raise ValueError("直立状态的出渣口须高于液面")
        return self


class Axis(Strict):
    min: float
    max: float
    speed: float = Field(ge=.001, le=1000)
    initial: float

    @model_validator(mode="after")
    def range(self):
        if not self.min < self.max or not self.min <= self.initial <= self.max:
            raise ValueError("轴初始位置须位于有效行程内")
        return self


class Box(Strict):
    name: str = Field(min_length=1, max_length=32)
    min_m: Vec3
    max_m: Vec3

    @model_validator(mode="after")
    def bounds(self):
        if any(a >= b for a, b in zip(self.min_m, self.max_m)):
            raise ValueError("禁入区每轴的上界须大于下界")
        return self


class Arm(Strict):
    source: Source
    base_m: Vec3
    boom_z_m: float
    tool_size_m: Vec3
    collision_margin_m: float = Field(gt=0, le=.25)
    skim_depth_m: float = Field(default=.014, ge=.003, le=.027)
    extension: Axis
    lift: Axis
    swing: Axis
    pitch: Axis
    safety_boxes: list[Box] = Field(max_length=12)

    @property
    def axes(self):
        return self.extension, self.lift, self.swing, self.pitch

    @model_validator(mode="after")
    def limits(self):
        if not .05 <= self.extension.min < self.extension.max <= 30 or self.boom_z_m <= self.base_m[2] + self.lift.max:
            raise ValueError("伸缩行程须位于 0.05–30 m，横梁须位于升降上限之上")
        if not -6 <= self.lift.min < self.lift.max <= 20:
            raise ValueError("升降行程须位于 -6–20 m")
        if any(not .025 <= size <= 3 for size in self.tool_size_m):
            raise ValueError("扒渣头尺寸须在 0.025–3 m 内")
        if not -160 <= self.swing.min < self.swing.max <= 160 or not -70 <= self.pitch.min < self.pitch.max <= 70:
            raise ValueError("回转范围限于 ±160°，俯仰范围限于 ±70°")
        return self


class Tilt(Strict):
    source: Source
    initial_deg: float
    min_deg: float = Field(ge=-20)
    max_deg: float = Field(le=30)
    speed_deg_s: float = Field(ge=.01, le=90)
    step_deg: float = Field(gt=0, le=12)
    allow_planner: bool
    working_deg: float | None = None

    @model_validator(mode="after")
    def limits(self):
        if not self.min_deg <= self.initial_deg <= self.max_deg or self.min_deg > self.max_deg:
            raise ValueError("初始倾角须位于允许倾角内")
        if self.working_deg is not None and not max(.1, self.min_deg) <= self.working_deg <= self.max_deg:
            raise ValueError("作业倾角须向出渣口倾翻并位于允许范围内")
        return self


class PlaneCalibration(Strict):
    video_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,40}$")
    image_size: tuple[int, int]
    image_to_plane: tuple[Vec3, Vec3, Vec3]
    rms_m: float = Field(ge=0)
    tilt_deg: float = Field(ge=-20, le=30)

    @model_validator(mode="after")
    def matrix(self):
        if min(self.image_size) < 2 or abs(np.linalg.det(self.image_to_plane)) < 1e-12:
            raise ValueError("标定图像尺寸或平面变换无效")
        return self


class Camera(Strict):
    source: Source
    resolution: tuple[int, int]
    fps: float = Field(ge=2, le=60)
    position_m: Vec3
    look_at_m: Vec3
    intrinsics: Vec4
    distortion: tuple[float, float, float, float, float]
    device_to_camera: tuple[Vec4, Vec4, Vec4, Vec4] | None
    video_plane: PlaneCalibration | None
    max_frame_age_s: float = Field(ge=.12, le=5)
    steam_limit: float = Field(gt=0, lt=1)
    min_confidence: float = Field(gt=0, lt=1)
    feedback_s: float = Field(ge=0, le=3)
    noise: float = Field(ge=0, le=30)

    @model_validator(mode="after")
    def dimensions(self):
        w, h = self.resolution
        fx, fy, cx, cy = self.intrinsics
        if not (240 <= w <= 1920 and 180 <= h <= 1200 and 0 < fx and 0 < fy and 0 <= cx < w and 0 <= cy < h):
            raise ValueError("相机分辨率、焦距或主点超出有效范围")
        direction = np.subtract(self.look_at_m, self.position_m)
        if np.linalg.norm(direction) < .1 or np.linalg.norm(np.cross(direction, (1, 0, 0))) < .01:
            raise ValueError("相机安装位置与朝向不能退化")
        if self.device_to_camera is not None:
            matrix = np.array(self.device_to_camera)
            rotation = matrix[:3, :3]
            if not (
                np.allclose(matrix[3], (0, 0, 0, 1))
                and np.allclose(rotation @ rotation.T, np.eye(3), atol=.002)
                and np.isclose(np.linalg.det(rotation), 1, atol=.002)
            ):
                raise ValueError("外参须为有效的设备到相机刚体变换")
        return self


class Slag(Strict):
    source: Source
    distribution: Literal["工况定义", "均匀", "聚集"]
    grid_step_m: float = Field(ge=.025, le=.14)
    area_scale: float = Field(gt=.2, le=2)
    drift_m_s: float = Field(ge=0, le=.4)
    aggregation: float = Field(ge=0, le=1)
    disturbance_shift_m: float = Field(ge=0, le=.7)
    contact_efficiency: float = Field(gt=0, le=1)
    initial_film: float = Field(default=.56, ge=.12, le=.95)
    coverage_threshold: float = Field(default=.075, ge=.015, le=.10)
    skim_fraction: float = Field(default=.58, ge=.10, le=.72)


class Weights(Strict):
    removal: float = Field(gt=0, le=20)
    iron: float = Field(ge=0, le=20)
    disturbance: float = Field(ge=0, le=20)
    time: float = Field(ge=0, le=20)
    future: float = Field(ge=0, le=20)


class Objectives(Strict):
    source: Source
    removal_target: float = Field(ge=.50, lt=1)
    time_limit_s: float = Field(ge=5, le=300)
    iron_limit: float = Field(gt=0, le=100)
    weights: Weights
    recognition_budget_s: float = Field(ge=0, le=2)
    planning_budget_s: float = Field(ge=0, le=5)
    hold_retry_s: float = Field(ge=.1, le=3)


class Config(Strict):
    name: str = Field(min_length=1, max_length=80)
    equipment_class: Literal["360t"]
    tank: Tank
    arm: Arm
    tilt: Tilt
    camera: Camera
    slag: Slag
    objectives: Objectives

    @property
    def revision(self):
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class Scenario(Strict):
    id: str
    name: str
    description: str
    split: Literal["development", "evaluation"]
    seed: int
    layout: Literal["scattered", "clustered", "band"]
    clusters: int
    radius_m: float
    drift_factor: float
    steam: float
    steam_window_s: tuple[float, float] | None = None
    tilt_after: int | None = None
    shift_after: int | None = None


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def default_config():
    return Config.model_validate_json((ROOT / "config/default.json").read_bytes())


def scenarios():
    return {row["id"]: Scenario.model_validate(row) for row in json.loads(
        (ROOT / "config/scenarios.json").read_text(encoding="utf-8")
    )}


def config_path(revision):
    if len(revision) != 12 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("配置版本格式无效")
    return RUNTIME / "configs" / f"{revision}.json"


def load_config(revision=None):
    if revision is None or revision == default_config().revision:
        return default_config()
    return Config.model_validate_json(config_path(revision).read_bytes())


def save_config(config):
    write_json(config_path(config.revision), config.model_dump(mode="json"))
    return config.revision

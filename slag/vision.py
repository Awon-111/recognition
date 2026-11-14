"""YOLO semantic inference and per-frame display geometry for recorded footage."""

import os
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import cv2
import numpy as np

from .config import RUNTIME

CLASSES = ("slag", "iron", "tool", "mouth", "steam", "unknown")
SLAG, IRON, TOOL, MOUTH, STEAM, UNKNOWN = range(len(CLASSES))
MODEL_NAME = "YOLO26s-sem"
MODEL_SPEC = "yolo26s-sem-v1"


@lru_cache(maxsize=1)
def yolo_package():
    directory = RUNTIME / "yolo" / "settings"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(directory))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    return YOLO


@dataclass
class Segmentation:
    labels: np.ndarray
    confidence: np.ndarray
    arm_tip: tuple[float, float] | None
    region: np.ndarray


def _surface_region(labels):
    surface = np.uint8((labels == SLAG) | (labels == IRON) | (labels == MOUTH))
    closed = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(surface)
    if contours:
        areas = [cv2.contourArea(contour) for contour in contours]
        # A long shaft can split the bath into two material components.
        minimum = max(80, .23 * max(areas))
        sections = [contour for contour, area in zip(contours, areas) if area >= minimum]
        if sections:
            cv2.fillConvexPoly(region, cv2.convexHull(np.vstack(sections)), 1)
    return region.astype(bool)


def _filter_tool(labels):
    """Keep the supported tool silhouette and return its display point."""
    mask = np.uint8(labels == TOOL)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    height = labels.shape[0]
    # The recorded rake enters from the lower edge. Short detached marks at the rim are often reflections.
    _, top, _, spans, area = stats[1:].T
    valid = (area >= max(60, labels.size * .0005)) & (spans >= .075 * height) & (top + spans >= .50 * height)
    largest = 1 + np.argmax(np.where(valid, area, 0)) if np.any(valid) else -1
    labels[(components > 0) & (components != largest)] = UNKNOWN
    if largest < 1:
        return None
    y = int(stats[largest, cv2.CC_STAT_TOP])
    height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    silhouette = components[y:y + height] == largest
    widths = silhouette.sum(axis=1)
    rows = silhouette[:max(1, round(height * .32))]
    shaft_width = np.median(widths[height // 2:])
    if shaft_width > 0 and rows.sum(axis=1).max() > 3.5 * shaft_width:
        # When the rim touches the tool mask, use the lower shaft to bound the wide end.
        yy, xx = np.nonzero(silhouette[height // 2:])
        slope, intercept = np.polyfit(yy + height // 2, xx, 1)
        ahead = np.arange(len(rows))
        center = intercept + slope * ahead
        taper = np.clip((ahead / height - .075) / .15, 0., 1.)
        radius = max(10., .72 * shaft_width) * (1 + 1.3 * taper)
        spill = rows & (np.abs(np.arange(labels.shape[1])[None, :] - center[:, None]) > radius[:, None])
        upper_labels = labels[y:y + len(rows)]
        upper_labels[spill] = UNKNOWN
        rows[spill] = False
    head_widths = rows.sum(axis=1)
    if not head_widths.any():
        return None
    plausible = np.where(head_widths <= max(26, 3.1 * shaft_width), head_widths, 0)
    tip = int(plausible.argmax() if plausible.any() else head_widths.argmax())
    return float(np.flatnonzero(rows[tip]).mean()), float(y + tip)


class YoloModel:
    def __init__(self, checkpoint, manifest, device=None):
        import torch

        self.torch = torch
        self.manifest = manifest
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(min(os.cpu_count() or 1, 6))
        model = yolo_package()(str(checkpoint), task="semantic")
        if model.task != "semantic" or not all(name in model.names.values() for name in CLASSES[:3]):
            raise ValueError("YOLO 权重不含残渣、铁水和机械臂分割类别")
        self.lookup = np.array([CLASSES.index(model.names[index]) if model.names[index] in CLASSES else UNKNOWN
                                for index in range(len(model.names))], dtype=np.int8)
        self.model = model.model.eval().fuse(verbose=False).to(self.device)
        self.fp16 = self.device.type == "cuda"
        if self.fp16:
            self.model.half()
        self.image_size = manifest["image_size"]
        self.lock = Lock()

    def _logits(self, image, shape):
        torch = self.torch
        from torch.nn.functional import interpolate

        h, w = image.shape[:2]
        size = self.image_size
        scale = min(size / w, size / h)
        ww, hh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (size - ww) // 2, (size - hh) // 2
        resized = cv2.resize(image, (ww, hh), interpolation=cv2.INTER_LINEAR)
        boxed = cv2.copyMakeBorder(resized, top, size - hh - top, left, size - ww - left,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = np.ascontiguousarray(boxed[..., ::-1].transpose(2, 0, 1)[None])
        with self.lock, torch.inference_mode():
            batch = torch.from_numpy(rgb).to(self.device, torch.float16 if self.fp16 else torch.float32)
            logits = self.model(batch.div_(255))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits = interpolate(logits.float(), (size, size), mode="bilinear", align_corners=False)
            logits = logits[..., top:top + hh, left:left + ww]
            return interpolate(logits, shape, mode="bilinear", align_corners=False)[0]

    def _predict(self, image, shape):
        with self.torch.inference_mode():
            logits = self._logits(image, shape)
            confidence, indices = logits.softmax(dim=0).max(dim=0)
            return self.lookup[indices.cpu().numpy()], confidence.cpu().numpy()

    def segment(self, image, gray):
        labels, confidence = self._predict(image, gray.shape)
        labels[confidence < .42] = UNKNOWN
        no_signal = (cv2.blur(gray, (7, 7)) <= 4) & ((labels == SLAG) | (labels == IRON))
        labels[no_signal] = UNKNOWN
        tip = _filter_tool(labels)
        region = _surface_region(labels)
        if not self.manifest["classes"]["steam"] and np.any(region):
            intensity = gray.astype(np.float32)
            local = cv2.blur(intensity, (13, 13))
            variance = cv2.blur(intensity * intensity, (13, 13)) - local * local
            candidate = (variance < 44) & (gray > 147) & (gray < 225) & region & (labels == UNKNOWN)
            steam = cv2.morphologyEx(np.uint8(candidate), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)) > 0
            labels[steam] = STEAM
        confidence[labels == UNKNOWN] = 0.
        return Segmentation(labels, confidence, tip, region)

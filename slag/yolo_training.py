"""Train the project's YOLO semantic segmenter from LabelMe polygons."""

import hashlib
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np

from .config import write_json
from .recording import enhance_frame
from .training import CLASSES, _annotation_rows, _archive_index, _image_from_annotation, _polygons, work_area
from .vision import yolo_package

BASE_WEIGHTS = {
    "yolo26s-sem-ade20k.pt": "feb2bd47eed6f721a4588213c670ccdad69c0ba29b4b5f97892df222b18c6c3a",
    "yolo26s-sem.pt": "bc3e2152329831303de83e1af91d4b57d547e8794de8bb6edb1f2a622d1d5435",
}
BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
REQUIRED = ("slag", "iron", "tool")
BACKGROUND = "background"
IMAGE_SIZE = 512
EPOCHS = 64
REFINEMENT_EPOCHS = 20
EXPOSURE_EPOCHS = 12
EXPOSURE_CLASS_WEIGHT = .95
POLYGON_ORDER = ("mouth", "iron", "slag", "tool", "steam", "unknown")


def checkpoint_is_complete(path):
    return path.is_file() and path.stat().st_size > 1_000_000 and is_zipfile(path)


def pretrained_weights(folder):
    from ultralytics.utils.downloads import safe_download

    for name, expected in BASE_WEIGHTS.items():
        path = folder / name
        if not checkpoint_is_complete(path):
            path.unlink(missing_ok=True)
            try:
                safe_download(BASE_URL + name, file=path, min_bytes=1_000_000, retry=2, unzip=False)
            except (OSError, ConnectionError):
                continue
        if checkpoint_is_complete(path):
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest == expected:
                return path
            path.unlink()
    raise RuntimeError("无法下载官方 YOLO26s-sem 预训练权重")


def render_polygons(obj, source_size, target_size, ids, fill, *, resolve_nested=False):
    w, h = source_size
    out_w, out_h = target_size
    mask = np.full((out_h, out_w), fill, np.uint8)
    scale = out_w / w, out_h / h
    polygons = list(_polygons(obj, w, h))
    if resolve_nested:
        # Smaller material islands inside an outer contour take precedence over that outer contour.
        materials = {"iron", "slag"}

        def precedence(item):
            label, points, _ = item
            if label in materials:
                return 1, -cv2.contourArea(points.astype(np.float32))
            return POLYGON_ORDER.index(label), 0

        polygons.sort(key=precedence)
    else:
        polygons.sort(key=lambda item: POLYGON_ORDER.index(item[0]))
    for label, points, _ in polygons:
        color = ids.get(label)
        if color is not None:
            cv2.fillPoly(mask, [np.int32(np.rint(points * scale))], color)
    return mask


def label_background(image, mask, class_id):
    """Only label background well away from both the vessel cue and annotated pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marked = mask != 255
    envelope = work_area(gray) | np.uint8(marked)
    locations = cv2.findNonZero(envelope)
    if locations is None:
        return
    hull = np.zeros_like(envelope)
    cv2.fillConvexPoly(hull, cv2.convexHull(locations), 1)
    distance = cv2.distanceTransform(1 - hull, cv2.DIST_L2, 3)
    safe = (distance >= max(15, image.shape[1] * .060)) & (gray < 28)
    mask[safe & ~marked] = class_id


def _source_counts(paths):
    counts = Counter()
    for path in paths:
        with ZipFile(path) as archive:
            for _, obj, _ in _annotation_rows(archive, _archive_index(archive)):
                counts.update(label for label, _, _ in _polygons(obj, obj["imageWidth"], obj["imageHeight"]))
    return counts


def prepare_dataset(work, paths, notify, *, resolve_nested=False, mixed_seed=None):
    cv2.setNumThreads(1)
    counts = _source_counts(paths)
    missing = [name for name in REQUIRED if not counts[name]]
    if missing:
        raise ValueError("训练录像缺少分割标注 " + "、".join(missing))
    names = [name for name in CLASSES if counts[name]] + [BACKGROUND]
    ids = {name: index for index, name in enumerate(names)}
    images = work / "images" / "train"
    masks = work / "masks" / "train"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    pixels = Counter()
    frames = 0
    exposure_rng = random.Random(mixed_seed) if mixed_seed is not None else None
    enhanced_frames = 0
    for source, path in enumerate(paths):
        notify(f"正在生成训练掩码 · 录像 {path.stem}")
        with ZipFile(path) as archive:
            for entry, obj, image_file in _annotation_rows(archive, _archive_index(archive)):
                image = _image_from_annotation(archive, entry, obj, image_file)
                h, w = image.shape[:2]
                shrink = min(1., 640 / max(w, h))
                target = max(2, round(w * shrink)), max(2, round(h * shrink))
                mask = render_polygons(obj, (w, h), target, ids, 255, resolve_nested=resolve_nested)
                if np.all(mask == 255):
                    continue
                image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
                label_background(image, mask, ids[BACKGROUND])
                if exposure_rng is None or exposure_rng.random() < .5:
                    image = enhance_frame(image)
                    enhanced_frames += 1
                stem = f"{source:02d}-{frames:05d}"
                cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])[1].tofile(str(images / f"{stem}.jpg"))
                cv2.imencode(".png", mask, [cv2.IMWRITE_PNG_COMPRESSION, 2])[1].tofile(str(masks / f"{stem}.png"))
                histogram = np.bincount(mask.ravel(), minlength=256)
                pixels.update({label: int(histogram[index]) for label, index in ids.items()})
                frames += 1
    if not frames or any(not pixels[label] for label in REQUIRED):
        raise ValueError("转换后缺少可用的训练掩码")
    import yaml
    data = work / "dataset.yaml"
    data.write_text(yaml.safe_dump({
        "path": work.resolve().as_posix(),
        "train": "images/train",
        # The training framework needs a val path. Keep the held-out recordings outside it.
        "val": "images/train",
        "masks_dir": "masks",
        "names": names,
    }, sort_keys=False), encoding="utf-8")
    stats = {"names": names, "frames": frames, "pixels": dict(pixels), "polygons": dict(counts),
             "enhanced_frames": enhanced_frames}
    write_json(work / "conversion.json", stats)
    return data, stats


def fit_weights(work: Path, data: Path, seed: int, notify, *, initial=None, epochs=EPOCHS,
                refinement=False, class_weight=None):
    YOLO = yolo_package()
    import torch

    if initial is None:
        notify("获取官方 YOLO 预训练权重")
        initial = pretrained_weights(work)
    device = 0 if torch.cuda.is_available() else "cpu"
    batch = 8 if device == 0 else 4
    if device == "cpu":
        torch.set_num_threads(6)
    name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    stage = "微调" if refinement else "训练"
    notify(f"使用 {name} {stage} YOLO26s-sem · {epochs} 轮")
    model = YOLO(str(initial), task="semantic")

    def epoch_finished(trainer):
        if trainer.epoch < epochs:
            notify(f"YOLO {stage} {trainer.epoch + 1}/{epochs} 轮")

    model.add_callback("on_fit_epoch_end", epoch_finished)
    model.train(
        data=str(data), epochs=epochs, imgsz=IMAGE_SIZE, batch=batch, device=device,
        workers=0, cache="ram", project=str(work), name="fit", exist_ok=True,
        seed=seed, deterministic=False, amp=device == 0, val=False, plots=False,
        patience=0, optimizer="AdamW", lr0=.00085 if refinement else .0016, lrf=.07, cos_lr=True,
        warmup_epochs=2., cls_pw=class_weight if class_weight is not None else (.85 if refinement else .7),
        hsv_h=0., hsv_s=0., hsv_v=.53 if refinement else .42, degrees=5., translate=.06, scale=.20,
        fliplr=.5, flipud=0., mosaic=0., mixup=0., copy_paste=0.,
        verbose=False, save=True,
    )
    checkpoint = Path(model.trainer.last)
    if not checkpoint_is_complete(checkpoint):
        raise RuntimeError("YOLO 训练未生成有效权重")
    return checkpoint, initial.name, name

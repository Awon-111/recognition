"""LabelMe inspection, held-out evaluation and the active YOLO model."""

import base64
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from shutil import copyfile
from tempfile import TemporaryDirectory
from threading import RLock
from zipfile import ZipFile

import cv2
import numpy as np

from .config import ROOT, RUNTIME, write_json
from .vision import CLASSES, MODEL_NAME, MODEL_SPEC, YoloModel

LABELS = dict(zip(CLASSES, ("残渣", "铁水", "机械臂", "罐口", "蒸汽", "无法判断")))
ALIASES = {"liquid": "iron", "work_area": "mouth", "arm": "tool", "扒渣头": "tool", **{v: k for k, v in LABELS.items()}}
PREVIEW_MAX_WIDTH = 416
PREVIEW_MAX_HEIGHT = 360
TRAINING_RECIPE = "mixed-exposure-v2"
ZIP_SIZE_LIMIT = 950_000_000
ASSET = ROOT / "assets"
_registry_lock = RLock()
_cached_model = None


def video_files():
    paths = {}
    for folder in (ASSET / "videos", RUNTIME / "videos"):
        if folder.exists():
            paths.update({p.stem: p for p in folder.iterdir() if p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv", ".webm")})
    return paths


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("视频无法解码")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not .1 <= fps <= 240 or count < 1 or w < 2 or h < 2:
            raise ValueError("需要带有效帧率、尺寸与帧数的录像文件")
        return {"id": path.stem, "name": path.name, "fps": fps, "frames": count,
                "duration_s": count / fps, "size": [w, h]}
    finally:
        cap.release()


def label_files():
    datasets = {path.stem: path for path in (ASSET / "labels").glob("*.zip")}
    datasets.update({path.stem: path for path in (RUNTIME / "labels").glob("*.zip")})
    return datasets


def furnace_for(path):
    metadata = path.with_suffix(".meta.json")
    if metadata.is_file():
        return json.loads(metadata.read_text(encoding="utf-8")).get("furnace_id")
    return None


def _archive_index(archive):
    items = archive.infolist()
    if len(items) > 16000 or sum(i.file_size for i in items) > ZIP_SIZE_LIMIT:
        raise ValueError("标注 ZIP 展开体积或文件数超限")
    names = {}
    for item in items:
        name = PurePosixPath(item.filename.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts or item.file_size > 45_000_000:
            raise ValueError("标注 ZIP 含无效文件路径或超大文件")
        if not item.is_dir():
            if name in names:
                raise ValueError("标注 ZIP 含重复路径")
            names[name] = item
    return names


def _annotation_rows(archive, names):
    for path in sorted(names):
        if path.suffix.lower() != ".json":
            continue
        try:
            obj = json.loads(archive.read(names[path]).decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取标注 {path.name}") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("shapes"), list):
            continue
        rel = PurePosixPath(str(obj.get("imagePath") or "").replace("\\", "/")).name
        img_path = path.parent / rel
        if not obj.get("imageData") and (not rel or img_path not in names):
            img_path = next((path.with_suffix(suffix) for suffix in (".jpg", ".png", ".jpeg", ".JPG")
                             if path.with_suffix(suffix) in names), None)
            if img_path is None:
                raise ValueError(f"{path.name} 缺少对应抽帧图片")
        yield path, obj, names.get(img_path)


def _polygons(obj, width, height):
    for shape in obj["shapes"]:
        if not isinstance(shape, dict) or not isinstance(shape.get("label"), str):
            raise ValueError("形状须包含有效 label")
        raw = shape["label"].strip().lower()
        label = ALIASES.get(raw, raw)
        if label not in CLASSES or shape.get("shape_type", "polygon") not in ("polygon", "rectangle"):
            continue
        try:
            points = np.asarray(shape["points"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{raw} 的轮廓点格式无效") from exc
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)) or len(points) > 10000:
            raise ValueError(f"{raw} 的轮廓点无效")
        if shape.get("shape_type") == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            points = np.float32([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
        if len(points) < 3:
            raise ValueError(f"{raw} 需要至少三个轮廓点")
        yield label, np.clip(points, [0, 0], [width - 1, height - 1]), shape.get("group_id")


def inspect_dataset(path, vid=None, check_images=False):
    labels, raw_labels = Counter(), Counter()
    frames = linked = tracks = 0
    issues = []
    with ZipFile(path) as archive:
        names = _archive_index(archive)
        for entry, obj, image_file in _annotation_rows(archive, names):
            w, h = obj.get("imageWidth"), obj.get("imageHeight")
            if not isinstance(w, int) or not isinstance(h, int) or min(w, h) < 2:
                raise ValueError(f"{entry.name} 缺少有效的图片宽高")
            if check_images:
                _image_from_annotation(archive, entry, obj, image_file)
            for label, _, group in _polygons(obj, w, h):
                labels[label] += 1
                if label == "slag" and group is not None:
                    tracks += 1
            raw_labels.update(str(s.get("label", "")) for s in obj["shapes"])
            frames += 1
            match = re.fullmatch(r"frame_(\d+)_(\d+(?:\.\d+)?)s", entry.stem)
            if match and vid:
                index, sec = int(match[1]), float(match[2])
                if index < vid["frames"] and abs(index / vid["fps"] - sec) <= max(.03, 1 / vid["fps"]):
                    linked += 1
    if not frames or not labels:
        raise ValueError("ZIP 内没有可训练的 LabelMe 图片标注")
    missing = [LABELS[key] for key in CLASSES if not labels[key]]
    if missing:
        issues.append("缺少类别 " + "、".join(missing))
    if tracks < labels["slag"]:
        issues.append("部分或全部渣块无跨帧 group_id，跟踪为在线估计")
    if linked < frames:
        issues.append("部分抽帧文件未能按文件名与视频时间对齐")
    return {"id": path.stem, "name": path.name, "frames": frames, "aligned": linked,
            "slag_track_annotations": tracks, "labels": dict(labels), "raw_labels": dict(raw_labels),
            "issues": issues}


def preview_gray(image):
    h, w = image.shape[:2]
    scale = min(1., PREVIEW_MAX_WIDTH / w, PREVIEW_MAX_HEIGHT / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 3)
    return resized, gray


def work_area(gray):
    """Adapt the existing footage pipeline's bright envelope to the preview dimensions."""
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = max(25, min(65, np.percentile(smooth, 85) * .60))
    visible = np.uint8(smooth > threshold)
    visible[:round(h * .065)] = 0
    visible[:, :round(w * .075)] = 0
    visible[:, round(w * .95):] = 0
    visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(visible)
    if contours:
        largest = max(cv2.contourArea(c) for c in contours)
        members = [c for c in contours if cv2.contourArea(c) >= max(w * h * .006, largest * .12)]
        if members:
            cv2.fillConvexPoly(result, cv2.convexHull(np.vstack(members)), 1)
    return result


def _image_from_annotation(archive, entry, obj, image_file):
    try:
        image_bytes = (archive.read(image_file) if image_file else base64.b64decode(obj["imageData"], validate=True))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{entry.name} 图片数据损坏") from exc
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{entry.name} 的图片无法解码")
    h, w = image.shape[:2]
    if (w, h) != (obj["imageWidth"], obj["imageHeight"]):
        raise ValueError(f"{entry.name} 声明的图片宽高与实际尺寸不同")
    return image


def _pixel_metrics(video_id, confusion, scope):
    count = int(confusion.sum())
    return {
        "video_id": video_id, "sampled_pixels": count, "accuracy": float(np.trace(confusion) / max(1, count)),
        "scope": scope,
        "classes": {
            label: {
                "support": int(confusion[j].sum()),
                "iou": float(confusion[j, j] / (confusion[j].sum() + confusion[:, j].sum() - confusion[j, j]))
                if confusion[j].sum() else None,
                "recall": float(confusion[j, j] / confusion[j].sum()) if confusion[j].sum() else None,
            } for j, label in enumerate(CLASSES)
        },
        "confusion": confusion.tolist(),
    }


def _model_root():
    return ASSET / "models"


def _manifest(version):
    if not version or not re.fullmatch(r"[a-f0-9]{12}", version):
        return None
    folder = _model_root() / version
    manifest = folder / "manifest.json"
    if not manifest.is_file() or not (folder / "weights.pt").is_file():
        return None
    item = json.loads(manifest.read_text(encoding="utf-8"))
    return item if item.get("algorithm") == MODEL_SPEC and item.get("version") == version else None


def model_versions():
    with _registry_lock:
        reference = _model_root() / "active.json"
        if reference.is_file():
            item = _manifest(json.loads(reference.read_text(encoding="utf-8")).get("version"))
            if item:
                return [item]
    return []


def active_model():
    versions = model_versions()
    return versions[0]["version"] if versions else None


def load_model(version=None):
    global _cached_model
    with _registry_lock:
        models = model_versions()
        if not models:
            raise ValueError("没有可用的 YOLO 模型，请先完成训练")
        manifest = models[0]
        active = manifest["version"]
        if version is not None and version != active:
            raise ValueError("该模型版本已停用")
        if _cached_model is None or _cached_model.manifest["version"] != active:
            from .recording import enhance_frame

            model = YoloModel(_model_root() / active / "weights.pt", manifest)
            probe = np.zeros((480, 640, 3), np.uint8)
            _, gray = preview_gray(probe)
            model.segment(enhance_frame(probe), gray)
            _cached_model = model
        return _cached_model


def _evaluate(datasets, validation_ids, model, notify):
    from .recording import enhance_frame
    from .yolo_training import render_polygons

    rows = []
    classes = len(CLASSES)
    ids = {label: index for index, label in enumerate(CLASSES, 1)}
    for key in validation_ids:
        notify(f"正在验证独立录像 {key}")
        confusion = np.zeros((classes, classes), dtype=np.int64)
        frames = 0
        with ZipFile(datasets[key]) as archive:
            for entry, obj, image_file in _annotation_rows(archive, _archive_index(archive)):
                image = _image_from_annotation(archive, entry, obj, image_file)
                _, gray = preview_gray(image)
                h, w = gray.shape
                truth = render_polygons(obj, image.shape[:2][::-1], (w, h), ids, 0, resolve_nested=True)
                known = truth > 0
                actual = truth[known].astype(np.int32) - 1
                predicted = model.segment(enhance_frame(image), gray).labels[known]
                confusion += np.bincount(actual * classes + predicted, minlength=classes ** 2).reshape(classes, classes)
                frames += 1
        if not confusion.sum():
            raise ValueError(f"验证录像 {key} 缺少可计分的像素")
        rows.append(_pixel_metrics(key, confusion, "全部已标注预览像素") | {"frames": frames})
    return rows


def _training_split(validation_ids, datasets=None):
    if datasets is None:
        datasets = label_files()
    validation_ids = sorted(set(validation_ids))
    if not validation_ids or not set(validation_ids) < datasets.keys():
        raise ValueError("请选择完整的独立验证录像，同时至少保留一段训练录像")
    training_ids = sorted(datasets.keys() - set(validation_ids))
    furnaces = {key: furnace_for(path) for key, path in datasets.items()}
    shared = ({furnaces[key] for key in training_ids} & {furnaces[key] for key in validation_ids}) - {None}
    if shared:
        raise ValueError("以下炉次横跨训练和验证，请将同炉次录像一并划分 " + "、".join(sorted(shared)))
    return datasets, training_ids, validation_ids


def _training_version(datasets, training_ids, validation_ids, seed):
    digest = hashlib.sha256(f"{MODEL_SPEC}:{TRAINING_RECIPE}:{seed}:{training_ids}:{validation_ids}".encode())
    for key, path in sorted(datasets.items()):
        with path.open("rb") as handle:
            digest.update(key.encode())
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()[:12]


def evaluate_weights(checkpoint, validation_ids, seed, stats, pretrained, device, epochs, image_size,
                     notify, comparison=None, *, datasets=None):
    datasets, training_ids, validation_ids = _training_split(validation_ids, datasets)
    manifest = {
        "version": _training_version(datasets, training_ids, validation_ids, seed),
        "algorithm": MODEL_SPEC, "name": MODEL_NAME, "image_size": image_size,
        "seed": seed, "training_ids": training_ids, "validation_ids": validation_ids,
        "classes": {key: key in stats["names"] for key in CLASSES},
        "training_frames": stats["frames"], "trained_pixels": sum(stats["pixels"].values()),
        "training_pixels_by_class": stats["pixels"], "pretrained": pretrained,
        "training_device": device, "epochs": epochs,
        "input_enhancement": "CLAHE",
        "training_recipe": TRAINING_RECIPE,
        "enhanced_training_frames": stats["enhanced_frames"],
    }
    model = YoloModel(checkpoint, manifest)
    manifest["validation"] = _evaluate(datasets, validation_ids, model, notify)
    with checkpoint.open("rb") as handle:
        manifest["weights_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    if comparison:
        manifest["comparison"] = comparison
    return manifest


def publish_weights(checkpoint, manifest):
    global _cached_model
    with _registry_lock:
        root = _model_root()
        root.mkdir(parents=True, exist_ok=True)
        version = manifest["version"]
        reference = root / "active.json"
        previous = json.loads(reference.read_text(encoding="utf-8")).get("version") if reference.is_file() else None
        if version == previous and _manifest(version):
            return
        folder = root / version
        with TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
            staged = Path(temporary)
            copyfile(checkpoint, staged / "weights.pt")
            write_json(staged / "manifest.json", manifest)
            folder.mkdir(exist_ok=True)
            for name in ("weights.pt", "manifest.json"):
                (staged / name).replace(folder / name)
            write_json(reference, {"version": version})
        _cached_model = None


def train(validation_ids, seed=360, notify=None):
    from .yolo_training import (
        EPOCHS, EXPOSURE_CLASS_WEIGHT, EXPOSURE_EPOCHS, IMAGE_SIZE, REFINEMENT_EPOCHS,
        fit_weights, prepare_dataset,
    )

    datasets, training_ids, validation_ids = _training_split(validation_ids)
    notify = notify or (lambda _: None)
    version = _training_version(datasets, training_ids, validation_ids, seed)
    current = model_versions()
    if current and current[0]["version"] == version:
        notify("同一标注划分和种子已有独立录像验证结果")
        return current[0]
    workspace = RUNTIME / "yolo"
    workspace.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="training-", dir=workspace) as temporary:
        work = Path(temporary)
        sources = [datasets[key] for key in training_ids]
        data, _ = prepare_dataset(work / "dataset", sources, notify)
        checkpoint, pretrained, device = fit_weights(work, data, seed, notify)
        refinement = work / "refined"
        data, _ = prepare_dataset(refinement / "dataset", sources, notify, resolve_nested=True)
        checkpoint, _, _ = fit_weights(refinement, data, seed, notify, initial=checkpoint,
                                       epochs=REFINEMENT_EPOCHS, refinement=True)
        exposure = work / "exposure"
        data, stats = prepare_dataset(exposure / "dataset", sources, notify, resolve_nested=True, mixed_seed=seed)
        checkpoint, _, _ = fit_weights(exposure, data, seed, notify, initial=checkpoint,
                                       epochs=EXPOSURE_EPOCHS, refinement=True, class_weight=EXPOSURE_CLASS_WEIGHT)
        manifest = evaluate_weights(checkpoint, validation_ids, seed, stats, pretrained, device,
                                    EPOCHS + REFINEMENT_EPOCHS + EXPOSURE_EPOCHS, IMAGE_SIZE, notify)
        publish_weights(checkpoint, manifest)
        return manifest

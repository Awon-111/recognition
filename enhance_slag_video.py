"""Export an enhanced segment from a constant-frame-rate camera recording."""

import argparse
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2


def enhance_frame(frame, clahe):
    """Blend CLAHE on Lab luminance without changing the frame dimensions."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    lab[:, :, 0] = cv2.addWeighted(light, .70, clahe.apply(light), .30, 0.)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def export_clip(source, output, start=0., end=None):
    if source.resolve() == output.resolve():
        raise ValueError("输出必须与源录像使用不同文件名")
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开录像 {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or total <= 0 or w <= 0 or h <= 0:
            raise ValueError("录像缺少有效的帧率、帧数或尺寸")
        duration = total / fps
        end = duration if end is None else end
        if not 0 <= start < end <= duration + 1e-7:
            raise ValueError(f"切片范围必须位于 0–{duration:.3f} 秒")
        first, stop = math.ceil(start * fps - 1e-7), min(total, math.ceil(end * fps - 1e-7))
        if first >= stop:
            raise ValueError("该切片范围没有视频帧")
        if first and not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError(f"无法定位源录像第 {first} 帧")
        output.parent.mkdir(parents=True, exist_ok=True)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        with TemporaryDirectory(prefix="slag-", dir=output.parent) as temporary:
            draft = Path(temporary) / "clip.mp4"
            command = [
                "ffmpeg", "-loglevel", "error", "-nostdin",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{w}x{h}",
                "-framerate", str(fps), "-i", "pipe:0", "-c:v", "libx264",
                "-threads", "3", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p" if w % 2 == h % 2 == 0 else "yuv444p",
                "-movflags", "+faststart", draft.name,
            ]
            with subprocess.Popen(command, cwd=temporary, stdin=subprocess.PIPE) as encoder:
                try:
                    for index in range(first, stop):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"无法读取源录像第 {index} 帧")
                        encoder.stdin.write(enhance_frame(frame, clahe).tobytes())
                finally:
                    encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError("FFmpeg 视频编码失败")
            draft.replace(output)
        return stop - first, fps
    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    frames, fps = export_clip(args.source, args.output, args.start, args.end)
    print(f"{args.output} · {frames} frames · {fps:g} fps")

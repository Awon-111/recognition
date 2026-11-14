"""Evaluate the recorded planning pipeline on one observation per video second."""

import argparse
from collections import Counter
from math import ceil
from pathlib import Path
from time import perf_counter

from .config import default_config, write_json
from .recording import Replay, plan_recording
from .training import load_model, video_info


def evaluate_video(path, cfg, model):
    info = video_info(path)
    video_id = path.stem.rsplit("-", 1)[-1]
    info["id"] = video_id
    replay = Replay(path, cfg, model, 0., analyze=True, info=info)
    rows = []
    try:
        for index in range(ceil(info["duration_s"])):
            deadline = min(index + 1., info["duration_s"])
            replay.catch_up(ceil(index * replay.fps - 1e-7))
            start = perf_counter()
            sample = replay.read()
            if sample is None:
                break
            obs = sample.observation
            row = {"start_s": index, "deadline_s": deadline, "frame": obs.frame,
                   "time_s": obs.captured_s, "tool_visible": bool(obs.tool), "tool_center_px": obs.tool_center_px,
                   "confidence": obs.confidence, "surface_basis": sample.details["reconstruction"]["basis"],
                   "timing_ms": sample.details["timing_ms"]}
            if sample.planning_surface is None:
                row.update(status="hold", reason=sample.details["planning_reason"], total_ms=(perf_counter() - start) * 1000)
            else:
                plan = plan_recording(sample, cfg, "balanced", frame_started=start,
                                      source_started_s=obs.captured_s, deadline_s=deadline)
                candidate = next((item for item in plan["candidates"] if item["id"] == plan["selected_id"]), None)
                status = "ready" if candidate else "no_path" if plan["timely"] else "late"
                row.update(status=status, reason=plan["reason"], total_ms=plan["total_ms"],
                           planning_ms=plan["planning_ms"], ready_at_s=plan["ready_at_s"], selected=candidate,
                           candidate_count=len(plan["candidates"]),
                           valid_count=sum(item["valid"] for item in plan["candidates"]))
            rows.append(row)
    finally:
        replay.close()
    return {"video": str(path), "fps": info["fps"], "duration_s": info["duration_s"],
            "status_counts": dict(Counter(row["status"] for row in rows)), "segments": rows}


def main():
    parser = argparse.ArgumentParser(description="用原视频逐秒复核录像分割与规划耗时")
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = default_config()
    load_started = perf_counter()
    model = load_model()
    load_ms = (perf_counter() - load_started) * 1000
    results = []
    for path in args.videos:
        result = evaluate_video(path.resolve(), cfg, model)
        print(path.name, result["status_counts"], flush=True)
        results.append(result)
    write_json(args.output, {"model_version": model.manifest["version"], "config_revision": cfg.revision,
                             "strategy": "balanced", "interval_s": 1, "model_load_ms": load_ms, "videos": results})


if __name__ == "__main__":
    main()

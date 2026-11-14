"""Batch evaluation entry point and machine-readable result generation."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, load_config, scenarios, write_json
from .engine import Run
from .planner import STRATEGIES


def evaluate(cfg, strategy_ids, notify=None):
    tests = [case for case in scenarios().values() if case.split == "evaluation"]
    rows = []
    for strategy in strategy_ids:
        for case in tests:
            if notify:
                notify(f"{len(rows) + 1}/{len(strategy_ids) * len(tests)} · {STRATEGIES[strategy]} · {case.name}")
            run = Run(cfg, case, case.seed, strategy)
            while not run.terminal:
                run.advance()
            row = run.report()
            row["steps"] = [
                {
                    "index": event["index"], "kind": event["kind"], "start_s": event["start_s"],
                    "end_s": event["end_s"], "note": event["note"],
                    "action": event["action"], "actual": event["actual"],
                    "frame_before": event["before"]["observation"]["frame"], "frame_after": event["after"]["observation"]["frame"],
                } for event in run.records
            ]
            rows.append(row)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "三维简化仿真",
        "config_revision": cfg.revision, "config": cfg.model_dump(mode="json"),
        "surface_model": "continuous-film",
        "reference": {"time_s_lte": 120, "area_removal_gt": .98}, "field_status": "现场待验证",
        "results": rows,
    }


def as_markdown(report):
    rows = report["results"]
    passed = sum(row["reference_passed"] for row in rows)
    lines = [
        "# 铁水罐扒渣闭环仿真评测", "",
        f"配置版本 `{report['config_revision']}` · 固定验收线 ≤ 120 秒、面积去除率 > 98% · 达标 {passed}/{len(rows)}",
        "",
        "各工况从近满面渣膜开始，先倾罐，再贴面回拉并越出渣口。覆盖面积按统一参考网格中超过浓度阈值的单元面积统计。浓度与渣量指数为仿真归一化量。120 秒与 98% 仍按原面积验收线判定。",
        "时间包含识别、规划、倾罐及工具运动、等待反馈与安全保持。设备参数为仿真假设，现场待验证。",
        "",
        "| 工况 | 种子 | 策略 | 用时 s | 初始覆盖 m² | 剩余覆盖 m² | 面积去除率 | 渣量指数下降 | 排渣轮数 | 保持 | 固定验收 | 停止原因 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']['name']} | {r['seed']} | {STRATEGIES[r['strategy']]} | {r['elapsed_s']:.3f} | "
            f"{r['initial_m2']:.4f} | {r['remaining_m2']:.4f} | {r['removal_rate']:.2%} | "
            f"{r['load_removal_rate']:.2%} | {r['strokes']} | {r['holds']} | "
            f"{'达标' if r['reference_passed'] else '未达标'} | {r['stop_reason']} |"
        )
    lines += [
        "", "## 首轮排渣前后", "",
        "| 工况 | 策略 | 覆盖率 | 平均浓度指数 | 首轮带出渣量指数 |",
        "|---|---|---|---|---:|",
    ]
    for r in rows:
        first = r["first_stroke"]
        if first:
            lines.append(f"| {r['case']['name']} | {STRATEGIES[r['strategy']]} | "
                         f"{first['coverage_before']:.2%} → {first['coverage_after']:.2%} | "
                         f"{first['density_before']:.3f} → {first['density_after']:.3f} | {first['removed_load']:.3f} |")
    lines += [
        "", "## 时间分项", "",
        "| 工况 | 策略 | 识别 s | 规划 s | 设备运动 s | 等反馈 s | 保持 s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t = r["timings"]
        lines.append(f"| {r['case']['name']} | {STRATEGIES[r['strategy']]} | {t['recognition']:.3f} | {t['planning']:.3f} | "
                     f"{t['motion']:.3f} | {t['feedback']:.3f} | {t['holding']:.3f} |")
    lines += [
        "", "完整 JSON 包含各段轨迹、耗时、预测与仿真真值变化、模拟铁损、失败原因、配置和残余网格。",
        "调试工况使用独立种子，未纳入验收统计。", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="运行六项固定验收工况")
    parser.add_argument("--strategy", action="append", choices=STRATEGIES, help="可重复指定策略")
    parser.add_argument("--revision", help="已保存的配置版本")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/acceptance.json")
    args = parser.parse_args()
    report = evaluate(load_config(args.revision), args.strategy or ["balanced"], notify=lambda p: print(p, flush=True))
    write_json(args.output, report)
    args.output.with_suffix(".md").write_text(as_markdown(report), encoding="utf-8")
    print(f"报告已写入 {args.output}")


if __name__ == "__main__":
    main()

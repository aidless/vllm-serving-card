#!/usr/bin/env python3
"""从 card.json 自动推导 SLO 并发上限档位。

判据（全部门槛显式可配，输出每档的判定依据，绝不拍脑袋）：
- 交互式档：TTFT p99 ≤ --ttft-p99-ms（默认 500ms）且 TPOT p99 ≤ --tpot-p99-ms（默认 100ms）
- 流式可容忍档：TTFT p99 ≤ 2000ms 且 TPOT p99 ≤ 200ms
- 吞吐批处理档：其余（无交互约束，按吞吐排序）

用法：
  python3 analyze_slo.py --card results/card.json [--ttft-p99-ms 500 --tpot-p99-ms 100]
输出：results/slo.json + 控制台表格。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = ["rows"]


def classify(rows: list[dict], ttft_p99_ms: float, tpot_p99_ms: float) -> list[dict]:
    out = []
    for r in sorted(rows, key=lambda x: x.get("max_concurrency") or x.get("concurrency")):
        tc = r.get("ttft_p99_ms")
        tp = r.get("tpot_p99_ms")
        if tc is None or tp is None:
            tier = "UNKNOWN"
            why = "缺 p99 TTFT 或 TPOT 载体"
        elif tc <= ttft_p99_ms and tp <= tpot_p99_ms:
            tier = "INTERACTIVE"
            why = f"TTFT p99 {tc:.1f}ms<= {ttft_p99_ms} 且 TPOT p99 {tp:.1f}ms<= {tpot_p99_ms}"
        elif tc <= 2000 and tp <= 200:
            tier = "STREAM_OK"
            why = f"TTFT p99 {tc:.1f}ms<= 2000 且 TPOT p99 {tp:.1f}ms<= 200"
        else:
            tier = "THROUGHPUT"
            why = f"TTFT p99 {tc:.1f}ms 或 TPOT p99 {tp:.1f}ms 超交互门槛"
        c = r.get("max_concurrency") or r.get("concurrency")
        out.append({"max_concurrency": c, "tier": tier, "why": why,
                    "output_throughput": r.get("output_throughput") or r.get("output_tok_per_s"),
                    "ttft_p99_ms": tc, "tpot_p99_ms": tp})
    return out


def summarize(tiers: list[dict]) -> dict:
    inter = [t["max_concurrency"] for t in tiers if t["tier"] == "INTERACTIVE"]
    return {
        "interactive_max_concurrency": max(inter) if inter else 0,
        "interactive_tiers": inter,
        "stream_ok": [t["max_concurrency"] for t in tiers if t["tier"] == "STREAM_OK"],
        "throughput_only": [t["max_concurrency"] for t in tiers if t["tier"] == "THROUGHPUT"],
        "unknown": [t["max_concurrency"] for t in tiers if t["tier"] == "UNKNOWN"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", default=str(Path(__file__).resolve().parent / "results" / "card.json"))
    ap.add_argument("--ttft-p99-ms", type=float, default=500.0)
    ap.add_argument("--tpot-p99-ms", type=float, default=100.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "slo.json"))
    args = ap.parse_args()

    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    for k in REQUIRED:
        if k not in card:
            print(f"card.json 缺字段 {k}，fail-closed", file=sys.stderr)
            return 2
    tiers = classify(card["rows"], args.ttft_p99_ms, args.tpot_p99_ms)
    summary = summarize(tiers)
    result = {"thresholds": {"ttft_p99_ms": args.ttft_p99_ms, "tpot_p99_ms": args.tpot_p99_ms},
              "tiers": tiers, "summary": summary}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'并发':>4}  {'档位':<12} {'输出 tok/s':>10}  依据")
    for t in tiers:
        tp = f"{t['output_throughput']:.2f}" if t["output_throughput"] is not None else "—"
        print(f"{t['max_concurrency']:>4}  {t['tier']:<12} {tp:>10}  {t['why']}")
    print(f"\n交互式并发上限 = {summary['interactive_max_concurrency']}（门槛 TTFT p99<={args.ttft_p99_ms}ms、TPOT p99<={args.tpot_p99_ms}ms）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

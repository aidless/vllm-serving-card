#!/usr/bin/env python3
"""把 vllm bench 的原始 JSON + nvidia-smi 功率采样，折成可引用的数字卡片。

所有数字都从 results/ 的原始文件重算；缺字段直接报错退出（不猜、不补）。
"""
from __future__ import annotations

import datetime
import json
import statistics
import sys
from pathlib import Path

CARD_DIR = Path(__file__).resolve().parent
RES = CARD_DIR / "results"

# 成本口径（唯一假设，写进卡片）：只算电，¥0.6/kWh；不含硬件折旧/机房/网络。
ELECTRICITY_CNY_PER_KWH = 0.6

REQUIRED = [
    "max_concurrency", "request_throughput", "output_throughput",
    "total_token_throughput", "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p99_tpot_ms", "mean_e2el_ms", "p99_e2el_ms",
    "completed", "failed", "duration", "total_output_tokens",
]


def read_power(path: Path, window: tuple[datetime.datetime, datetime.datetime] | None = None) -> dict:
    """nvidia-smi 采样 CSV：timestamp, power.draw, memory.used, utilization.gpu

    window = (start, end) 时只取请求窗口内的样本。功率采样器在 bench 客户端启动前就开始，
    那段是导入/编译的空闲态（~28W），把它算进「每 token 能耗」会把高并发行低估，
    所以必须用 bench 结果里的 date + duration 把窗口切出来。
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            rows.append((ts, float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if not rows:
        raise SystemExit(f"{path} 里没有可解析的功率采样")
    idle = [r[1] for r in rows if window and r[0] < window[0]]
    sel = [r for r in rows if window is None or window[0] <= r[0] <= window[1]]
    if len(sel) < 3:
        raise SystemExit(f"{path} 在请求窗口内只有 {len(sel)} 个采样，无法算功率")
    powers = [r[1] for r in sel]
    mems = [r[2] for r in sel]
    utils = [r[3] for r in sel]
    return {
        "samples": len(sel),
        "power_mean_w": round(statistics.mean(powers), 2),
        "power_p50_w": round(statistics.median(powers), 2),
        "power_max_w": round(max(powers), 2),
        "power_idle_w": round(statistics.mean(idle), 2) if idle else None,
        "mem_mean_mib": round(statistics.mean(mems), 1),
        "mem_max_mib": round(max(mems), 1),
        "util_mean_pct": round(statistics.mean(utils), 1),
    }


def main() -> int:
    files = sorted(RES.glob("bench-c*.json"),
                   key=lambda p: int(p.stem.removeprefix("bench-c")))
    if len(files) < 2:
        raise SystemExit("至少需要两个并发档的 bench-c*.json")
    rows = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in data]
        if missing:
            raise SystemExit(f"{fp.name} 缺字段: {missing}")
        conc = int(data["max_concurrency"])
        end = datetime.datetime.strptime(data["date"], "%Y%m%d-%H%M%S")
        start = end - datetime.timedelta(seconds=float(data["duration"]))
        power = read_power(RES / f"power-c{conc}.csv", (start, end))
        out_tps = float(data["output_throughput"])
        gpu_s_per_1m = 1e6 / out_tps
        kwh_per_1m = power["power_mean_w"] * gpu_s_per_1m / 3.6e6
        rows.append({
            "concurrency": conc,
            "completed": data["completed"],
            "failed": data["failed"],
            "duration_s": round(float(data["duration"]), 2),
            "req_per_s": round(float(data["request_throughput"]), 3),
            "output_tok_per_s": round(out_tps, 2),
            "total_tok_per_s": round(float(data["total_token_throughput"]), 2),
            "ttft_mean_ms": round(float(data["mean_ttft_ms"]), 1),
            "ttft_p50_ms": round(float(data["p50_ttft_ms"]), 1),
            "ttft_p99_ms": round(float(data["p99_ttft_ms"]), 1),
            "tpot_mean_ms": round(float(data["mean_tpot_ms"]), 2),
            "tpot_p99_ms": round(float(data["p99_tpot_ms"]), 2),
            "e2el_mean_ms": round(float(data["mean_e2el_ms"]), 1),
            "e2el_p99_ms": round(float(data["p99_e2el_ms"]), 1),
            "gpu_min_per_1m_out_tok": round(gpu_s_per_1m / 60, 1),
            "kwh_per_1m_out_tok": round(kwh_per_1m, 3),
            "cny_per_1m_out_tok": round(kwh_per_1m * ELECTRICITY_CNY_PER_KWH, 3),
            **{k: power[k] for k in ("power_mean_w", "power_p50_w", "power_max_w",
                                     "power_idle_w", "samples", "mem_max_mib", "util_mean_pct")},
        })
    rows.sort(key=lambda r: r["concurrency"])

    (RES / "card.json").write_text(json.dumps({
        "generated_from": [f.name for f in files],
        "electricity_cny_per_kwh": ELECTRICITY_CNY_PER_KWH,
        "power_window": "bench date - duration .. bench date（排除采样器启动时的空闲段）",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    hdr = ("| 并发 | 输出吞吐 tok/s | 总吞吐 tok/s | req/s | TTFT p50 / p99 (ms) | "
           "TPOT p99 (ms) | E2E p99 (ms) | 板功率均值 W | GPU·min / 1M out tok | 电费 ¥ / 1M out tok |")
    sep = "|" + "---|" * 10
    print(hdr)
    print(sep)
    for r in rows:
        print(f"| {r['concurrency']} | {r['output_tok_per_s']} | {r['total_tok_per_s']} | "
              f"{r['req_per_s']} | {r['ttft_p50_ms']} / {r['ttft_p99_ms']} | "
              f"{r['tpot_p99_ms']} | {r['e2el_p99_ms']} | {r['power_mean_w']} | "
              f"{r['gpu_min_per_1m_out_tok']} | {r['cny_per_1m_out_tok']} |")
    print(f"\ncard.json -> {RES / 'card.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

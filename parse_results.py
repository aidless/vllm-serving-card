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
    "mean_tpot_ms", "p99_tpot_ms",
    "completed", "failed", "duration", "total_output_tokens",
]
# e2el 可选：bench-cg-c4/c16 落盘时未含 e2el 字段（--percentile-metrics 未带 e2el），没有就不报、不猜
OPTIONAL = ["mean_e2el_ms", "p99_e2el_ms"]


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


def validate_required(data: dict) -> None:
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise SystemExit(f"缺字段: {missing}")


def compute_row(data: dict, power: dict) -> dict:
    """单档折算：bench JSON 字段 + 窗口功率 dict → 卡片一行的全部数字（纯函数，便于自检）。"""
    conc = int(data["max_concurrency"])
    out_tps = float(data["output_throughput"])
    gpu_s_per_1m = 1e6 / out_tps
    pw = power.get("power_mean_w")
    kwh_per_1m = pw * gpu_s_per_1m / 3.6e6 if pw is not None else None
    return {
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
        "e2el_mean_ms": round(float(data["mean_e2el_ms"]), 1) if "mean_e2el_ms" in data else None,
        "e2el_p99_ms": round(float(data["p99_e2el_ms"]), 1) if "p99_e2el_ms" in data else None,
        "gpu_min_per_1m_out_tok": round(gpu_s_per_1m / 60, 1) if pw is not None else None,
        "kwh_per_1m_out_tok": round(kwh_per_1m, 3) if kwh_per_1m is not None else None,
        "cny_per_1m_out_tok": round(kwh_per_1m * ELECTRICITY_CNY_PER_KWH, 3) if kwh_per_1m is not None else None,
        **{k: power[k] for k in ("power_mean_w", "power_p50_w", "power_max_w",
                                 "power_idle_w", "samples", "mem_max_mib", "util_mean_pct")},
    }


def guard_all_zero_p99(rows: list[dict]) -> None:
    """实测踩过的坑：--base-url 带 /v1 再拼 /v1/completions → 全 404 → 所有 P99=0 伪装成功。

    全零是「请求根本没打到模型」的信号，必须阻断而不是出一张零卡片。
    """
    keys = ("ttft_p99_ms", "tpot_p99_ms", "e2el_p99_ms")
    vals = [r[k] for r in rows for k in keys if r[k] is not None]
    if vals and all(v == 0 for v in vals):
        raise SystemExit(
            "三档 P99 全 0 —— 请求大概率没打到模型（典型原因：--base-url 带 /v1 又拼 /v1/xxx "
            "得到 404）。拒绝生成数字卡片。")


def selftest() -> int:
    """零依赖自检：GREEN（正常 fixture 算对）+ RED（缺字段 / 全零 P99 必须阻断）。"""
    import tempfile

    # ---- fixture：1 档 bench + 对应功率窗口，手算期望值 ----
    bench = {
        "date": "20261109-120001", "duration": 100.0, "max_concurrency": 4,
        "completed": 10, "failed": 0,
        "request_throughput": 0.1, "output_throughput": 25.0, "total_token_throughput": 75.0,
        "mean_ttft_ms": 400.0, "p50_ttft_ms": 380.0, "p99_ttft_ms": 500.0,
        "mean_tpot_ms": 40.0, "p99_tpot_ms": 50.0,
        "mean_e2el_ms": 8000.0, "p99_e2el_ms": 9000.0,
        "total_output_tokens": 2500,
    }
    power = {"power_mean_w": 60.0, "power_p50_w": 60.0, "power_max_w": 70.0,
             "power_idle_w": 28.0, "samples": 100, "mem_max_mib": 5500.0, "util_mean_pct": 95.0}
    # 手算：gpu_s_per_1m = 1e6/25 = 40000s；kwh = 60*40000/3.6e6 = 0.666667；¥ = 0.4
    row = compute_row(bench, power)
    assert row["output_tok_per_s"] == 25.0
    assert row["gpu_min_per_1m_out_tok"] == 666.7, row
    assert row["kwh_per_1m_out_tok"] == 0.667, row
    assert row["cny_per_1m_out_tok"] == 0.4, row
    print(f"[PASS] compute_row 正常 fixture：{row['cny_per_1m_out_tok']} ¥/1M out tok")

    # ---- RED 1：REQUIRED 缺字段必须 SystemExit ----
    bad = {k: v for k, v in bench.items() if k != "p99_ttft_ms"}
    try:
        validate_required(bad)
    except SystemExit as e:
        assert "p99_ttft_ms" in str(e)
        print(f"[PASS] 缺字段阻断：{e}")
    else:
        raise AssertionError("缺字段没有阻断")
    validate_required(bench)  # GREEN：不抛

    # ---- RED 2：read_power 空窗口/样本不足必须 SystemExit ----
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("2026/11/09 11:59:00.000, 28.0, 5000, 0\n")    # 窗口前：空闲段
        f.write("2026/11/09 12:00:01.000, 60.0, 5500, 95\n")   # 窗口内
        f.write("2026/11/09 12:00:30.000, 61.0, 5500, 95\n")   # 窗口内
        f.write("2026/11/09 12:03:00.000, 61.0, 5500, 95\n")   # 窗口后
        csv_path = Path(f.name)
    window = (datetime.datetime(2026, 11, 9, 12, 0, 1),
              datetime.datetime(2026, 11, 9, 12, 0, 1) + datetime.timedelta(seconds=100))
    try:
        read_power(csv_path, window)
    except SystemExit as e:
        assert "采样" in str(e)
        print(f"[PASS] 窗口样本不足阻断：{e}")
    else:
        raise AssertionError("样本不足没有阻断")
    good = read_power(csv_path, (datetime.datetime(2026, 11, 9, 11, 58, 0), window[1]))
    assert good["samples"] == 3 and good["power_mean_w"] == 49.67, good
    assert good["power_idle_w"] is None, "12:00 之前没有样本，power_idle 应为 None"
    print(f"[PASS] read_power 窗口过滤：{good['samples']} 条 / {good['power_mean_w']}W")

    # ---- RED 3：全零 P99（双 /v1 404 场景）必须阻断 ----
    zero = dict(bench)
    zero.update({"mean_ttft_ms": 0.0, "p50_ttft_ms": 0.0, "p99_ttft_ms": 0.0,
                 "mean_tpot_ms": 0.0, "p99_tpot_ms": 0.0,
                 "mean_e2el_ms": 0.0, "p99_e2el_ms": 0.0,
                 "request_throughput": 0.0, "output_throughput": 0.0000001})
    try:
        zrow = compute_row(zero, power)
        guard_all_zero_p99([zrow])
    except SystemExit as e:
        assert "/v1" in str(e)
        print(f"[PASS] 全零 P99 阻断（双 /v1 404 伪装成功）：{str(e)[:40]}…")
    else:
        raise AssertionError("全零 P99 没有阻断")
    guard_all_zero_p99([row])  # GREEN：正常行不抛

    # ---- GREEN：e2el 字段缺失（bench-cg-c4/c16 实况）→ None，不阻断 ----
    noe2el = {k: v for k, v in bench.items() if k not in OPTIONAL}
    nrow = compute_row(noe2el, power)
    assert nrow["e2el_p99_ms"] is None and nrow["output_tok_per_s"] == 25.0
    guard_all_zero_p99([nrow])
    print("[PASS] e2el 缺字段降级为 None（不猜数）")

    print(f"SELFTEST PASS: 7/7 通过")
    return 0


def main() -> int:
    eager = sorted(RES.glob("bench-c[0-9]*.json"),
                   key=lambda p: int(p.stem.removeprefix("bench-c")))
    graph = sorted(RES.glob("bench-cg-c[0-9]*.json"),
                   key=lambda p: int(p.stem.split("-c")[-1]))
    if len(eager) < 2:
        raise SystemExit("至少需要两个并发档的 bench-c<N>.json")
    rows = []
    for variant, files, power_glob in (("eager", eager, "power-c{conc}.csv"),
                                       ("graph", graph, "power-cg-c{conc}.csv")):
        for fp in files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            validate_required(data)
            conc = int(data["max_concurrency"])
            end = datetime.datetime.strptime(data["date"], "%Y%m%d-%H%M%S")
            start = end - datetime.timedelta(seconds=float(data["duration"]))
            pw = RES / power_glob.format(conc=conc)
            if pw.exists():
                power = read_power(pw, (start, end))
                row = compute_row(data, power)
            else:
                # 功率证据未落盘：吞吐/延迟照报（bench json 是载体），成本列记 None 不猜数
                power = {k: None for k in ("power_mean_w", "power_p50_w", "power_max_w",
                                           "power_idle_w", "samples", "mem_max_mib", "util_mean_pct")}
                row = compute_row(data, power)
                row["cny_per_1m_out_tok"] = None
                row["kwh_per_1m_out_tok"] = None
                row["gpu_min_per_1m_out_tok"] = None
                row["power_evidence"] = None
            row["variant"] = variant
            rows.append(row)
    rows.sort(key=lambda r: (r["variant"] != "eager", r["concurrency"]))
    guard_all_zero_p99(rows)

    (RES / "card.json").write_text(json.dumps({
        "generated_from": [f.name for f in files],
        "electricity_cny_per_kwh": ELECTRICITY_CNY_PER_KWH,
        "power_window": "bench date - duration .. bench date（排除采样器启动时的空闲段）",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    hdr = ("| 档 | 输出吞吐 tok/s | 总吞吐 tok/s | req/s | TTFT p50 / p99 (ms) | "
           "TPOT p99 (ms) | E2E p99 (ms) | 板功率均值 W | GPU·min / 1M out tok | 电费 ¥ / 1M out tok |")
    sep = "|" + "---|" * 10
    print(hdr)
    print(sep)
    for r in rows:
        fmt = lambda v: "—" if v is None else v
        print(f"| {r['variant']}-{r['concurrency']} | {r['output_tok_per_s']} | {r['total_tok_per_s']} | "
              f"{r['req_per_s']} | {r['ttft_p50_ms']} / {r['ttft_p99_ms']} | "
              f"{r['tpot_p99_ms']} | {r['e2el_p99_ms']} | {fmt(r['power_mean_w'])} | "
              f"{fmt(r['gpu_min_per_1m_out_tok'])} | {fmt(r['cny_per_1m_out_tok'])} |")
    print(f"\ncard.json -> {RES / 'card.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())

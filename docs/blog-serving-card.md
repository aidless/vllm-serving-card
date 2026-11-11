# 6GB 笔记本显卡跑出 261 tok/s——代价是首 token 等 1.8 秒

> 中文稿（掘金 / V2EX / 知乎）。英文标题备选：
> *My 6 GB laptop GPU serves 261 tokens/second — if you can tolerate a 1.8-second first token*
> （HN / Reddit 用）。仓库：https://github.com/aidless/vllm-serving-card
> 姊妹篇（prefix-cache 的 TIL）：*My prefix cache never hit — because my prompt was shorter than one 16-token block*

## 先说反直觉的那个数

把并发从 1 拉到 16，同一张 RTX 3060 Laptop 6GB、同一个 Qwen3-4B-AWQ，
输出吞吐从 **19.99 → 261.41 tok/s（13.1×）**，每百万输出 token 的 GPU 电费
从 **¥0.441 → ¥0.043**（便宜 10.3×）。但首 token p99 从 **211.8 ms → 1796.0 ms**。
所以这不是「优化」，是一次取舍：吞吐和单价好了 13 倍，交互体验坏了 8.5 倍。
我把交互式 SLO 的并发上限定在 **4**（TTFT p99 481.2 ms），16 档留给批处理/流式。

## 配置与测法（可复算，不是截图）

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ~/models/Qwen3-4B-AWQ \
  --max-model-len 2048 --gpu-memory-utilization 0.90 --enforce-eager \
  --port 8123 --disable-log-requests
bash bench_tiers.sh   # 每并发档 48 条，input 256 / output 128，--ignore-eos，seed 42
python3 parse_results.py
```

| 并发 | 输出 tok/s | TTFT p50/p99 ms | TPOT p99 | E2E p99 | 功率 W | ¥/1M out |
|---|---|---|---|---|---|---|
| 1 | 19.99 | 131.2 / 211.8 | 65.06 | 8383 | 52.86 | 0.441 |
| 4 | 71.71 | 432.6 / 481.2 | 79.43 | 10357 | 52.59 | 0.122 |
| 16 | 261.41 | 1140.1 / 1796.0 | 61.78 | 8129 | 67.57 | 0.043 |

电费公式：`(窗口平均功率 W × 1e6 / 输出吞吐 tok/s) / 3.6e6 × ¥0.6/kWh`，
只算 GPU 电，不含整机/折旧/机房。功率 CSV 按 bench 结果 JSON 的
`[date − duration, date]` 切窗口——否则导入期空闲功耗会把 c=16 的均值
从 67.57 W 稀释到 45.28 W，成本低估三分之一。这个坑是 `parse_results.py --selftest`
里的一条反证钉住的。

**第二个杠杆是 CUDA graph**：去掉 `--enforce-eager` 后，同三档吞吐再
×4.0 / ×3.4 / ×2.1（c=16 到 534.82 tok/s），TPOT p99（c=1 档）65.06 → 12.18 ms；
代价是 KV cache 从 15,232 → 8,304 tokens（并发余量 7.44x → 4.05x），
`--gpu-memory-utilization 0.95` 直接 OOM。graph 档功率采样没落盘，
所以电费列**留空**——宁可留空也不拿别档的数补。

## 最能说明工程成熟度的一段：我先写预测，再跑

怕自己事后编解释，我先在 `docs/prediction-vs-measurement.md` 里写死 c=8 的预测
（输出 135 tok/s、TTFT p99 920 ms、TPOT p99 70 ms，误差带 ±15%），
才去跑 bench。实测 162.48 tok/s / 956.17 ms / 53.10 ms：10 格里 7 格命中，
超带的是吞吐（+20.4%）与 TPOT（−24%）。原因不是调度形状错了——
把原曲线乘一个全局因子后，c=8 的插值只差 +5.4%——而是**绝对速率本身就漂**。

于是做了第三次复测：同一个服务进程、同一配置、同一 seed，c=16 连跑三次得
291.41 / 318.45 / 278.44 tok/s；连同前两次，五次读数
**均值 289.65 ± 21.42（1σ，7.4%），极差 21.8%**。同步采的 `clocks.sm` 窗口均值
1634–1743 MHz、功率 65–76 W、温度 64–70 ℃ 与吞吐单调同向，
和「动态调频 / 功耗墙」相容；但 n=3 只有 2 个自由度，
**机制我标未证实**，锁频复测留给 v0.3。

所以这张卡正确的读法是：绝对值当量级参考，同一次 run 内的比值（13.1×、
CUDA graph 倍数）分子分母同源才敢引用。原卡 261.41 恰好是五次里最低的一次——
拿单次扫描做 SLO 承诺会系统性保守，拿某次高读数写简历则会虚高。

## 边界

单卡 6GB、2048 上下文（KV 只有 15,232 tokens，别外推到长文档）、
`--ignore-eos` 强制定长输出、eager 基线偏保守、客户端端到端计时（含排队）。
c=16 的功率窗口只有 23 个 1Hz 样本，不确定性明显高于 c=1 的 308 个样本，
这一条写在卡片 README 里，不藏在脚注。

（配图建议：`docs/trace-timeline.html` 之外，卡片的「三档表 + 预测/实测四列」
足够做首图；不要放 3D 柱状图。）

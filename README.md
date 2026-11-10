# vLLM 单卡服务实测卡片（Qwen3-4B-AWQ / RTX 3060 Laptop 6GB）

一句话结论：**把并发从 1 提到 16，输出吞吐 19.99 → 261.41 tok/s（13.1×），
每 token 电费 ¥0.441 → ¥0.043 / 1M 输出 token（10.3× 便宜），
而 TPOT p99 反而从 65.06 ms 降到 61.78 ms；代价是 TTFT p99 从 211.8 ms 涨到 1796.0 ms。
所以交互式 SLO 的并发上限定在 4（TTFT p99 481.2 ms），批处理/离线才吃 16 档。**

日期：2026-11-09。所有数字从 `results/` 原始文件重算，命令 `python3 parse_results.py`。

## 环境与启动

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 595.99.02 |
| 软件 | vLLM 0.33.0 / torch 2.14.0+cu130 / CUDA 13.2 |
| 模型 | `Qwen3-4B-AWQ`（4-bit AWQ，本地 `~/models/Qwen3-4B-AWQ`，2.5 GiB 权重） |
| 服务 | `--max-model-len 2048 --gpu-memory-utilization 0.90 --enforce-eager` |
| KV cache | 可用 2.09 GiB / 15,232 tokens；2048-token 请求的理论最大并发 7.44× |
| 采样 | `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99` |

```bash
# 起服务（本机无 nvcc，必须关掉 FlashInfer 采样器，否则它去 JIT 编 kernel 直接崩）
VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/vllm serve \
  /home/liuzewen/models/Qwen3-4B-AWQ \
  --served-model-name Qwen3-4B-AWQ --host 127.0.0.1 --port 8123 \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --enforce-eager --disable-log-requests

# 三档基准 + 每档 1 Hz 采 GPU 板功率
bash bench_tiers.sh
python3 parse_results.py          # 出表 + results/card.json
```

启动踩过的两个坑（复现时会撞上）：
1. `8000` 端口已被本机另一个 FastAPI 服务占着 → 用 `8123`。
2. `vllm bench serve` 不传本地 `--tokenizer` 会去 HuggingFace 拉 tokenizer 卡住；
   脚本里固定了本地路径 + `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
3. 首次请求有 JIT 冷启动（smoke 里 TTFT 5820 ms），所以每档加 `--num-warmups 4`，
   下表数字不含冷启动。

## 三档实测数字

每档 48 条定长请求（random dataset，输入 256 token / 输出强制 128 token，`--ignore-eos`，
`--seed 42`），三档都是 `completed 48 / failed 0`。

| 并发 | 输出吞吐 tok/s | 总吞吐 tok/s | req/s | TTFT p50 / p99 (ms) | TPOT p99 (ms) | E2E p99 (ms) | 板功率均值 W | GPU·min / 1M out tok | 电费 ¥ / 1M out tok |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 19.99 | 59.96 | 0.156 | 131.2 / 211.8 | 65.06 | 8383.0 | 52.86 | 833.9 | 0.441 |
| 4 | 71.71 | 215.13 | 0.560 | 432.6 / 481.2 | 79.43 | 10357.1 | 52.59 | 232.4 | 0.122 |
| 16 | 261.41 | 784.24 | 2.042 | 1140.1 / 1796.0 | 61.78 | 8128.9 | 67.57 | 63.8 | 0.043 |

`GPU·min / 1M out tok` = `1e6 / 输出吞吐 / 60`，即按该档吞吐产 100 万输出 token 要占多少 GPU 分钟：
833.9 min ≈ 13.9 GPU·h（c=1）→ 63.8 min ≈ 1.06 GPU·h（c=16）。

### 读数（不是推断，是上表相除）

- 吞吐 261.41 / 19.99 = **13.1×**；req/s 2.042 / 0.156 = **13.1×**；板功率 67.57 / 52.86 = **1.28×**。
  批量化几乎白拿吞吐，功率只涨 28%。
- TPOT p99 65.06 → 61.78 ms、E2E p99 8383.0 → 8128.9 ms：**decode 侧没有变慢**，
  排队主要发生在首 token 之前。
- TTFT p99 211.8 → 1796.0 ms（**8.5×**）：c=16 时 16 路 prefill 挤在一起，
  首 token 延迟是这份硬件上唯一明显恶化的指标。
- 电费 0.441 → 0.043 ¥/1M 输出 token（**10.3×**）；c=4 的 0.122 是中间档。

## 配置敏感性：enforce-eager vs CUDA graph（2026-11-09 增测）

同一硬件、同基准参数（48 条、256in/128out、ignore-eos、seed 42、warmup 4），
唯一差别是去掉 `--enforce-eager`（eager 基线；推荐配置 = 去掉该参数启用 CUDA graph，见下文配置敏感性）（vLLM 0.33 默认 `FULL_AND_PIECEWISE` CUDA graph +
inductor 编译）：

| 并发 | eager 输出 tok/s | graph 输出 tok/s | 加速 | eager TPOT p99 | graph TPOT p99 | eager E2E p99 | graph E2E p99 | eager ¥/1M out | graph ¥/1M out |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 19.99 | **80.19** | 4.0× | 65.06 ms | **12.18 ms** | 8383 ms | **1667 ms** | 0.441 | —¹ |
| 4 | 71.71 | **242.95** | 3.4× | 79.43 ms | **15.54 ms** | 10357 ms | —² | 0.122 | —¹ |
| 16 | 261.41 | **534.82** | 2.1× | 61.78 ms | **29.15 ms** | 8129 ms | —² | 0.043 | —¹ |

¹ graph 档功率采样只存在 /tmp、未随仓库落盘（2026-11-10 复核发现，违反本卡片
  「数字必有落盘载体」纪律，原 0.096/0.027/0.016 无载体已撤）。吞吐 ×2.1–4.0 意味着
  每 1M 输出 token 的 GPU 占用时间至少降同倍数；功率未落盘，成本不报数。
² `bench-cg-c4/c16.json` 落盘时未含 e2el 字段（`--percentile-metrics` 未带 e2el），
  载体没有就不报。c=1 的 1667 ms 有载体（bench-cg-c1.json）。

（c=1 与 c=4、c=16 的 graph 档各测一次，c=4/c=16 复测一次，两次吞吐 247.04/242.95、
546.46/534.82 tok/s，波动 <5%。graph 档成本窗口功率 40.0–51.5 W，口径与 eager 相同。）

**代价是 KV cache**：graph 捕获吃掉约 1.3 GiB——KV cache 从 15,232 tokens（7.44x 并发）
掉到 **8,304 tokens（4.05x）**。本基准 384 tokens/请求不受影响，但长上下文会先撞 KV。

**边界实测**：`--gpu-memory-utilization 0.95` + CUDA graph 在 6GB 卡上 **OOM**
（capture 时要 1.80 GiB、只剩 1.75 GiB；vLLM 的 graph-aware 显存预算把 0.95 折算成
等效 0.8482 仍不够）。0.90 + graph 是这张卡的上限配置。另：vLLM 0.33 的
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认开启，日志会显式给出等效利用率。

**结论**：6GB 卡上 serve Qwen3-4B-AWQ，CUDA graph 是第一杠杆——吞吐 ×2.1–4.0、
TPOT p99 降 4–5×，代价是 KV 减半。eager 只在显存极度受限（要给 KV 让路）时才值得。
（graph 电费本可按吞吐比同降，但功率证据未落盘，卡片纪律是不报无载体数字。）

## 成本口径（唯一假设，别外推）

`电费 ¥/1M 输出 tok = (窗口内平均板功率 W × 1e6/输出吞吐 秒) / 3.6e6 kWh × ¥0.6/kWh`。

- 只算 GPU 板级电力，**¥0.6/kWh 是假设电价**；不含整机 CPU/内存/散热、硬件折旧、机房与网络。
  这不是云厂商报价，只是「这台机器自己跑一度电多少钱」的量级。
- 功率采样用 `nvidia-smi -l 1`（1 Hz）。采样器在 bench 客户端启动前就开始，
  那段是导入/编译空闲态（约 28 W）；直接对整份 CSV 求均值会把 c=16 档平均功率
  从 67.57 W 稀释到 45.28 W，成本被低估约 1/3。`parse_results.py` 用 bench 结果里的
  `date - duration .. date` 把请求窗口切出来再求均值，窗口内采样数 308 / 85 / 23 条，
  与各档 duration 307.4 / 85.7 / 23.5 s 对齐。
- `card.json` 同时存了 `power_idle_w`、窗口内 `samples`、`util_mean_pct`，可复核。

## SLO 自动推导（analyze_slo.py，2026-11-10）

「交互式并发上限=4」不再手推——`python3 analyze_slo.py --card results/card.json`
按显式门槛（默认 TTFT p99≤500ms 且 TPOT p99≤100ms）逐档分类：
c=1 与 c=4（含 CUDA graph 档）均判 **INTERACTIVE**，c=16 判 STREAM_OK，
**交互式并发上限 = 4**，判定依据逐档落盘 `results/slo.json`。
门槛可用 `--ttft-p99-ms/--tpot-p99-ms` 调整；缺载体的档判 UNKNOWN 不编档，
card.json 缺 rows 直接 exit 2（fail-closed）。

## prefix-cache 观测（2026-11-10，定位后撤销）

曾报告「相同 prompt 永不命中 prefix cache」并报上游
[vllm#66786](https://github.com/vllm-project/vllm/issues/66786)——
**已自答关闭（误报）**。根因：复现 prompt 只有 13 tokens，vLLM 只哈希
完整 16-token block（`request_block_hasher` 的 "We only hash full blocks"），
不足一个 block 的 prompt 在 lookup 时 `block_hashes` 为空、永远不命中——
block 粒度缓存的设计行为，与 LBNHC layout 无关。同一服务换 ~72 token
相同 prompt ×2，第二次即命中 64 tokens（89%）。
对本卡结论无影响：`--ignore-eos` 随机定长输入本就不依赖 prefix cache；
真实服务若共享长系统提示词（>1 block），命中正常。

## 适用范围与已知限制

- 单卡 6GB、**单轮扫描**（每档一次，没有重复运行 → 没给置信区间），数字是这台机器这一次读数。
- `--enforce-eager`（eager 基线；推荐配置 = 去掉该参数启用 CUDA graph，见下文配置敏感性）：没开 CUDA graph，absolute 吞吐偏保守；`--ignore-eos` 强制每条恰好 128
  输出 token，真实业务变长输出会改变 prefill/decode 比例；random 输入使 prefix cache 命中率很低。
- `--max-model-len 2048`：长上下文场景 KV cache 只有 15,232 tokens，并发余量会缩水，别把 c=16 外推到长文档。
- TTFT/TPOT/E2E 都是**客户端端到端**（含网络与排队），不是引擎内部计时。

## 简历可用的一句话（带取舍，可当场展开）

> 单张 RTX 3060 Laptop 6GB 上用 vLLM 0.33 部署 Qwen3-4B-AWQ（KV cache 2.09 GiB /
> 15,232 tokens），用并发档扫描把输出吞吐从 19.99 提到 261.41 tok/s（13.1×），
> TPOT p99 维持 65.06→61.78 ms、电费口径 ¥0.441→¥0.043 / 1M 输出 token；
> 同时量到 TTFT p99 由 211.8 涨至 1796.0 ms，据此把交互式 SLO 的并发上限定为 4
> （TTFT p99 481.2 ms），16 档只给离线批处理。

## 复现与原始证据

```
vllm-serving-card/
├── README.md              # 本卡片
├── bench_tiers.sh         # 三档基准 + 功率采样（可重跑）
├── parse_results.py       # 原始 JSON/CSV → 本表 + card.json（缺字段直接退出，不猜）
└── results/
    ├── environment.txt    # GPU / vllm / torch 版本
    ├── models.json        # /v1/models 快照
    ├── bench-c{1,4,16}.json       # eager 档 vllm bench 原始结果（p50/p95/p99 全套）
    ├── bench-cg-c{1,4,16}.json    # graph 档原始结果（c4/c16 无 e2el 字段，见脚注 ²）
    ├── bench-cg95-c16.json        # util 0.95 + graph OOM 记录（无有效指标，仅留证）
    ├── bench-c{1,4,16}.log        # 原始终端输出
    ├── power-c{1,4,16}.csv        # 1 Hz 功率/显存/利用率采样（仅 eager 档）
    └── card.json                  # 本表所有数字的结构化副本（graph 成本=null）
```

服务日志在本机 `/tmp/vllm-serve.log`（KV cache、模型加载之类的事实来自这里）。

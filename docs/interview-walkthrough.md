# vllm-serving-card · 15 分钟面试 walk-through

目标读者：面推理/性能方向（L4 层）的面试官。全部数字可在仓库里离线重算
（`parse_results.py --selftest` + README「复现」节）。

## 0. 一句话

> 在一张 RTX 3060 Laptop 6GB 上把 Qwen3-4B-AWQ 用 vLLM 0.33 serve 起来，
> 用并发档扫描（1/4/16 × eager/CUDA graph）把输出吞吐从 20 拉到 535 tok/s，
> 量化了每 1M token 的电费口径，并把「哪些数字有落盘证据、哪些没有」写进卡片纪律。

## 1. 为什么这件事有含金量（30 秒）

部署数字网上很多，但简历上的每个数都要能当场展开：这张卡是单卡 6GB、
KV cache 15,232 tokens、max-model-len 2048 的真实约束下量出来的，
每格数字对应一个落盘载体（bench JSON + 1Hz 功率 CSV），缺证据的格子
明确写「—（证据未落盘，已撤）」而不是猜一个。

## 2. 服务事实（2 分钟）

```
vllm serve Qwen3-4B-AWQ --port 8123 --max-model-len 2048
        --gpu-memory-utilization 0.90 --enforce-eager
```
- 权重 2.5 GiB / 加载 10.3s；`Available KV cache memory: 2.09 GiB`；
  `GPU KV cache size: 15,232 tokens`；2048-token 请求的最大并发 7.44x。
- 踩坑三连：flashinfer 采样 JIT 需 nvcc（关 `VLLM_USE_FLASHINFER_SAMPLER`）、
  `vllm bench serve` 必须传本地 `--tokenizer` 否则卡在拉 HF、
  `--base-url .../v1 --endpoint /v1/completions` 双拼 404 造成 P99 全 0 假成功
  （已在 parse_results.py 里加了全零 P99 守卫）。

## 3. 数字速查（面试现场可直接引用）

**eager 三档**（48 条/档，input 256/output 128，`--ignore-eos` 定长）：

| 并发 | 输出 tok/s | TTFT p50/p99 ms | TPOT p99 | 电费 ¥/1M out |
|---|---|---|---|---|
| 1 | 19.99 | 131.2 / 211.8 | 65.06 | 0.441 |
| 4 | 71.71 | 432.6 / 481.2 | 79.43 | 0.122 |
| 16 | 261.41 | 1140.1 / 1796.0 | 61.78 | 0.043 |

**CUDA graph 对比**（同一硬件，eager → graph）：

| 并发 | 输出 tok/s | 加速 | TPOT p99 |
|---|---|---|---|
| 1 | 80.19 | 4.0× | 12.18 ms |
| 4 | 242.95 | 3.4× | 15.54 ms |
| 16 | 534.82 | 2.1× | 29.15 ms |

graph 档成本列为 —：功率采样只存在 /tmp、没随仓库落盘，卡片纪律是不报
无载体数字（吞吐 ×2.1–4.0 ⇒ GPU 时间占用至少同倍数下降，但电费不猜）。

## 4. 三个必追问（预演）

**Q1：为什么功率要按请求窗口对齐？**
采样器在 bench 客户端启动前就开了，导入/编译那段是 ~28W 空闲态；
直接对整份 CSV 求均值会把 c=16 档的 67.57W 稀释成 45.28W、成本低估 1/3。
`parse_results.py` 用 bench JSON 的 `date` 和 `duration` 切出
`[date-duration, date]` 窗口再求均值，窗口采样数（308/85/23 条）与各档
duration（307.4/85.7/23.5s）能对上——这是复核成本口径的锚。

**Q2：CUDA graph 为什么快，代价是什么？**
graph 消除了每次 step 的 kernel launch 开销，decode 密集（128 定长输出）时
收益最大：TPOT p99 从 65ms 降到 12ms。代价是显存：KV cache 从 15,232 tokens
掉到 8,304（并发余量 7.44x → 4.05x），且 util=0.95 + graph 实测 OOM
（Tried to allocate 1.80 GiB, 1.75 free）。所以 6GB 卡上 graph 是默认项，
KV 余量是你要用 OOM 换回来的判断。

**Q3：¥0.043/1M token 这个数怎么用？**
它只是 GPU 板级电（51–68W 窗口均值 × 1e6/吞吐 ÷ 3.6e6 kWh × ¥0.6），
不含整机、折旧、机房——不是云报价。它的用途是做**相对判断**：并发 16 档
比并发 1 档便宜 10.3×，CUDA graph 在同档位再省 2–4×（按吞吐折算）；
量级和方向可信，绝对值别当采购依据。

## 5. 边界（主动交代）

单轮扫描（每档一次，无重复运行 → 无置信区间）；`--ignore-eos` 定长输出，
真实变长业务会改变 prefill/decode 比例；random 输入让 prefix cache 命中很低；
2048 上下文别外推长文档。CI 只跑数字管线（selftest + card.json 结构断言），
GPU 基准本身标 manual。

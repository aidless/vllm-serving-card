#!/usr/bin/env bash
# 单一硬件 · vLLM serving 实测：3 个并发档，每档 48 条定长请求，同步采 GPU 板功率。
# 复现前提：vLLM server 已在 127.0.0.1:8123 起好（见 README.md 的 serve 命令）。
set -euo pipefail

CARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${VENV_BIN:-$HOME/venvs/vllm/bin}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8123}"
MODEL="${MODEL:-Qwen3-4B-AWQ}"
NUM_PROMPTS="${NUM_PROMPTS:-48}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B-AWQ}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CONCURRENCIES=(1 4 16)
OUT="$CARD_DIR/results"
mkdir -p "$OUT"

echo "硬件/服务快照" > "$OUT/environment.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> "$OUT/environment.txt"
"$VENV_BIN/python" -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)" >> "$OUT/environment.txt"
curl -s "$BASE_URL/v1/models" > "$OUT/models.json"

for C in "${CONCURRENCIES[@]}"; do
  echo "=== concurrency=$C ==="
  nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "$OUT/power-c$C.csv" &
  SAMPLER_PID=$!
  "$VENV_BIN/vllm" bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --endpoint /v1/completions \
    --model "$MODEL" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups 4 \
    --max-concurrency "$C" \
    --ignore-eos \
    --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "$OUT" \
    --result-filename "bench-c$C.json" \
    > "$OUT/bench-c$C.log" 2>&1
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
  echo "  -> $OUT/bench-c$C.json + $OUT/power-c$C.csv"
done

echo "全部档位完成；用 parse_results.py 生成数字卡片。"

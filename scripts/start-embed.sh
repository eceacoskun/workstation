#!/usr/bin/env bash
set -euo pipefail

# ── GPU0 -> embedding (LLM ile ayni kartta co-located) ─────────────
# NOT: LLM ile ayni GPU'da. Iki process VRAM'i onden ayirir; util
# fraction'lari TOPLAM < 1 olmali. LLM'i (start-qwen.sh) ONCE baslat.
export CUDA_VISIBLE_DEVICES=0

MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"

exec vllm serve "$MODEL" \
  --served-model-name qwen3-embedding \
  --task embed \
  --host 127.0.0.1 --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.14 \
  --max-model-len 8192 \
  --disable-log-requests

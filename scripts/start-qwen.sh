#!/usr/bin/env bash
set -euo pipefail

# ── GPU0 -> buyuk LLM (embedding ile ayni kartta co-located) ───────
# Embedding de bu GPU'da. util TOPLAM'i (bu + embed 0.14) < 1 kalmali.
# Bu process ONCE baslamali (embed sonra kalan yere sigar).
export CUDA_VISIBLE_DEVICES=0

# NOT: "qwen3.6" ailesinden KESIN HF repo id'sini buraya yaz.
# (Bu model muhtemelen bilgi kesme tarihimden yeni; dogrula.)
MODEL="${QWEN_MODEL:-Qwen/Qwen3-32B}"

# Tek kartta co-locate icin FP8 onerilir (Blackwell native FP8):
# 32B FP8 ~40GB -> KV cache'e ~45GB kalir. bf16 istersen QUANT'i bosalt
# ve util'i 0.70'e dusur (KV daralir).
QUANT="${QWEN_QUANT:-fp8}"

exec vllm serve "$MODEL" \
  --served-model-name qwen \
  --host 127.0.0.1 --port 8001 \
  --tensor-parallel-size 1 \
  --quantization "$QUANT" \
  --gpu-memory-utilization 0.72 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --disable-log-requests

# LLM Serving Stack — Qwen (LLM) + Qwen3-Embedding

2× RTX 6000 Pro (Blackwell, 96GB) workstation icin native (Docker'siz) kurulum.

```
istemci
  │  https://llm.example.com/{project}/v1/...
  ▼
nginx (443, TLS)                     ← nginx/llm.conf
  ▼
gateway (127.0.0.1:8080, FastAPI)    ← gateway/  (proje auth + token log)
  ├─ 127.0.0.1:8001  vLLM Qwen        (GPU0)   ← scripts/start-qwen.sh
  └─ 127.0.0.1:8002  vLLM Embedding   (GPU0)   ← scripts/start-embed.sh
```

## GPU yerlesimi
Her iki model de **GPU0**'da (co-located). **GPU1 bos** — ileride ikinci
replika ya da daha buyuk bir model icin ayrilabilir.

- **LLM** → `CUDA_VISIBLE_DEVICES=0`, **FP8**, util **0.72** (once baslar)
- **embedding** → `CUDA_VISIBLE_DEVICES=0`, util **0.14** (sonra baslar)

### VRAM butcesi (tek 96GB kart)
Iki ayri vLLM process VRAM'i **onden** ayirir; `--gpu-memory-utilization`
fraction'lari **TOPLAM < 1** olmali ve **LLM once** baslamali
(systemd'de `vllm-embed`, `vllm-qwen`'e `Requires/After` ile bagli).

| Kurulum | Weights | KV cache'e kalan |
|---|---|---|
| 32B **FP8** + embed (varsayilan) | ~40GB | ~45GB (rahat) |
| 32B **bf16** + embed | ~72GB | ~15GB (dar) |

bf16 istersen: `start-qwen.sh` icinde `QWEN_QUANT=""` ve util'i `0.70`'e dusur.
"qwen3.6" reposu zaten FP8 ise `--quantization` verme (cift quantize etme).

## ⚠️ Blackwell notu
RTX 6000 Pro = **sm_120**. CUDA **12.8+** ve yeni bir vLLM/PyTorch build'i sart;
eski wheel'lar kernel bulamayip patlar. Kurulumdan once:
```bash
nvidia-smi                       # driver + CUDA surumu
python -c "import torch; print(torch.cuda.get_device_capability())"   # (12, 0) beklenir
```

## Kurulum
```bash
sudo useradd -r -s /usr/sbin/nologin llm || true
sudo mkdir -p /opt/llm-gateway /var/log/llm-gateway
sudo cp -r gateway scripts /opt/llm-gateway/
sudo cp env.example /opt/llm-gateway/env          # sonra duzenle
sudo cp gateway/config.example.yaml /opt/llm-gateway/gateway/config.yaml   # sonra duzenle

# gateway venv (vLLM'den ayri ortam)
python3 -m venv /opt/llm-gateway/venv
/opt/llm-gateway/venv/bin/pip install -r /opt/llm-gateway/gateway/requirements.txt

sudo chmod +x /opt/llm-gateway/scripts/*.sh
sudo chown -R llm:llm /opt/llm-gateway /var/log/llm-gateway

# servisler
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen vllm-embed llm-gateway

# nginx
sudo cp nginx/llm.conf /etc/nginx/conf.d/llm.conf
sudo nginx -t && sudo systemctl reload nginx
```

vLLM ayri kurulur (Blackwell uyumlu surum), ornegin:
```bash
sudo -u llm pip install "vllm>=0.8"    # sm_120 destekleyen surumu dogrula
```

## Yeni proje ekleme
`gateway/config.yaml` icine bir blok ekle, key uret, gateway'i restart et:
```bash
python -c "import secrets; print('sk-proj-' + secrets.token_urlsafe(32))"
sudo systemctl restart llm-gateway
```

## Kullanim (istemci tarafi)
OpenAI-uyumlu; sadece `base_url` proje prefix'ini icerir, `model` sabit isim:
```python
from openai import OpenAI
c = OpenAI(base_url="https://llm.example.com/proj-a/v1", api_key="sk-proja-...")
c.chat.completions.create(model="qwen", messages=[{"role":"user","content":"selam"}])
c.embeddings.create(model="qwen3-embedding", input=["metin"])
```

## Token loglari
- `/var/log/llm-gateway/{project}.jsonl` — proje bazli
- `/var/log/llm-gateway/all.jsonl` — merkezi

Her satir: `ts, project, endpoint, model, prompt_tokens, completion_tokens,
total_tokens, stream, status, latency_ms, client_ip`.

Proje toplamı:
```bash
jq -s 'map(.total_tokens // 0) | add' /var/log/llm-gateway/proj-a.jsonl
```

### Streaming + token muhasebesi
Stream isteklerinde gateway `stream_options.include_usage=true` ekler; boylece
son chunk'ta usage gelir ve loglanir. Bu, istemciye sonda `choices: []` olan bir
usage-chunk'i da iletir (standart OpenAI davranisi, cogu client tolere eder).
Istemez isen `gateway/main.py` icinde include_usage enjeksiyonunu kaldir —
o zaman stream'lerde completion_tokens loglanamaz.

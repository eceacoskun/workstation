"""
LLM Gateway — OpenAI-uyumlu, proje bazli auth + token loglama.

Akis:
  istemci -> nginx (TLS) -> bu gateway (127.0.0.1:8080) -> vLLM backend'ler

  POST /{project}/v1/chat/completions  -> llm backend        (GPU0)
  POST /{project}/v1/completions       -> llm backend        (GPU0)
  POST /{project}/v1/embeddings        -> embedding backend  (GPU1)
  GET  /{project}/v1/models            -> projenin erisebildigi modeller

Her proje kendi API key'i ile kimlik dogrular; token kullanimi hem
logs/{project}.jsonl (proje bazli) hem logs/all.jsonl (merkezi) dosyasina
JSONL olarak yazilir.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

CONFIG_PATH = os.environ.get("GATEWAY_CONFIG", str(Path(__file__).with_name("config.yaml")))
LOG_DIR = Path(os.environ.get("GATEWAY_LOG_DIR", "/var/log/llm-gateway"))
REQUEST_TIMEOUT = float(os.environ.get("GATEWAY_TIMEOUT", "600"))


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()
BACKENDS: dict[str, dict[str, Any]] = CONFIG["backends"]   # {"llm": {url, model}, "embedding": {...}}
PROJECTS: dict[str, dict[str, Any]] = CONFIG["projects"]   # {name: {keys: [...], allowed: [...]}}

# endpoint tipi -> backend adi
ENDPOINT_BACKEND = {
    "chat": "llm",
    "completions": "llm",
    "embeddings": "embedding",
}
UPSTREAM_PATH = {
    "chat": "/v1/chat/completions",
    "completions": "/v1/completions",
    "embeddings": "/v1/embeddings",
}

LOG_DIR.mkdir(parents=True, exist_ok=True)
_loggers: dict[str, logging.Logger] = {}


def _jsonl_logger(name: str, filename: str) -> logging.Logger:
    """Verilen dosyaya tek satir JSON yazan, cache'lenen logger dondurur."""
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    _loggers[name] = logger
    return logger


def log_usage(record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    _jsonl_logger(f"proj.{record['project']}", f"{record['project']}.jsonl").info(line)
    _jsonl_logger("gateway.all", "all.jsonl").info(line)


_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0))
    yield
    await _client.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def authorize(project: str, endpoint_type: str, authorization: str | None) -> None:
    proj = PROJECTS.get(project)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"unknown project '{project}'")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    keys = proj.get("keys", [])
    if not any(hmac.compare_digest(token, str(k)) for k in keys):
        raise HTTPException(status_code=401, detail="invalid api key for project")
    backend_name = ENDPOINT_BACKEND[endpoint_type]
    allowed = proj.get("allowed", list(BACKENDS.keys()))
    if backend_name not in allowed:
        raise HTTPException(status_code=403, detail=f"project not allowed to use '{backend_name}'")


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "-"


async def forward(project: str, endpoint_type: str, request: Request, authorization: str | None):
    authorize(project, endpoint_type, authorization)
    backend_name = ENDPOINT_BACKEND[endpoint_type]
    backend = BACKENDS[backend_name]

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be valid JSON")

    # Model adini backend'in gercek served-model-name'i ile sabitle:
    # istemcilerin dogru model id'sini bilmesine gerek kalmaz.
    body["model"] = backend["model"]

    is_stream = bool(body.get("stream", False)) and endpoint_type != "embeddings"
    if is_stream:
        # Token muhasebesi icin usage chunk'ini zorla (vLLM son chunk'ta usage doner).
        opts = body.get("stream_options") or {}
        opts["include_usage"] = True
        body["stream_options"] = opts

    url = backend["url"].rstrip("/") + UPSTREAM_PATH[endpoint_type]
    payload = json.dumps(body).encode("utf-8")
    headers = {"content-type": "application/json"}
    started = time.monotonic()

    base_record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "endpoint": endpoint_type,
        "backend": backend_name,
        "model": backend["model"],
        "stream": is_stream,
        "client_ip": _client_ip(request),
    }

    if is_stream:
        return await _stream(url, headers, payload, base_record, started)
    return await _unary(url, headers, payload, base_record, started)


async def _unary(url, headers, payload, base_record, started):
    assert _client is not None
    resp = await _client.post(url, headers=headers, content=payload)
    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = None
    usage = (data or {}).get("usage") or {}
    log_usage({
        **base_record,
        "status": resp.status_code,
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    })
    if data is not None:
        return JSONResponse(status_code=resp.status_code, content=data)
    return JSONResponse(status_code=resp.status_code, content={"error": resp.text[:2000]})


async def _stream(url, headers, payload, base_record, started):
    assert _client is not None

    async def gen():
        usage: dict[str, Any] = {}
        status = 200
        buffer = ""
        async with _client.stream("POST", url, headers=headers, content=payload) as resp:
            status = resp.status_code
            if status != 200:
                text = (await resp.aread()).decode("utf-8", "ignore")
                log_usage({**base_record, "status": status,
                           "latency_ms": round((time.monotonic() - started) * 1000),
                           "error": text[:500]})
                yield f"data: {json.dumps({'error': text[:2000]})}\n\n".encode()
                return
            async for chunk in resp.aiter_bytes():
                yield chunk  # istemciye oldugu gibi ilet
                # kopyasini usage icin parse et (satirlar chunk sinirinda bolunebilir)
                buffer += chunk.decode("utf-8", "ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
        log_usage({
            **base_record,
            "status": status,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/{project}/v1/chat/completions")
async def chat_completions(project: str, request: Request, authorization: str | None = Header(None)):
    return await forward(project, "chat", request, authorization)


@app.post("/{project}/v1/completions")
async def completions(project: str, request: Request, authorization: str | None = Header(None)):
    return await forward(project, "completions", request, authorization)


@app.post("/{project}/v1/embeddings")
async def embeddings(project: str, request: Request, authorization: str | None = Header(None)):
    return await forward(project, "embeddings", request, authorization)


@app.get("/{project}/v1/models")
async def models(project: str, authorization: str | None = Header(None)):
    proj = PROJECTS.get(project)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"unknown project '{project}'")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not any(hmac.compare_digest(token, str(k)) for k in proj.get("keys", [])):
        raise HTTPException(status_code=401, detail="invalid api key for project")
    allowed = proj.get("allowed", list(BACKENDS.keys()))
    now = int(time.time())
    data = [{"id": BACKENDS[b]["model"], "object": "model", "created": now, "owned_by": "local"}
            for b in allowed if b in BACKENDS]
    return {"object": "list", "data": data}

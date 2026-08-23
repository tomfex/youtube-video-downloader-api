"""Backend adapter.

Talks to a local, OpenAI-compatible inference server (Ollama, llama.cpp,
vLLM, SGLang, LM Studio, ...). Provides both a blocking call and a streaming
line iterator so the gateway can re-emit in either OpenAI or Anthropic shape.
"""
import json

import requests

from .config import Config


class BackendError(Exception):
    """Raised when the local backend is unreachable or errors out."""


def _headers():
    h = {"Content-Type": "application/json"}
    if Config.BACKEND_API_KEY:
        h["Authorization"] = f"Bearer {Config.BACKEND_API_KEY}"
    return h


def _url():
    return Config.BACKEND_BASE_URL.rstrip("/") + "/chat/completions"


def chat(messages, model=None, temperature=None, max_tokens=None,
         top_p=None, stop=None, extra=None):
    """Non-streaming chat completion. Returns the backend's JSON dict."""
    payload = _build_payload(messages, model, temperature, max_tokens,
                             top_p, stop, stream=False, extra=extra)
    try:
        resp = requests.post(_url(), headers=_headers(), json=payload,
                             timeout=Config.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise BackendError(f"cannot reach local backend at {_url()}: {exc}")
    if resp.status_code >= 400:
        raise BackendError(f"backend returned {resp.status_code}: {resp.text}")
    return resp.json()


def chat_stream(messages, model=None, temperature=None, max_tokens=None,
                top_p=None, stop=None, extra=None):
    """Streaming chat completion.

    Yields parsed delta dicts from the backend's SSE stream, i.e. the
    `choices[0].delta` objects, plus a final None sentinel when done.
    """
    payload = _build_payload(messages, model, temperature, max_tokens,
                             top_p, stop, stream=True, extra=extra)
    try:
        resp = requests.post(_url(), headers=_headers(), json=payload,
                             timeout=Config.REQUEST_TIMEOUT, stream=True)
    except requests.RequestException as exc:
        raise BackendError(f"cannot reach local backend at {_url()}: {exc}")
    if resp.status_code >= 400:
        raise BackendError(f"backend returned {resp.status_code}: {resp.text}")

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        finish = choices[0].get("finish_reason")
        yield {"delta": delta, "finish_reason": finish}
    yield None


def _build_payload(messages, model, temperature, max_tokens, top_p, stop,
                   stream, extra):
    payload = {
        "model": Config.resolve_model(model),
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if stop:
        payload["stop"] = stop
    if extra:
        for k, v in extra.items():
            payload.setdefault(k, v)
    return payload

# Personal Model API — a Kimi-compatible, fully-local LLM gateway

This is a small gateway that speaks the **same API dialects as Kimi**
(OpenAI-compatible *and* Anthropic-compatible) but is backed entirely by a
model **you run locally**. Nothing is ever sent to `platform.moonshot.ai`.

> ### Why not the real Kimi-K2.5?
> Kimi-K2.5 is a **1-trillion-parameter** Mixture-of-Experts model (32B active,
> 256K context). Its weights are ~1TB even in FP8 and it needs a multi-GPU
> data-center machine (think 8× H200) to serve. It cannot run on a personal
> laptop/desktop or a single GPU. This gateway gives you *the Kimi developer
> experience* — the exact same endpoints and request/response shapes — while
> a model that fits your hardware actually does the work. If you later rent a
> big multi-GPU box and deploy the real K2.5 with vLLM/SGLang, just point
> `PMA_BACKEND_URL` at it; the gateway doesn't change.

## What it does

- Exposes `POST /v1/chat/completions` — **OpenAI-compatible** (streaming + non).
- Exposes `POST /v1/messages` — **Anthropic-compatible** (streaming + non).
- Exposes `GET /v1/models` and `GET /health`.
- Presents **your** model name (default `kimi-k2.5-local`) no matter which
  model a client asks for, and routes the actual work to a local backend.
- Optional API-key auth so only your clients can use it.

## How it fits together

```
your app / Claude Code / OpenAI SDK / Kimi SDK
        │  (base_url = http://your-machine:8080)
        ▼
  Personal Model API  (this gateway, Flask)
        │  (OpenAI /v1/chat/completions)
        ▼
  Local backend: Ollama / llama.cpp / vLLM / SGLang / LM Studio
        ▼
  A model that fits your hardware (Llama, Qwen, Mistral, …)
```

## Quick start

**1. Run a local model.** The simplest is [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b
ollama serve            # serves an OpenAI-compatible API on :11434
```

(Any OpenAI-compatible server works — see `.env.example` for llama.cpp,
vLLM, SGLang and LM Studio URLs.)

**2. Configure and launch the gateway:**

```bash
pip install -r personal_model_api/requirements.txt

export PMA_BACKEND_MODEL=llama3.1:8b      # the model you pulled
export PMA_MODEL_NAME=kimi-k2.5-local     # the name YOU present
export PMA_API_KEY=change-me              # optional; empty = open

python -m personal_model_api.server
# -> listening on http://0.0.0.0:8080
```

All settings live in environment variables — copy `.env.example` for the
full list. No secrets or machine paths are committed to the repo.

## Using it

### OpenAI SDK / any OpenAI-compatible client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="change-me")
resp = client.chat.completions.create(
    model="kimi-k2.5-local",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### Anthropic SDK / Claude Code

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="change-me")
msg = client.messages.create(
    model="kimi-k2.5-local",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(msg.content[0].text)
```

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.5-local","messages":[{"role":"user","content":"hi"}]}'
```

## Configuration reference

| Env var                 | Default                        | Meaning                                              |
|-------------------------|--------------------------------|------------------------------------------------------|
| `PMA_HOST`              | `0.0.0.0`                      | Gateway bind address                                 |
| `PMA_PORT`              | `8080`                         | Gateway port                                         |
| `PMA_MODEL_NAME`        | `kimi-k2.5-local`              | The model name you present to clients                |
| `PMA_API_KEY`           | *(empty)*                      | If set, clients must send it (Bearer / `x-api-key`)  |
| `PMA_BACKEND_URL`       | `http://localhost:11434/v1`    | OpenAI-compatible local backend                      |
| `PMA_BACKEND_KEY`       | `ollama`                       | Backend API key (Ollama ignores it)                  |
| `PMA_BACKEND_MODEL`     | `llama3.1:8b`                  | The real model tag the backend loads                 |
| `PMA_MODEL_PASSTHROUGH` | `false`                        | Forward client's model name instead of forcing above |
| `PMA_TIMEOUT`           | `600`                          | Upstream request timeout (seconds)                   |

## Notes & limitations

- The Anthropic bridge currently handles **text** content. Image/tool blocks
  are not translated (your local backend usually can't process them anyway).
- Token counts in Anthropic streaming responses are approximate (chunk count),
  since local backends don't always report usage mid-stream.
- Keep the gateway on a trusted network, or set `PMA_API_KEY`. It binds
  `0.0.0.0` by default so other devices on your LAN can reach it — narrow to
  `127.0.0.1` if you want localhost-only.

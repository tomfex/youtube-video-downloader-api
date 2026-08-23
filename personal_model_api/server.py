"""Kimi-compatible personal model API gateway.

Exposes OpenAI-compatible and Anthropic-compatible endpoints backed entirely by
a local inference server. No request ever leaves your machine for Moonshot's
platform — you point your existing Kimi/OpenAI/Anthropic clients at this
gateway's base URL instead.

Run:  python -m personal_model_api.server
"""
import json
import time
import uuid

from flask import Flask, request, jsonify, Response, stream_with_context

from .config import Config
from . import backends
from .backends import BackendError
from . import anthropic_compat

app = Flask(__name__)


# --------------------------------------------------------------------------
# Auth helper
# --------------------------------------------------------------------------
def _check_auth():
    """Return None if authorized, else a (json, status) error tuple."""
    if not Config.SERVER_API_KEY:
        return None
    presented = None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        presented = auth[len("Bearer "):].strip()
    if not presented:
        presented = request.headers.get("x-api-key")
    if presented != Config.SERVER_API_KEY:
        return jsonify({"error": {"type": "authentication_error",
                                  "message": "invalid or missing API key"}}), 401
    return None


def _openai_error(message, status=500, err_type="server_error"):
    return jsonify({"error": {"message": message, "type": err_type}}), status


# --------------------------------------------------------------------------
# Health / info
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "personal-model-api",
        "model": Config.PERSONAL_MODEL_NAME,
        "backend": Config.BACKEND_BASE_URL,
    })


# --------------------------------------------------------------------------
# OpenAI-compatible: GET /v1/models
# --------------------------------------------------------------------------
@app.route("/v1/models", methods=["GET"])
def list_models():
    err = _check_auth()
    if err:
        return err
    return jsonify({
        "object": "list",
        "data": [{
            "id": Config.PERSONAL_MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "personal",
        }],
    })


# --------------------------------------------------------------------------
# OpenAI-compatible: POST /v1/chat/completions
# --------------------------------------------------------------------------
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    err = _check_auth()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not messages:
        return _openai_error("'messages' is required", 400, "invalid_request_error")

    stream = bool(body.get("stream"))
    params = dict(
        model=body.get("model"),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        top_p=body.get("top_p"),
        stop=body.get("stop"),
    )

    if stream:
        return _openai_stream(messages, params)

    try:
        result = backends.chat(messages, **params)
    except BackendError as exc:
        return _openai_error(str(exc), 502, "backend_error")

    # Present our personal model name rather than the backend tag.
    result["model"] = Config.PERSONAL_MODEL_NAME
    result.setdefault("id", "chatcmpl-" + uuid.uuid4().hex[:24])
    result.setdefault("object", "chat.completion")
    return jsonify(result)


def _openai_stream(messages, params):
    created = int(time.time())
    cmpl_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    model = Config.PERSONAL_MODEL_NAME

    def generate():
        def chunk(delta, finish=None):
            return "data: " + json.dumps({
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}],
            }) + "\n\n"

        try:
            yield chunk({"role": "assistant"})
            finish_reason = "stop"
            for item in backends.chat_stream(messages, **params):
                if item is None:
                    break
                if item.get("finish_reason"):
                    finish_reason = item["finish_reason"]
                text = (item.get("delta") or {}).get("content")
                if text:
                    yield chunk({"content": text})
            yield chunk({}, finish=finish_reason)
            yield "data: [DONE]\n\n"
        except BackendError as exc:
            yield "data: " + json.dumps({"error": {"message": str(exc),
                                                    "type": "backend_error"}}) + "\n\n"
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream")


# --------------------------------------------------------------------------
# Anthropic-compatible: POST /v1/messages
# --------------------------------------------------------------------------
@app.route("/v1/messages", methods=["POST"])
def messages():
    err = _check_auth()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    if not body.get("messages"):
        return jsonify({"type": "error",
                        "error": {"type": "invalid_request_error",
                                  "message": "'messages' is required"}}), 400

    openai_messages = anthropic_compat.anthropic_to_openai_messages(body)
    stream = bool(body.get("stream"))
    params = dict(
        model=body.get("model"),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        top_p=body.get("top_p"),
        stop=body.get("stop_sequences"),
    )

    if stream:
        def generate():
            try:
                delta_iter = backends.chat_stream(openai_messages, **params)
                for event in anthropic_compat.stream_anthropic(
                        delta_iter, Config.PERSONAL_MODEL_NAME):
                    yield event
            except BackendError as exc:
                yield ("event: error\ndata: " + json.dumps({
                    "type": "error",
                    "error": {"type": "backend_error", "message": str(exc)},
                }) + "\n\n")

        return Response(stream_with_context(generate()),
                        mimetype="text/event-stream")

    try:
        result = backends.chat(openai_messages, **params)
    except BackendError as exc:
        return jsonify({"type": "error",
                        "error": {"type": "backend_error",
                                  "message": str(exc)}}), 502

    anthropic_result = anthropic_compat.openai_result_to_anthropic(
        result, Config.PERSONAL_MODEL_NAME)
    return jsonify(anthropic_result)


def main():
    print("=" * 60)
    print(" Personal Model API  (Kimi-compatible, fully local)")
    print("=" * 60)
    print(Config.summary())
    print("=" * 60)
    print("Endpoints:")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions   (OpenAI-compatible)")
    print("  POST /v1/messages           (Anthropic-compatible)")
    print("=" * 60)
    app.run(host=Config.HOST, port=Config.PORT, threaded=True)


if __name__ == "__main__":
    main()

"""Anthropic Messages API <-> OpenAI chat translation.

Kimi exposes an Anthropic-compatible endpoint (so tools like Claude Code can
point at it). We reproduce that surface here and map it onto the local
OpenAI-compatible backend.
"""
import json
import time
import uuid


def _text_from_content(content):
    """Anthropic content may be a string or a list of typed blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def anthropic_to_openai_messages(body):
    """Turn an Anthropic Messages request body into OpenAI messages."""
    messages = []

    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _text_from_content(system)})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": _text_from_content(msg.get("content"))})

    return messages


def _new_id():
    return "msg_" + uuid.uuid4().hex[:24]


def openai_result_to_anthropic(openai_json, model_name):
    """Convert a non-streaming OpenAI completion into an Anthropic message."""
    choice = (openai_json.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "") or ""
    usage = openai_json.get("usage") or {}
    stop_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
        None: "end_turn",
    }
    return {
        "id": _new_id(),
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_map.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_anthropic(delta_iter, model_name):
    """Turn the backend delta iterator into Anthropic SSE events."""
    msg_id = _new_id()

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    yield _sse("ping", {"type": "ping"})

    finish_reason = "stop"
    output_tokens = 0
    for item in delta_iter:
        if item is None:
            break
        delta = item.get("delta") or {}
        if item.get("finish_reason"):
            finish_reason = item["finish_reason"]
        text = delta.get("content")
        if text:
            output_tokens += 1
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            })

    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    stop_map = {"stop": "end_turn", "length": "max_tokens"}
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_map.get(finish_reason, "end_turn"),
            "stop_sequence": None,
        },
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})

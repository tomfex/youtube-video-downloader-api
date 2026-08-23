"""Configuration for the personal (Kimi-compatible) local API gateway.

Everything is driven by environment variables so no secrets or machine-specific
paths live in the repo. Nothing here ever points at platform.moonshot.ai — the
backend is always a model you run yourself.
"""
import os


def _env(name, default=None):
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _env_bool(name, default=False):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # --- Where THIS gateway listens ---------------------------------------
    HOST = _env("PMA_HOST", "0.0.0.0")
    PORT = int(_env("PMA_PORT", "8080"))

    # --- The name you present to the world -------------------------------
    # This is "your personal model". Clients ask for it (or anything else,
    # see MODEL_PASSTHROUGH) and get answered by your local backend model.
    PERSONAL_MODEL_NAME = _env("PMA_MODEL_NAME", "kimi-k2.5-local")

    # --- The local backend that actually runs the weights -----------------
    # Any OpenAI-compatible server works: Ollama, llama.cpp (llama-server),
    # vLLM, SGLang, LM Studio, text-generation-webui, etc.
    # Default targets a local Ollama install.
    BACKEND_BASE_URL = _env("PMA_BACKEND_URL", "http://localhost:11434/v1")
    BACKEND_API_KEY = _env("PMA_BACKEND_KEY", "ollama")

    # The real model tag the backend should load (e.g. "llama3.1:8b",
    # "qwen2.5:14b", "mistral-nemo"). Requests are always routed to this.
    BACKEND_MODEL = _env("PMA_BACKEND_MODEL", "llama3.1:8b")

    # If true, forward the client's requested model name to the backend
    # unchanged instead of forcing BACKEND_MODEL. Useful when your backend
    # hosts several models.
    MODEL_PASSTHROUGH = _env_bool("PMA_MODEL_PASSTHROUGH", False)

    # --- Optional auth on THIS gateway ------------------------------------
    # If set, clients must send this as a Bearer token (OpenAI style) or in
    # the x-api-key header (Anthropic style). Empty = open (localhost only).
    SERVER_API_KEY = _env("PMA_API_KEY", "")

    # Networking
    REQUEST_TIMEOUT = int(_env("PMA_TIMEOUT", "600"))

    @classmethod
    def resolve_model(cls, requested):
        """Map an incoming model name to the real backend model."""
        if cls.MODEL_PASSTHROUGH and requested:
            return requested
        return cls.BACKEND_MODEL

    @classmethod
    def summary(cls):
        return (
            f"personal model name : {cls.PERSONAL_MODEL_NAME}\n"
            f"backend url         : {cls.BACKEND_BASE_URL}\n"
            f"backend model       : {cls.BACKEND_MODEL}\n"
            f"model passthrough   : {cls.MODEL_PASSTHROUGH}\n"
            f"auth required       : {'yes' if cls.SERVER_API_KEY else 'no (open)'}\n"
            f"listening on        : http://{cls.HOST}:{cls.PORT}"
        )

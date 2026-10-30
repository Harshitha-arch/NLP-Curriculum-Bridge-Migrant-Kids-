"""Minimal LLM client over plain HTTPS for Google Gemini, OpenAI and Anthropic.

Keys are sent only in request headers, never in URLs, logs or error messages.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

PROVIDERS = ["gemini", "openai", "anthropic"]

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

# Model IDs that cannot write text (embeddings, speech, images); hidden in "List models".
NON_TEXT = (
    "embedding",
    "tts",
    "whisper",
    "dall-e",
    "image",
    "audio",
    "realtime",
    "moderation",
    "transcribe",
    "lyria",
    "robotics",
    "aqa",
    "imagen",
    "veo",
    "live",
)


class LLMError(RuntimeError):
    """A provider call failed; the message is safe to show to the user."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status  # HTTP status, or None for a network error / bad response
        self.retry_after = retry_after  # seconds from a Retry-After header, if any

    @property
    def retryable(self) -> bool:
        """Rate limits, server errors and network errors may succeed on a later try."""
        return self.status is None or self.status == 429 or self.status >= 500


@dataclass
class Generation:
    text: str
    model: str
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)


def _headers(provider: str, api_key: str) -> dict[str, str]:
    if provider == "gemini":
        return {"x-goog-api-key": api_key}
    if provider == "openai":
        return {"Authorization": f"Bearer {api_key}"}
    if provider == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    raise ValueError(f"provider must be one of {PROVIDERS}")


def build_request(
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float | None,
    max_tokens: int,
) -> tuple[str, dict]:
    """(url, json body) for one text generation call."""
    if provider == "gemini":
        cfg: dict = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": cfg,
        }
        return f"{GEMINI_URL}/models/{model}:generateContent", body
    if provider == "openai":
        # Current OpenAI models reject custom temperatures, so it is never sent.
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        return f"{OPENAI_URL}/chat/completions", body
    if provider == "anthropic":
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        return f"{ANTHROPIC_URL}/messages", body
    raise ValueError(f"provider must be one of {PROVIDERS}")


def parse_response(provider: str, model: str, data: dict) -> Generation:
    """Text, finish reason and token usage from a provider's JSON response."""
    if provider == "gemini":
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no text ({reason}).")
        c = cands[0]
        parts = (c.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = data.get("usageMetadata") or {}
        usage = {
            "input_tokens": u.get("promptTokenCount"),
            "output_tokens": u.get("candidatesTokenCount"),
            # Thinking counts against maxOutputTokens but is reported separately.
            "thinking_tokens": u.get("thoughtsTokenCount"),
        }
        gen = Generation(text, data.get("modelVersion", model), c.get("finishReason"), usage)
    elif provider == "openai":
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("OpenAI returned no choices.")
        ch = choices[0]
        u = data.get("usage") or {}
        usage = {
            "input_tokens": u.get("prompt_tokens"),
            "output_tokens": u.get("completion_tokens"),  # includes reasoning tokens
            "thinking_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        }
        text = (ch.get("message") or {}).get("content") or ""
        gen = Generation(text, data.get("model", model), ch.get("finish_reason"), usage)
    elif provider == "anthropic":
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")}
        gen = Generation(text, data.get("model", model), data.get("stop_reason"), usage)
    else:
        raise ValueError(f"provider must be one of {PROVIDERS}")
    if not gen.text.strip():
        raise LLMError(f"The model returned no text (finish reason: {gen.finish_reason}).")
    return gen


def _error_message(provider: str, resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except ValueError:
        msg = None
    return f"{provider} API error {resp.status_code}: {msg or resp.reason_phrase}"


def _retry_after(resp: httpx.Response) -> float | None:
    """Seconds from a numeric Retry-After header (HTTP-date values are ignored)."""
    try:
        return max(0.0, float(resp.headers.get("retry-after", "")))
    except ValueError:
        return None


def _call(client, provider, api_key, model, system, user, temperature, max_tokens) -> Generation:
    """One generation; retried once without temperature if the model rejects it."""
    for temp in (temperature, None) if temperature is not None else (None,):
        url, body = build_request(provider, model, system, user, temp, max_tokens)
        try:
            resp = client.post(url, json=body, headers=_headers(provider, api_key))
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach the {provider} API: {type(e).__name__}") from None
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                raise LLMError(f"{provider} API returned a response that is not JSON") from None
            try:
                return parse_response(provider, model, data)
            except LLMError as e:  # empty or blocked answer: retrying would not help
                raise LLMError(str(e), resp.status_code) from None
        msg = _error_message(provider, resp)
        if not (resp.status_code == 400 and "temperature" in msg.lower() and temp is not None):
            raise LLMError(msg, resp.status_code, _retry_after(resp))
    raise LLMError(msg, resp.status_code)


def generate(
    provider: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float | None = None,
    max_tokens: int = 8192,
    timeout: float = 180,
    client: httpx.Client | None = None,
    retries: int = 0,
    sleep: Callable[[float], None] = time.sleep,
) -> Generation:
    """Generate text. With ``retries``, rate limits (429), server errors (5xx) and network
    errors are retried with exponential backoff (2, 4, 8 ... s, at most 60 s), waiting for the
    server's Retry-After instead when it sends one."""
    if not api_key:
        raise LLMError("No API key.")
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        for attempt in range(retries + 1):
            try:
                return _call(
                    client, provider, api_key, model, system, user, temperature, max_tokens
                )
            except LLMError as e:
                if not e.retryable or attempt == retries:
                    raise
                wait = e.retry_after if e.retry_after is not None else 2 ** (attempt + 1)
                sleep(min(wait, 60.0))
        raise AssertionError("unreachable")
    finally:
        if own:
            client.close()


def list_models(provider: str, api_key: str, client: httpx.Client | None = None) -> list[str]:
    """Text-generation model IDs available to this key, sorted."""
    own = client is None
    client = client or httpx.Client(timeout=30)
    headers = _headers(provider, api_key)
    try:
        if provider == "gemini":
            resp = client.get(f"{GEMINI_URL}/models", params={"pageSize": 1000}, headers=headers)
        elif provider == "openai":
            resp = client.get(f"{OPENAI_URL}/models", headers=headers)
        else:
            resp = client.get(f"{ANTHROPIC_URL}/models", params={"limit": 1000}, headers=headers)
    except httpx.HTTPError as e:
        raise LLMError(f"Could not reach the {provider} API: {type(e).__name__}") from None
    finally:
        if own:
            client.close()
    if resp.status_code != 200:
        raise LLMError(_error_message(provider, resp))
    data = resp.json()
    if provider == "gemini":
        ids = [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
    else:
        ids = [m["id"] for m in data.get("data", [])]
    return sorted(i for i in ids if not any(t in i.lower() for t in NON_TEXT))

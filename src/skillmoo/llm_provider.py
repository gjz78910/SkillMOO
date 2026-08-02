from __future__ import annotations

import http.client
import json
import os
import socket
import time
from typing import Any
from urllib import error, request

_MAX_ATTEMPTS = 4
_RETRY_BACKOFF_SEC = 0.75


class LLMProviderError(RuntimeError):
    pass


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content.strip()
    raise LLMProviderError("LLM response missing text content.")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end < start:
        raise LLMProviderError("LLM response did not contain a JSON object.")
    try:
        result = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"LLM response contained invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise LLMProviderError("LLM JSON root must be an object.")
    return result


class LLMProvider:
    """Minimal OpenAI-compatible LLM client configured from environment variables."""

    def __init__(
        self,
        api_key_envs: tuple[str, ...] = ("SKILLMOO_LLM_API_KEY", "OPENAI_API_KEY"),
        base_url_envs: tuple[str, ...] = ("SKILLMOO_LLM_BASE_URL", "OPENAI_BASE_URL"),
        model_envs: tuple[str, ...] = ("SKILLMOO_LLM_MODEL",),
        timeout_envs: tuple[str, ...] = ("SKILLMOO_LLM_TIMEOUT_SEC",),
    ) -> None:
        self._api_key_envs = api_key_envs
        self._base_url_envs = base_url_envs
        self._model_envs = model_envs
        self._timeout_envs = timeout_envs

    def available(self) -> bool:
        return bool(self._base_url()) and bool(self._api_key())

    def _api_key(self) -> str | None:
        return _env(*self._api_key_envs)

    def _base_url(self) -> str | None:
        v = _env(*self._base_url_envs)
        return v.rstrip("/") if v else None

    def _model(self) -> str | None:
        return _env(*self._model_envs)

    def _timeout(self) -> float:
        raw = _env(*self._timeout_envs)
        if raw:
            try:
                v = float(raw)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
        return 180.0

    def complete_json(self, prompt: str, *, system: str = "Return concise valid JSON only.") -> dict[str, Any]:
        model = self._model() or ""
        if not model:
            raise LLMProviderError("SKILLMOO_LLM_MODEL is not configured.")
        base_url = self._base_url()
        if not base_url:
            raise LLMProviderError("SKILLMOO_LLM_BASE_URL is not configured.")
        api_key = self._api_key()
        if not api_key:
            raise LLMProviderError("SKILLMOO_LLM_API_KEY is not configured.")
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        req = request.Request(f"{base_url}/chat/completions", data=body, headers=headers, method="POST")
        response_body = ""
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with request.urlopen(req, timeout=self._timeout()) as resp:
                    response_body = resp.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                raise LLMProviderError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
            except http.client.IncompleteRead as exc:
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise LLMProviderError("Response truncated (IncompleteRead).") from exc
                time.sleep(_RETRY_BACKOFF_SEC * (2 ** attempt))
            except (ConnectionResetError, BrokenPipeError) as exc:
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise LLMProviderError("Connection dropped reading response.") from exc
                time.sleep(_RETRY_BACKOFF_SEC * (2 ** attempt))
            except error.URLError as exc:
                raise LLMProviderError(f"Request failed: {exc.reason}") from exc
            except (TimeoutError, socket.timeout) as exc:
                raise LLMProviderError("Request timed out.") from exc
        payload = json.loads(response_body)
        return _extract_json(_extract_text(payload))

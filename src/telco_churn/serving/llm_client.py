"""The single LLM integration seam (OpenAI-compatible chat completions).

Every backend (vLLM on Modal, Ollama locally, MockBed in tests) uses this
client; no other module imports a provider protocol. Grammar-constrained
decoding rides on ``response_format`` (vLLM guided_json / Ollama JSON mode).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from telco_churn.chat.schemas import (
    FEATURE_NAMES,
    FEATURE_VALUE_MENUS,
    NUMERIC_FEATURE_NAMES,
)
from telco_churn.config import get_settings

# Digits only: keeps the model from dumping stray text ("no", "fiber optic")
# into numeric slots where it would fail canonicalisation at runtime.
NUMERIC_PATTERN = r"^[0-9]+(\.[0-9]+)?$"


def _filter_slot(name: str) -> dict[str, Any]:
    """Slot spec for one filter: dataset literals for categoricals (the model
    can only pick real values), a digits-only pattern for numerics (stray text
    is unrepresentable in the grammar)."""
    if name in NUMERIC_FEATURE_NAMES:
        return {"type": "string", "pattern": NUMERIC_PATTERN}
    return {"enum": list(FEATURE_VALUE_MENUS[name])}


def churn_schema() -> dict[str, Any]:
    """JSON Schema of the extraction object (used as the grammar)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_features": {
                "type": "array",
                "items": {"type": "string", "enum": list(FEATURE_NAMES)},
            },
            # Categorical values are the dataset's own literals (menu slots)
            # and numeric values must match a digits-only pattern: constrained
            # decoding can only emit a real value in a real slot, so fabricated
            # strings like {"Monthly_Charges": "no online security"} become
            # unrepresentable.
            "filters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {name: _filter_slot(name) for name in FEATURE_NAMES},
            },
            "out_of_scope": {"type": "boolean"},
        },
        "required": ["target_features", "filters", "out_of_scope"],
    }


class LlmClient:
    """OpenAI-compatible chat client with grammar-enforced JSON output."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = settings.llm_base_url.rstrip("/")
        self._model = settings.llm_model
        headers = {"Content-Type": "application/json"}
        timeout = httpx.Timeout(timeout=600.0, connect=10.0)  # deep inference allowed
        if transport is not None:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers=headers,
                transport=transport,
                timeout=timeout,
            )
        else:
            self._client = httpx.Client(
                base_url=self._base_url, headers=headers, timeout=timeout
            )

    def extract_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """One chat completion constrained to the extraction JSON grammar."""
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "FeatureRequest", "schema": churn_schema()},
            },
        }
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        raw_content = choice["message"]["content"]
        candidate = json.loads(raw_content)
        if not isinstance(candidate, dict):
            raise TypeError("extraction candidate is not a JSON object")
        self._no_numeric_fields(candidate)
        return candidate

    def reachable(self) -> bool:
        """Lightweight liveness probe of the LLM backend (never raises)."""
        try:
            response = self._client.get(
                "/models",
                timeout=httpx.Timeout(timeout=5.0, connect=5.0),
            )
            response.raise_for_status()
        except (httpx.HTTPError, httpx.InvalidURL, OSError):
            return False
        return response.status_code == 200

    @staticmethod
    def _no_numeric_fields(candidate: dict[str, Any]) -> None:
        """Schema-invalid output dies here, before the tool executor."""
        allowed = {"target_features", "filters", "out_of_scope"}
        if not set(candidate).issubset(allowed):
            raise ValueError(
                f"extraction candidate has fields outside the grammar: "
                f"{sorted(set(candidate) - allowed)}"
            )
        if candidate.get("out_of_scope") not in (True, False):
            raise ValueError("out_of_scope must be a boolean")


class FakeExtractionTransport(httpx.BaseTransport):
    """Test double returning canned extraction JSONs (no network).

    ``candidate`` scripts a single reply; ``candidates`` scripts a sequence
    (the repair retry consumes the next entry; the last one repeats once the
    queue is exhausted).
    """

    def __init__(
        self,
        candidate: dict[str, Any] | None = None,
        *,
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        if candidate is None and not candidates:
            raise ValueError("provide candidate or candidates")
        self.candidate = candidate
        self._queue = list(candidates) if candidates else None
        self.calls: list[dict[str, Any]] = []

    def _next_candidate(self) -> dict[str, Any]:
        if self._queue:
            if len(self._queue) > 1:
                return self._queue.pop(0)
            return self._queue[0]
        if self.candidate is None:  # guarded in __init__
            raise RuntimeError("FakeExtractionTransport has no candidate")
        return self.candidate

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append({"url": str(request.url)})
        content = json.dumps(self._next_candidate())
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": content}}]},
        )

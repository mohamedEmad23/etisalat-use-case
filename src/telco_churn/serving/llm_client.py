"""The single LLM integration seam (OpenAI-compatible chat completions).

Every backend (vLLM on Modal, Ollama locally, MockBed in tests) uses this
client; no other module imports a provider protocol. Grammar-constrained
decoding rides on ``response_format`` (vLLM guided_json / Ollama JSON mode).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from telco_churn.chat.schemas import FEATURE_NAMES
from telco_churn.config import get_settings


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
            "filters": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": list(FEATURE_NAMES),
                },
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
    """Test double returning a canned extraction JSON (no network)."""

    def __init__(
        self,
        candidate: dict[str, Any],
    ) -> None:
        self.candidate = candidate
        self.calls: list[dict[str, Any]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append({"url": str(request.url)})
        content = json.dumps(self.candidate)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": content}}]},
        )

"""Demo-first deployment smoke (demo-surface §3, deployment spec 'Demo-first smoke').

Gated by the ``eval_llm`` marker: skips cleanly when the configured
OpenAI-compatible backend (local Ollama) is unreachable — there is nothing to
smoke. When the backend IS reachable the module performs ONE real authenticated
chat turn through the served application (``create_app`` wires the real
``LlmClient`` and the real ``ChurnChatPipeline``; no test doubles) and records
the evidence to ``runs/demo_smoke_report.json``: model identity, backend, turn
latency, and schema validity of the response payload.

Failure contract: if the backend answers but the turn or the payload fails
validation, the report is written with ``ok: false`` (deployment not
demo-ready) and the test fails loudly — never a fabricated payload.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from telco_churn.api.app import create_app
from telco_churn.config import get_settings

SMOKE_REPORT = "demo_smoke_report.json"
QUESTION = "Will a Fiber optic customer on a month-to-month contract churn?"


def _ollama_reachable(settings) -> bool:
    try:
        probe = httpx.Client(base_url=settings.llm_base_url, timeout=2.0)
        probe.get("/models")
        return True
    except (httpx.HTTPError, OSError):
        return False


@pytest.mark.eval_llm
def test_demo_smoke_single_real_turn() -> None:
    settings = get_settings()
    if not _ollama_reachable(settings):
        pytest.skip("demo smoke requires a reachable local Ollama endpoint")

    app = create_app()
    headers = {"Authorization": f"Bearer {settings.api_bearer_token}"}
    client = TestClient(app, raise_server_exceptions=False)

    started = time.perf_counter()
    transport_error: str | None = None
    response = None
    try:
        response = client.post(
            "/chat",
            json={"session_id": "demo-smoke", "message": QUESTION},
            headers=headers,
        )
    except httpx.HTTPError as exc:  # pragma: no cover - transport-level failure
        response = None
        transport_error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    parsed: object = None
    if response is not None:
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
    body = parsed if isinstance(parsed, dict) else {}
    payload = body.get("payload")
    schema_ok = bool(
        isinstance(payload, dict)
        and isinstance(payload.get("churn_probability"), (int, float))
        and 0.0 <= float(payload["churn_probability"]) <= 1.0
        and payload.get("churn_label") in {"Churner", "Not churner"}
        and isinstance(payload.get("drivers"), list)
        and len(payload["drivers"]) == 3
    )
    turn_ok = response is not None and response.status_code == 200
    error_body = body.get("error") if not turn_ok else None

    report = {
        "kind": "demo_smoke",
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.llm_model,
        "backend": settings.llm_backend.value,
        "base_url": settings.llm_base_url,
        "question": QUESTION,
        "latency_ms": latency_ms,
        "http_status": None if response is None else response.status_code,
        "ok": bool(turn_ok and schema_ok),
        "schema_validity": schema_ok,
        "text": body.get("text"),
        "payload": payload,
        "error": transport_error or error_body,
    }
    out = Path(settings.model_registry_dir) / SMOKE_REPORT
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if report["ok"] is not True:
        pytest.fail(
            f"demo smoke failed — deployment not demo-ready; report at {out} "
            f"(http_status={report['http_status']}, "
            f"schema_validity={schema_ok}, error={report['error']})"
        )
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True
    assert latency_ms > 0

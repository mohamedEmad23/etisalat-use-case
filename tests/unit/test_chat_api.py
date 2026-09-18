"""HTTP-level unit tests for the chat API (auth, health, errors, limiter, logs)."""

from __future__ import annotations

import os
from typing import Any

import httpx
import polars as pl
import pytest
from fastapi.testclient import TestClient

from telco_churn.api.app import create_app
from telco_churn.api.rate_limit import RateLimiter
from telco_churn.model.train import fit_frame
from telco_churn.serving.llm_client import FakeExtractionTransport, LlmClient

os.environ.setdefault("TELCO_API_BEARER_TOKEN", "unit-test-bearer-0")
_TOKEN = {"Authorization": "Bearer unit-test-bearer-0"}

_GOOD_CANDIDATE: dict[str, Any] = {
    "target_features": ["Contract"],
    "filters": {"Contract": "month-to-month"},
    "out_of_scope": False,
}
_OOS_CANDIDATE: dict[str, Any] = {
    "target_features": [],
    "filters": {},
    "out_of_scope": True,
}


class _DeadTransport(httpx.BaseTransport):
    """Every request dies — simulates an unreachable LLM backend."""

    def handle_request(self, _: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("backend down")


def make_frame(n: int = 240, seed: int = 0) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n).astype(float)
    contract = np.where(rng.random(n) < 0.55, "Month-to-month", "Two year")
    charges = rng.integers(20, 120, n).astype(float)
    churn = np.where(
        (contract == "Month-to-month") & (rng.random(n) < 0.6), "Yes", "No"
    )
    churn = np.where(rng.random(n) < 0.05, "Yes", churn)
    return pl.DataFrame(
        {
            "tenure": pl.Series(tenure),
            "Contract": pl.Series(contract),
            "Monthly_Charges": pl.Series(charges),
            "Churn": pl.Series(churn),
        }
    )


@pytest.fixture(scope="module")
def artifact() -> Any:
    return fit_frame(make_frame(), seed=42, n_trials=1, auc_band=(0.0, 1.0))


@pytest.fixture()
def good_client(artifact: Any) -> TestClient:
    llm = LlmClient(transport=FakeExtractionTransport(dict(_GOOD_CANDIDATE)))
    app = create_app(artifact=artifact, llm=llm, store=None)
    return TestClient(app, raise_server_exceptions=False)


def _post(
    client: TestClient,
    message: str = "Will a month-to-month customer churn?",
    session: str = "api-test",
) -> httpx.Response:
    return client.post(
        "/chat", json={"message": message, "session_id": session}, headers=_TOKEN
    )


class TestAuth:
    def test_401_when_header_missing(self, good_client: TestClient) -> None:
        response = good_client.post("/chat", json={"message": "hi", "session_id": "s"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_401_when_token_invalid(self, good_client: TestClient) -> None:
        response = good_client.post(
            "/chat",
            json={"message": "hi", "session_id": "s"},
            headers={"Authorization": "Bearer wrong-token-value"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_200_authenticated_round_trip(self, good_client: TestClient) -> None:
        response = _post(
            good_client, "Churn risk for fiber month-to-month, tenure 3 months?"
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"session_id", "text", "payload"}
        payload = body["payload"]
        assert isinstance(payload, dict)
        assert "churn_probability" in payload
        assert "churn_label" in payload
        assert isinstance(payload["drivers"], list) and len(payload["drivers"]) == 3


class TestHealth:
    def test_health_reachable_backend(self, artifact: Any) -> None:
        llm = LlmClient(transport=FakeExtractionTransport(dict(_GOOD_CANDIDATE)))
        client = TestClient(create_app(artifact=artifact, llm=llm))
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"service": "ok", "llm": "reachable"}

    def test_health_surfaces_unreachable_backend(self, artifact: Any) -> None:
        llm = LlmClient(transport=_DeadTransport())
        client = TestClient(create_app(artifact=artifact, llm=llm))
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"service": "ok", "llm": "unreachable"}

    def test_health_needs_no_token(self, good_client: TestClient) -> None:
        assert good_client.get("/health").status_code == 200


class TestErrorContract:
    def test_internal_error_masked(
        self, artifact: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = LlmClient(transport=FakeExtractionTransport(dict(_GOOD_CANDIDATE)))
        app = create_app(artifact=artifact, llm=llm)
        pipeline = app.state.pipeline

        def _boom(_: str, __: str) -> None:
            raise RuntimeError("secret stack trace material")

        monkeypatch.setattr(pipeline, "handle", _boom)
        client = TestClient(app, raise_server_exceptions=False)
        response = _post(client)
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_error"
        assert "secret stack trace material" not in response.text
        assert "RuntimeError" not in response.text

    def test_malformed_body_maps_to_invalid_request(
        self, good_client: TestClient
    ) -> None:
        response = good_client.post("/chat", json={"session_id": "s"}, headers=_TOKEN)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_out_of_scope_is_refusal_payload_none(self, artifact: Any) -> None:
        llm = LlmClient(transport=FakeExtractionTransport(dict(_OOS_CANDIDATE)))
        client = TestClient(
            create_app(artifact=artifact, llm=llm), raise_server_exceptions=False
        )
        response = _post(client, "What is the weather tomorrow?")
        assert response.status_code == 200
        body = response.json()
        assert body["payload"] is None
        assert "churn" in body["text"].lower()


class TestRateLimiting:
    def test_extra_call_is_429_with_contract_body(self, artifact: Any) -> None:
        llm = LlmClient(transport=FakeExtractionTransport(dict(_GOOD_CANDIDATE)))
        limiter = RateLimiter(max_calls=1, window_seconds=3600)
        client = TestClient(
            create_app(artifact=artifact, llm=llm, limiter=limiter),
            raise_server_exceptions=False,
        )
        first = _post(client)
        assert first.status_code == 200
        second = _post(client)
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "rate_limited"

    def test_limiter_window_logic(self) -> None:
        limiter = RateLimiter(max_calls=2, window_seconds=60)
        assert limiter.allow("k") and limiter.allow("k")
        assert not limiter.allow("k")
        assert limiter.allow("other")


class TestRedactionLogging:
    def test_log_line_is_scrubbed(
        self, good_client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="telco_churn.api.requests"):
            response = _post(
                good_client,
                "mail user.name@example.com or call 0805553434, id 7590VHVEG",
            )
        assert response.status_code == 200
        joined = caplog.text
        assert "user.name@example.com" not in joined
        assert "7590VHVEG" not in joined
        assert "[EMAIL]" in joined
        assert "[PHONE]" in joined
        assert "[CUSTOMER_ID]" in joined

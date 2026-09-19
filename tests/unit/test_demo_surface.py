"""Unit tests for the demo surface: page shell, /demo route, health identity."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from telco_churn.api.app import create_app
from telco_churn.config import get_settings
from telco_churn.model.train import fit_frame
from telco_churn.serving.llm_client import FakeExtractionTransport, LlmClient

os.environ.setdefault("TELCO_API_BEARER_TOKEN", "unit-test-bearer-0")

_SHELL_FORBIDDEN = (
    "http://",
    "https://",
    "<script src",
    'rel="stylesheet"',
    "//cdn",
    "integrity=",
    "localStorage",
    "sessionStorage",
    "TELCO_API_BEARER_TOKEN",
)


def make_frame(n: int = 240, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "Two year"], size=n)
    churn = np.where(
        (contract == "Month-to-month") & (rng.random(n) < 0.6), "Yes", "No"
    )
    return pl.DataFrame(
        {
            "tenure": pl.Series(rng.integers(1, 72, n).astype(float)),
            "Contract": pl.Series(contract),
            "Churn": pl.Series(churn),
        }
    )


@pytest.fixture(scope="module")
def artifact() -> Any:
    return fit_frame(make_frame(), seed=42, n_trials=1, auc_band=(0.0, 1.0))


def _client(artifact: Any) -> TestClient:
    candidate = {"target_features": [], "filters": {}, "out_of_scope": True}
    llm = LlmClient(transport=FakeExtractionTransport(candidate))
    app = create_app(artifact=artifact, llm=llm)
    return TestClient(app, raise_server_exceptions=False)


class TestDemoRoute:
    def test_serves_self_contained_page(self, artifact: Any) -> None:
        response = _client(artifact).get("/demo")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-store"
        assert "Telco Churn Copilot" in response.text

    def test_route_is_hidden_from_openapi(self, artifact: Any) -> None:
        client = _client(artifact)
        assert "/demo" not in client.get("/openapi.json").json()["paths"]

    def test_shell_needs_no_token(self, artifact: Any) -> None:
        assert _client(artifact).get("/demo").status_code == 200


class TestPageShellLeaksNothing:
    def test_no_external_or_persistent_references(self, artifact: Any) -> None:
        html = _client(artifact).get("/demo").text
        for needle in _SHELL_FORBIDDEN:
            assert needle not in html, f"demo page must not contain {needle!r}"

    def test_no_secret_material_embedded(self, artifact: Any) -> None:
        html = _client(artifact).get("/demo").text
        assert get_settings().api_bearer_token not in html


class TestHealthIdentity:
    def test_reports_configured_model_identity(self, artifact: Any) -> None:
        body = _client(artifact).get("/health").json()
        settings = get_settings()
        assert body["llm_model"] == settings.llm_model
        assert body["llm_backend"] == settings.llm_backend.value

    def test_health_still_open(self, artifact: Any) -> None:
        assert _client(artifact).get("/health").status_code == 200

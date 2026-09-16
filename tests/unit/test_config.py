from __future__ import annotations

import pytest
from pydantic import ValidationError

from telco_churn.config import AppConfig, LlmBackend

MIN_ENV = {
    "TELCO_API_BEARER_TOKEN": "super-secret-token-0123",
}


def test_config_loads_with_minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in MIN_ENV.items():
        monkeypatch.setenv(k, v)
    cfg = AppConfig()  # type: ignore[call-arg]  # pydantic-settings populates from env
    assert cfg.llm_backend is LlmBackend.OLLAMA
    assert cfg.data_csv_path.as_posix() == "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_config_fails_fast_on_missing_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELCO_API_BEARER_TOKEN", raising=False)
    with pytest.raises(ValidationError) as err:
        AppConfig()  # type: ignore[call-arg]  # pydantic-settings populates from env
    assert "api_bearer_token" in str(err.value)


def test_config_fails_fast_on_invalid_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {**MIN_ENV, "TELCO_LLM_BACKEND": "not-a-backend"}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError) as err:
        AppConfig()  # type: ignore[call-arg]  # pydantic-settings populates from env
    assert "llm_backend" in str(err.value)


def test_config_rejects_short_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELCO_API_BEARER_TOKEN", "short")
    with pytest.raises(ValidationError):
        AppConfig()  # type: ignore[call-arg]  # pydantic-settings populates from env

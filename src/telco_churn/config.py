"""Application configuration. Import-time safe; construction is fail-fast."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class LlmBackend(str, Enum):
    """Runtime provider behind the single OpenAI-compatible client seam."""

    VLLM = "vllm"
    OLLAMA = "ollama"
    MOCK = "mock"


class AppConfig(BaseSettings):
    """Validated application settings; instantiating with invalid values raises with a clear name."""

    model_config = {"env_prefix": "TELCO_", "env_file": ".env", "extra": "ignore"}

    # Serving identity
    api_host: str = "0.0.0.0"  # nosec B104 - bind all interfaces for container/serverless serving; override via TELCO_API_HOST
    api_port: int = 8000
    api_bearer_token: str = Field(min_length=8)  # required: no insecure default

    # Data
    data_csv_path: Path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    random_seed: int = 42

    # LLM seam (D-D): one seam, backend chosen by config
    llm_backend: LlmBackend = LlmBackend.OLLAMA
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    llm_base_url: str = "http://localhost:11434/v1"

    # Model artifacts
    model_registry_dir: Path = Path("runs")
    extraction_failover_model: str | None = None  # documented Qwen3-8B swap path

    @field_validator("api_host")
    @classmethod
    def _reject_placeholder_host(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("api_host must not be empty")
        return v.strip()


def get_settings() -> AppConfig:
    """Build (and memoize) AppConfig. Raises SettingsError on missing/invalid required env."""
    return AppConfig()  # type: ignore[call-arg]  # pydantic-settings populates from env

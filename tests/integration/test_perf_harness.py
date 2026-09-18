"""Gated perf harness test: records measured latency numbers (needs Ollama)."""

from __future__ import annotations

import json

import httpx
import pytest

from telco_churn.api import perf
from telco_churn.chat.loop import ChurnChatPipeline, load_default_artifact
from telco_churn.config import get_settings
from telco_churn.serving.llm_client import LlmClient


def _ollama_reachable() -> bool:
    from telco_churn.config import get_settings

    base = get_settings().llm_base_url.rstrip("/")
    candidates = {base, base.replace("/v1", "")}
    for candidate in candidates:
        try:
            response = httpx.get(candidate + "/models", timeout=2.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            continue
    return False


@pytest.mark.eval_llm
def test_perf_budget_numbers_recorded_locally() -> None:
    if not _ollama_reachable():
        pytest.skip("local Ollama endpoint unreachable (opt-in harness)")
    pipeline = ChurnChatPipeline(load_default_artifact(), LlmClient())
    summary = perf.measure(pipeline, session_id="perf-test", repeats=3)
    path = perf.report_path()
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    budgets = summary["budgets"]
    # recorded regardless: a miss is documented with the number, not the adjective
    assert budgets["p95_e2e_max_seconds"] == perf.BUDGET_P95_E2E_SECONDS
    assert budgets["first_token_max_seconds"] == perf.BUDGET_FIRST_TOKEN_SECONDS
    assert len(summary["e2e_seconds"]) == 3
    get_settings()  # settings validated upstream; loading report keeps parity

"""Perf budget harness: p95 end-to-end turn latency + first-token latency.

First-token latency is approximated at the single LLM seam: the time to
first response byte of the extraction completion (a TimedTransport warms
nothing else — the deterministic composition after extraction adds only
sub-millisecond work by design).

Run as a CLI:  PYTHONPATH=src .venv/bin/python -m telco_churn.api.perf
Needs a real LLM backend (local Ollama). Report lands in runs/.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from telco_churn.chat.loop import ChurnChatPipeline, load_default_artifact
from telco_churn.serving.llm_client import LlmClient

logger = logging.getLogger(__name__)

BUDGET_P95_E2E_SECONDS = 5.0
BUDGET_FIRST_TOKEN_SECONDS = 1.0
SAMPLE_MESSAGES = (
    (
        "Will this customer churn? Contract is month-to-month, "
        "fiber internet, tenure 3 months."
    ),
    "What about a paperless-billing user on a two-year contract?",
    "Churn risk for electronic check payment with tech support?",
)


class TimedTransport(httpx.BaseTransport):
    """Wraps a real transport, recording per-request time-to-first-byte."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner
        self.ttfb_seconds: list[float] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started = time.perf_counter()
        response = self._inner.handle_request(request)
        # read the stream fully (non-streaming completions: header+body once)
        _ = response.read()
        self.ttfb_seconds.append(time.perf_counter() - started)
        return response


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(round(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def measure(
    pipeline: ChurnChatPipeline, *, session_id: str, repeats: int
) -> dict[str, Any]:
    """Run ``repeats`` full chat turns; return latency stats + budgets verdicts."""
    e2e: list[float] = []
    llm = pipeline._llm
    inner = getattr(llm._client, "_transport", None)
    if inner is None:
        raise RuntimeError("LLM client transport not reachable for instrumentation")
    timed = TimedTransport(inner)
    llm._client = httpx.Client(
        base_url=llm._client.base_url,
        transport=timed,
        timeout=httpx.Timeout(timeout=600.0, connect=10.0),
    )
    for index in range(repeats):
        message = SAMPLE_MESSAGES[index % len(SAMPLE_MESSAGES)]
        session = f"{session_id}-{index}"
        started = time.perf_counter()
        pipeline.handle(session, message)
        e2e.append(time.perf_counter() - started)
    p95_e2e = percentile(e2e, 0.95)
    first_token = percentile(timed.ttfb_seconds, 0.95)
    return {
        "kind": "chat_api_perf_report",
        "repeats": repeats,
        "sample_size": len(e2e),
        "e2e_seconds": e2e,
        "first_token_seconds": timed.ttfb_seconds,
        "p50_e2e": round(percentile(e2e, 0.50), 4),
        "p95_e2e": round(p95_e2e, 4),
        "p50_first_token": round(percentile(timed.ttfb_seconds, 0.50), 4),
        "p95_first_token": round(first_token, 4),
        "budgets": {
            "p95_e2e_max_seconds": BUDGET_P95_E2E_SECONDS,
            "first_token_max_seconds": BUDGET_FIRST_TOKEN_SECONDS,
            "p95_e2e_pass": p95_e2e <= BUDGET_P95_E2E_SECONDS,
            "first_token_pass": first_token <= BUDGET_FIRST_TOKEN_SECONDS,
        },
    }


def report_path() -> Any:
    from telco_churn.config import get_settings

    return get_settings().model_registry_dir / "perf_report.json"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="measure chat API latency budgets")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args(argv)

    pipeline = ChurnChatPipeline(load_default_artifact(), LlmClient())
    summary = measure(pipeline, session_id="perf-cli", repeats=args.repeats)
    path = report_path()
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary["budgets"], indent=2))
    print(f"report saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

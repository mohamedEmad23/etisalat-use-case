"""Extraction evaluation suite (spec 4.7).

Curated cases cover every one of the 19 features, multi-intent utterances,
and out-of-scope utterances. Acceptance bar: 100% schema-validity of the
FeatureRequest and ≥95% slot accuracy; results are recorded per case and
written to a report artifact.

Opt-in: runs only when a local Ollama (OpenAI-compatible) endpoint is
reachable; otherwise the whole module skips in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from telco_churn.chat.prompts import extraction_messages
from telco_churn.chat.schemas import FEATURE_NAMES, FeatureRequest
from telco_churn.config import get_settings
from telco_churn.serving.llm_client import LlmClient

SYNTHETIC_REPORT = "extraction_suite_report.json"


# One utterance per feature; expected slots = features expected in target_features.
CASES: list[dict[str, str | set[str]]] = [
    {"utterance": "Is a male customer more likely to churn?", "slots": {"gender"}},
    {
        "utterance": "Does being a senior citizen raise churn risk?",
        "slots": {"Senior_Citizen"},
    },
    {"utterance": "Will an unmarried customer churn?", "slots": {"Is_Married"}},
    {"utterance": "Do customers with dependents stay longer?", "slots": {"Dependents"}},
    {"utterance": "How does tenure affect churn?", "slots": {"tenure"}},
    {
        "utterance": "Does having phone service change churn?",
        "slots": {"Phone_Service"},
    },
    {"utterance": "What about customers with dual lines?", "slots": {"Dual"}},
    {
        "utterance": "Churn risk for fiber optic internet customers?",
        "slots": {"Internet_Service"},
    },
    {"utterance": "Is online security linked to churn?", "slots": {"Online_Security"}},
    {"utterance": "Does online backup matter for churn?", "slots": {"Online_Backup"}},
    {
        "utterance": "Do device protection customers churn less?",
        "slots": {"Device_Protection"},
    },
    {"utterance": "Does tech support reduce churn?", "slots": {"Tech_Support"}},
    {"utterance": "Do streaming TV users churn more?", "slots": {"Streaming_TV"}},
    {"utterance": "Is churn tied to streaming movies?", "slots": {"Streaming_Movies"}},
    {"utterance": "What's churn for a two-year contract?", "slots": {"Contract"}},
    {
        "utterance": "Does paperless billing increase churn?",
        "slots": {"Paperless_Billing"},
    },
    {"utterance": "Churn for bank transfer payments?", "slots": {"Payment_Method"}},
    {
        "utterance": "Does a high monthly charge predict churn?",
        "slots": {"Monthly_Charges"},
    },
    {"utterance": "Is high total charge tied to churn?", "slots": {"Total_Charges"}},
    # multi-intent
    {
        "utterance": "Will a fiber-optic month-to-month customer without tech support churn?",
        "slots": {"Internet_Service", "Contract", "Tech_Support"},
    },
    {
        "utterance": "Compare churn for DSL versus fiber optic on one-year contracts",
        "slots": {"Internet_Service", "Contract"},
    },
    # out-of-scope
    {"utterance": "What's the weather in Cairo today?", "slots": set()},
    {"utterance": "Tell me a joke about routers", "slots": set()},
]


def _ollama_reachable(settings) -> bool:
    try:
        probe = httpx.Client(base_url=settings.llm_base_url, timeout=2.0)
        probe.get("/models")
        return True
    except (httpx.HTTPError, OSError):
        return False


@pytest.mark.eval_llm
def test_extraction_suite(tmp_path: Path) -> None:
    settings = get_settings()
    if not _ollama_reachable(settings):
        pytest.skip("evaluation target requires a reachable local Ollama endpoint")
    llm = LlmClient()
    records: list[dict[str, object]] = []
    valid_requests = 0
    expected_slot_hits = 0
    expected_slot_total = 0
    for case in CASES:
        utterance = str(case["utterance"])
        expected = case["slots"]
        candidate = llm.extract_json(extraction_messages(utterance))
        request = FeatureRequest.model_validate(candidate)  # raises → invalid
        valid_requests += 1
        predicted = {f.value for f in request.target_features}
        expected_names = {s for s in expected if s in FEATURE_NAMES}
        out_of_scope = expected_names == set()
        if out_of_scope:
            matched = request.out_of_scope
            slot_correct = matched and not predicted
            matched_slots = 0
        else:
            matched_slots = len(expected_names & predicted)
            slot_correct = matched_slots == len(expected_names)
        expected_slot_hits += matched_slots
        expected_slot_total += len(expected_names)
        records.append(
            {
                "utterance": utterance,
                "expected": sorted(expected_names),
                "predicted_target_features": sorted(predicted),
                "predicted_filters": {k.value: v for k, v in request.filters.items()},
                "out_of_scope": request.out_of_scope,
                "correct": slot_correct,
            }
        )
    assert valid_requests == len(CASES), "schema validity must be 100%"
    accuracy = expected_slot_hits / max(expected_slot_total, 1)
    assert accuracy >= 0.95, f"slot accuracy {accuracy:.2%} < 95% acceptance bar"
    report = {
        "kind": "extraction_evaluation",
        "model": settings.llm_model,
        "test_cases": len(CASES),
        "schema_validity": 1.0,
        "slot_accuracy": round(accuracy, 4),
        "records": records,
    }
    out = Path(settings.model_registry_dir) / SYNTHETIC_REPORT
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["slot_accuracy"] >= 0.95

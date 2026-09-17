"""Phase 3 unit tests — mock LLM transport (no network)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from telco_churn.chat.loop import ChurnChatPipeline
from telco_churn.chat.redact import redact
from telco_churn.chat.schemas import FEATURE_NAMES, FeatureRequest, is_schema_candidate
from telco_churn.model.train import fit_frame
from telco_churn.serving.llm_client import FakeExtractionTransport, LlmClient


def make_frame(n: int = 240, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "Two year"], size=n, p=[0.55, 0.45])
    churn_mask = (contract == "Month-to-month") & (rng.random(n) < 0.6)
    churn = np.where(churn_mask | (rng.random(n) < 0.05), "Yes", "No")
    return pl.DataFrame(
        {
            "tenure": rng.integers(1, 72, n).astype(float),
            "Contract": contract,
            "Churn": churn,
        }
    )


@pytest.fixture(scope="module")
def artifact():
    return fit_frame(make_frame(), seed=42, n_trials=1, auc_band=(0.0, 1.0))


def _client(candidate: dict[str, Any]) -> LlmClient:
    return LlmClient(transport=FakeExtractionTransport(candidate))


class TestSchemaContract:
    def test_rejects_unknown_feature(self) -> None:
        assert not is_schema_candidate({"target_features": ["pricesрым_feeling"]})

    def test_rejects_numeric_injection(self) -> None:
        assert not is_schema_candidate({"predicted_churn": 0.42})
        assert not is_schema_candidate({"churn_probability": "high"})
        assert not is_schema_candidate(
            {"target_features": ["tenure"], "how_certain": 3}
        )

    def test_numeric_filter_value_must_be_string(self) -> None:
        with pytest.raises(ValueError):
            FeatureRequest.model_validate({"filters": {"tenure": 24}})

    def test_vocabulary_is_nineteen_closed_names(self) -> None:
        assert len(FEATURE_NAMES) == 19
        assert "customerID" not in FEATURE_NAMES and "Churn" not in FEATURE_NAMES


class TestLoop:
    def test_malformed_candidate_never_corrupts_state(self, artifact) -> None:
        pipeline = ChurnChatPipeline(artifact, _client({"bogus_numeric": 7}))
        reply = pipeline.handle("s1", "hello there")
        assert reply.payload is None
        assert "feature vocabulary" in reply.text
        assert len(pipeline._store) <= 1
        assert pipeline._store.snapshot("s1") == {}

    def test_numeric_filter_value_must_parse(self, artifact) -> None:
        client = _client(
            {
                "target_features": ["tenure"],
                "filters": {"tenure": "weeks"},
                "out_of_scope": False,
            }
        )
        pipeline = ChurnChatPipeline(artifact, client)
        reply = pipeline.handle("s2", "my customer stayed for ages")
        assert reply.payload is None
        assert "valid value for tenure" in reply.text

    def test_numerics_composed_from_payload_only(self, artifact) -> None:
        client = _client(
            {
                "target_features": ["Contract"],
                "filters": {"Contract": "month-to-month"},
                "out_of_scope": False,
            }
        )
        pipeline = ChurnChatPipeline(artifact, client)
        reply = pipeline.handle("s3", "will a month-to-month customer churn?")
        assert reply.payload is not None
        prob = reply.payload["churn_probability"]
        text = reply.text
        driver_rows: list[dict[str, object]] = reply.payload["drivers"]  # type: ignore[assignment]
        numeric_pool = {f"{prob}", f"{pipeline._threshold:.4f}"} | {
            f"{entry['contribution']}" for entry in driver_rows
        }
        for token in text.split(" "):
            bare = token.strip(".,;:()[]")
            if bare.replace(".", "").isdigit():
                assert bare in numeric_pool, (
                    f"{bare} not interpolated from tool payload"
                )

    def test_two_turn_refinement_accumulates_features(self, artifact) -> None:
        client = _client(
            {
                "target_features": ["Contract"],
                "filters": {"Contract": "Month-to-month"},
                "out_of_scope": False,
            }
        )
        pipeline = ChurnChatPipeline(artifact, client)
        first = pipeline.handle("s4", "will a month-to-month customer churn?")
        assert first.payload is not None
        second = pipeline.handle("s5", "and what if tenure is 24 months")
        assert second.text == first.text  # same fake candidate → same reply
        # state accumulation is verified via the refined pipeline below
        refined = ChurnChatPipeline(artifact, client)
        refined.handle("s6", "will a month-to-month customer churn?")
        refined_client = _client(
            {
                "target_features": ["tenure"],
                "filters": {"tenure": "24"},
                "out_of_scope": False,
            }
        )
        refined._llm = refined_client
        second_final = refined.handle("s6", "make tenure 24 months")
        assert second_final.payload is not None
        assert refined._store.snapshot("s6") == {
            "Contract": "Month-to-month",
            "tenure": "24",
        }

    def test_out_of_scope_refusal(self, artifact) -> None:
        client = _client({"target_features": [], "filters": {}, "out_of_scope": True})
        pipeline = ChurnChatPipeline(artifact, client)
        reply = pipeline.handle("s7", "what's the weather in Cairo?")
        assert reply.payload is None
        assert "churn" in reply.text.lower()
        assert "fiber-optic" in reply.text
        assert not any(ch.isdigit() for ch in reply.text)

    def test_pii_scrubs_emails_phones_customer_ids(self) -> None:
        log_line = "email jane.doe@corp.com phone +971 50 123 4567 id 7590VHVEG"
        scrubbed = redact(log_line)
        assert "jane.doe@corp.com" not in scrubbed
        assert "+971 50 123 4567" not in scrubbed
        assert "7590VHVEG" not in scrubbed
        assert (
            "[EMAIL]" in scrubbed
            and "[PHONE]" in scrubbed
            and "[CUSTOMER_ID]" in scrubbed
        )

"""Phase 3 unit tests — mock LLM transport (no network)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from telco_churn.chat.loop import ChurnChatPipeline, _canon_value, _grounded_number
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


class TestCanonFolding:
    """Regression: the LLM copies user wording verbatim, so canon must fold it.

    Two defects met here. (1) The extraction grammar restricted filter VALUES
    to FEATURE_NAMES, so constrained decoding could only emit nonsense such as
    {"Internet_Service": "Dual"} — fixed in llm_client.churn_schema. (2) Even
    with a free-string grammar the copied words carry punctuation and context
    ('Fiber-optic', 'a fiber optic customer'), so canon folds punctuation and
    matches whole phrases, longest canon key first.
    """

    @pytest.mark.parametrize(
        ("feature", "value", "canonical"),
        [
            ("Internet_Service", "Fiber-optic", "Fiber optic"),
            ("Internet_Service", "fiber_optic", "Fiber optic"),
            ("Internet_Service", "FIBER  OPTIC", "Fiber optic"),
            ("Internet_Service", "fibre optic", "Fiber optic"),
            ("Internet_Service", "DSL", "DSL"),
            ("Contract", "month-to-month", "Month-to-month"),
            ("Contract", "Month-to-month", "Month-to-month"),
            ("Payment_Method", "credit card (automatic)", "Credit card (automatic)"),
            ("Payment_Method", "bank transfer automatic", "Bank transfer (automatic)"),
            ("Paperless_Billing", "yes!", "Yes"),
            ("Online_Security", "No internet service", "No internet service"),
            ("Internet_Service", "Fiber optic customer", "Fiber optic"),
            ("Internet_Service", "a fiber-optic customer", "Fiber optic"),
            ("Contract", "on a month-to-month contract", "Month-to-month"),
            (
                "Payment_Method",
                "credit card automatic payment",
                "Credit card (automatic)",
            ),
            ("Tech_Support", "yes please", "Yes"),
            ("Tech_Support", "I am not subscribed", "No"),
            ("Paperless_Billing", "no thanks", "No"),
            ("Dual", "no phone service", "No phone service"),
            ("tenure", "about 24 months", "24"),
            ("Monthly_Charges", "$40.50 a month", "40.50"),
        ],
    )
    def test_folds_punctuation_and_case(
        self, feature: str, value: str, canonical: str
    ) -> None:
        assert _canon_value(feature, value) == canonical

    def test_reported_fiber_optic_turn_now_predicts(self, artifact) -> None:
        client = _client(
            {
                "target_features": ["Internet_Service", "Contract"],
                "filters": {
                    "Internet_Service": "Fiber-optic",
                    "Contract": "month-to-month",
                },
                "out_of_scope": False,
            }
        )
        pipeline = ChurnChatPipeline(artifact, client)
        reply = pipeline.handle(
            "s8", "Will a fiber-optic customer on a month-to-month contract churn?"
        )
        assert reply.payload is not None
        assert pipeline._store.snapshot("s8") == {
            "Internet_Service": "Fiber optic",
            "Contract": "Month-to-month",
        }

    def test_contextual_phrase_emission_now_predicts(self, artifact) -> None:
        """The grammar fix lets the model copy phrases; canon absorbs context."""
        client = _client(
            {
                "target_features": ["Internet_Service", "Contract"],
                "filters": {
                    "Internet_Service": "Fiber optic customer",
                    "Contract": "on a month-to-month contract",
                },
                "out_of_scope": False,
            }
        )
        pipeline = ChurnChatPipeline(artifact, client)
        reply = pipeline.handle(
            "s9",
            "Will a Fiber optic customer on a month-to-month contract churn?",
        )
        assert reply.payload is not None
        assert pipeline._store.snapshot("s9") == {
            "Internet_Service": "Fiber optic",
            "Contract": "Month-to-month",
        }


class TestRepairRetry:
    """A canon miss triggers exactly one menu-echoing repair attempt."""

    def test_mis_keyed_filter_is_repaired(self, artifact) -> None:
        bad: dict[str, Any] = {
            "target_features": ["Online_Security"],
            "filters": {"Monthly_Charges": "no online security"},
            "out_of_scope": False,
        }
        good: dict[str, Any] = {
            "target_features": ["Online_Security"],
            "filters": {"Online_Security": "No"},
            "out_of_scope": False,
        }
        transport = FakeExtractionTransport(candidates=[bad, good])
        pipeline = ChurnChatPipeline(artifact, LlmClient(transport=transport))
        reply = pipeline.handle(
            "s10", "a customer without online security — will they churn?"
        )
        assert reply.payload is not None
        assert pipeline._store.snapshot("s10") == {"Online_Security": "No"}
        assert len(transport.calls) == 2

    def test_repair_is_bounded_and_still_graceful(self, artifact) -> None:
        bad: dict[str, Any] = {
            "target_features": ["tenure"],
            "filters": {"tenure": "weeks"},
            "out_of_scope": False,
        }
        transport = FakeExtractionTransport(bad)
        pipeline = ChurnChatPipeline(artifact, LlmClient(transport=transport))
        reply = pipeline.handle("s11", "their tenure is measured in weeks")
        assert reply.payload is None
        assert "valid value for tenure" in reply.text
        assert len(transport.calls) == 2  # initial attempt + one repair, no more


class TestNumericGrounding:
    """Fabricated numeric fillers must never enter the classifier profile.

    The digits-only grammar for numeric slots turns a mis-slotted stray fact
    into an invented filler ("0") that canonicalisation happily accepts; a
    numeric filter is kept only when the user's own words carry the number.
    """

    @pytest.mark.parametrize(
        ("value", "user_text", "expected"),
        [
            ("24", "with us for about 24 months", True),
            ("40.50", "pays $40.50 a month", True),
            ("40.5", "pays $40.50 a month", True),
            ("0", "paying by mailed check, no numbers here", False),
            ("0", "pays 40 a month", False),
        ],
    )
    def test_grounding_rule(self, value: str, user_text: str, expected: bool) -> None:
        assert _grounded_number(value, user_text) is expected

    def test_ungrounded_number_is_dropped(self, artifact) -> None:
        client = _client(
            {
                "target_features": ["Payment_Method", "Online_Security"],
                "filters": {
                    "Payment_Method": "Mailed check",
                    "Monthly_Charges": "0",
                },
                "out_of_scope": False,
            }
        )
        pipeline = ChurnChatPipeline(artifact, client)
        reply = pipeline.handle(
            "s12",
            "a customer paying by mailed check without online security — "
            "do they churn?",
        )
        assert reply.payload is not None
        assert pipeline._store.snapshot("s12") == {"Payment_Method": "Mailed check"}

    def test_grounded_number_is_kept(self, artifact) -> None:
        client = _client(
            {
                "target_features": ["tenure"],
                "filters": {"tenure": "24"},
                "out_of_scope": False,
            }
        )
        pipeline = ChurnChatPipeline(artifact, client)
        reply = pipeline.handle("s13", "with us for about 24 months")
        assert reply.payload is not None
        assert pipeline._store.snapshot("s13") == {"tenure": "24"}

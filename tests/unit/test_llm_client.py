"""Unit tests for the extraction seam: grammar shape + transport parsing."""

from __future__ import annotations

from typing import Any

import pytest

from telco_churn.chat.schemas import (
    FEATURE_NAMES,
    FEATURE_VALUE_MENUS,
    NUMERIC_FEATURE_NAMES,
)
from telco_churn.serving.llm_client import (
    NUMERIC_PATTERN,
    FakeExtractionTransport,
    LlmClient,
    churn_schema,
)


class TestChurnSchema:
    """The grammar is the contract the small model is decoded against.

    Regression history: filter VALUES were once restricted to
    ``enum FEATURE_NAMES``, so grammar-constrained decoding could only emit
    feature names as values (live failure: ``{"Internet_Service": "Dual"}``).
    Free strings then let the model invent values or mis-key them (live
    failure: ``{"Monthly_Charges": "no online security"}``, and a digits-less
    text dump into a numeric slot). Now every categorical slot carries its
    dataset menu and numeric slots carry a digits-only pattern, so both
    mis-keys are unrepresentable in the grammar.
    """

    def test_filter_keys_are_closed_vocabulary(self) -> None:
        schema: dict[str, Any] = churn_schema()
        filters = schema["properties"]["filters"]
        assert filters["additionalProperties"] is False
        assert set(filters["properties"]) == set(FEATURE_NAMES)

    def test_categorical_slots_are_menu_constrained(self) -> None:
        schema: dict[str, Any] = churn_schema()
        slots = schema["properties"]["filters"]["properties"]
        for name, menu in FEATURE_VALUE_MENUS.items():
            assert slots[name] == {"enum": list(menu)}

    def test_numeric_slots_are_digit_patterned(self) -> None:
        schema: dict[str, Any] = churn_schema()
        slots = schema["properties"]["filters"]["properties"]
        for name in NUMERIC_FEATURE_NAMES:
            assert slots[name] == {"type": "string", "pattern": NUMERIC_PATTERN}

    def test_target_features_stay_closed_vocabulary(self) -> None:
        schema: dict[str, Any] = churn_schema()
        items = schema["properties"]["target_features"]["items"]
        assert items["enum"] == list(FEATURE_NAMES)

    def test_top_level_shape_is_strict(self) -> None:
        schema: dict[str, Any] = churn_schema()
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["target_features", "filters", "out_of_scope"]


class TestExtractJson:
    def test_parses_transport_candidate(self) -> None:
        candidate: dict[str, Any] = {
            "target_features": ["Internet_Service"],
            "filters": {"Internet_Service": "Fiber optic customer"},
            "out_of_scope": False,
        }
        client = LlmClient(transport=FakeExtractionTransport(candidate))
        assert client.extract_json([{"role": "user", "content": "hi"}]) == candidate

    def test_rejects_numeric_top_level_injection(self) -> None:
        client = LlmClient(
            transport=FakeExtractionTransport(
                {
                    "target_features": [],
                    "filters": {},
                    "out_of_scope": False,
                    "predicted_churn": 0.9,
                }
            )
        )
        with pytest.raises(ValueError):
            client.extract_json([{"role": "user", "content": "hi"}])

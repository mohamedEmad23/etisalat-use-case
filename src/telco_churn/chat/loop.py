"""The extract-only tool loop: extract → validate → predict → compose.

Every numeric reaching the user is interpolated from the prediction
tool's payload; the LLM never restates or transforms numbers, and a
malformed candidate never corrupts session state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import polars as pl

from telco_churn.chat.prompts import extraction_messages, retry_messages
from telco_churn.chat.schemas import FeatureRequest
from telco_churn.chat.session import SessionStore
from telco_churn.config import get_settings
from telco_churn.data.clean import clean
from telco_churn.data.ingest import load_churn_csv
from telco_churn.model.artifact import ChurnArtifact, load_artifact
from telco_churn.model.explain import TopDrivers
from telco_churn.serving.llm_client import LlmClient

_ON_SUITE = {"yes", "y", "true", "subscribed"}
_OFF_SUITE = {"no", "n", "false", "not subscribed"}
_NOT_INTERNET = {"no internet service"}
_NOT_PHONE = {"no phone service"}
_CAT_CANON: dict[str, dict[str, str]] = {
    "gender": {"male": "Male", "m": "Male", "female": "Female", "f": "Female"},
    "Internet_Service": {
        "dsl": "DSL",
        "fiber": "Fiber optic",
        "fiber optic": "Fiber optic",
        "no": "No",
    },
    "Contract": {
        "month-to-month": "Month-to-month",
        "one year": "One year",
        "two year": "Two year",
    },
    "Payment_Method": {
        "mailed check": "Mailed check",
        "electronic check": "Electronic check",
        "credit card": "Credit card (automatic)",
        "bank transfer": "Bank transfer (automatic)",
    },
}
NUMERIC_FEATURES = frozenset({"tenure", "Monthly_Charges", "Total_Charges"})
YN_FEATURES = frozenset(
    {
        "Is_Married",
        "Dependents",
        "Phone_Service",
        "Dual",
        "Online_Security",
        "Online_Backup",
        "Device_Protection",
        "Tech_Support",
        "Streaming_TV",
        "Streaming_Movies",
        "Paperless_Billing",
    }
)


@dataclass(frozen=True, slots=True)
class TurnReply:
    """One turn's result: text plus the tool payload its numerics came from."""

    text: str
    payload: dict[str, object] | None


def load_default_artifact() -> ChurnArtifact:
    settings = get_settings()
    paths = sorted(
        p for p in settings.model_registry_dir.glob("churn_artifact_seed*.joblib")
    )
    if not paths:
        raise FileNotFoundError(
            f"no churn artifact under {settings.model_registry_dir} — run train.py first"
        )
    return load_artifact(paths[0])


def _canon_value(feature: str, value: str) -> str | None:
    """Map an LLM-copied user word to a canonical dataset value (or None)."""
    needle = value.strip().lower()
    if feature in _CAT_CANON:
        return _CAT_CANON[feature].get(needle)
    if feature == "Senior_Citizen":
        if needle in _ON_SUITE:
            return "Yes"
        if needle in _OFF_SUITE:
            return "No"
        return None
    if feature in YN_FEATURES:
        if needle in _NOT_INTERNET:
            return "No internet service"
        if needle in _NOT_PHONE:
            return "No phone service"
        if needle in _ON_SUITE:
            return "Yes"
        if needle in _OFF_SUITE:
            return "No"
        return None
    if feature in NUMERIC_FEATURES:
        digits = re.search(r"\d+(?:\.\d+)?", value)
        return digits.group(0) if digits else None
    return value.strip()


class ChurnChatPipeline:
    """Wires the LLM extraction seam, session store, model + SHAP tools."""

    def __init__(
        self,
        artifact: ChurnArtifact,
        llm: LlmClient,
        store: SessionStore | None = None,
        threshold: float | None = None,
    ) -> None:
        self._artifact = artifact
        self._llm = llm
        self._store = store or SessionStore()
        self._baseline = _baseline_frame()
        self._drivers = TopDrivers(artifact.pipeline, k=3)
        self._threshold = artifact.threshold if threshold is None else threshold

    def handle(self, session_id: str, user_text: str) -> TurnReply:
        state = self._store.get_or_create(session_id)
        try:
            candidate = self._llm.extract_json(extraction_messages(user_text))
            request = FeatureRequest.model_validate(candidate)
        except ValueError:
            try:
                candidate = self._llm.extract_json(retry_messages(user_text))
                request = FeatureRequest.model_validate(candidate)
            except ValueError:
                return TurnReply(
                    "I could not map that to a churn question with the allowed "
                    "feature vocabulary — no prediction was made. Try naming "
                    "features like contract type, internet service, or tenure.",
                    None,
                )
        if request.out_of_scope:
            return TurnReply(
                "I only answer customer-churn questions about a subscriber "
                "profile (for example: contract type, internet service, "
                "tenure). Try: 'Will a fiber-optic customer on month-to-month "
                "contract churn?'",
                None,
            )
        canonical: dict[str, str] = {}
        for feature, value in request.filters.items():
            clean_value = _canon_value(feature.value, str(value))
            if clean_value is None:
                return TurnReply(
                    f"I could not read a valid value for {feature.value} — no "
                    "prediction was made. Try words like 'yes', 'Month-to-month', "
                    "'Fiber optic', or numeric amounts.",
                    None,
                )
            canonical[feature.value] = clean_value
        state.features.update(canonical)
        profile = _profile_row(self._baseline, state.features)
        prob = self._artifact.calibrated_proba(profile)[0]
        churn = prob >= self._threshold
        drivers = self._drivers.top3(profile)
        driver_rows: list[dict[str, str | float]] = [
            {"feature": d.feature, "contribution": float(f"{d.contribution:.4f}")}
            for d in drivers
        ]
        payload: dict[str, object] = {
            "churn_probability": float(f"{prob:.4f}"),
            "churn_label": "Churner" if churn else "Not churner",
            "drivers": driver_rows,
        }
        feature_view = ", ".join(f"{f}={v}" for f, v in sorted(state.features.items()))
        driver_view = "; ".join(
            f"{entry['feature']} weight {entry['contribution']}"
            for entry in driver_rows
        )
        text = (
            f"Estimated churn probability {prob:.4f} — "
            f"{payload['churn_label']} at threshold {self._threshold:.4f}. "
            f"Profile used: {feature_view or 'baseline customer'}. "
            f"Top drivers: {driver_view}."
        )
        return TurnReply(text, payload)


def _baseline_frame() -> pd.DataFrame:
    """One baseline row (median numerics / mode categoricals) from the data."""
    frame = clean(load_churn_csv())  # ingest already drops customerID
    row: dict[str, object] = {}
    for column in [c for c in frame.columns if c not in ("Churn", "avg_monthly")]:
        if column in NUMERIC_FEATURES:
            mean_value = float(frame[column].cast(pl.Float64).mean())  # type: ignore[arg-type]
            row[column] = mean_value
        else:
            row[column] = frame[column].drop_nulls().mode()[0]
    total_charges = float(str(row["Total_Charges"]))
    tenure_median = float(str(row["tenure"]))
    row["avg_monthly"] = total_charges / tenure_median if tenure_median > 0 else None
    return pd.DataFrame([row])


def _profile_row(baseline: pd.DataFrame, features: dict[str, str]) -> pd.DataFrame:
    """Baseline row overridden by the session's accumulated canonical values."""
    row = baseline.iloc[0].to_dict()
    for feature, value in features.items():
        if feature in NUMERIC_FEATURES:
            row[feature] = float(value)
        elif feature == "Senior_Citizen":
            row[feature] = value == "Yes"
        else:
            row[feature] = value
    total_charges = float(str(row["Total_Charges"]))
    tenure = float(str(row["tenure"]))
    row["avg_monthly"] = total_charges / tenure if tenure > 0 else None
    return pd.DataFrame([row])

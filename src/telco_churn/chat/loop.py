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

from telco_churn.chat.prompts import (
    extraction_messages,
    repair_messages,
    retry_messages,
)
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
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(value: str) -> str:
    """Fold wording so canon lookups survive punctuation and case drift.

    The extract-only design has the model copy the user's own words, so
    'Fiber-optic', 'fiber_optic', 'FIBER  OPTIC' and 'fibre optic' must all
    reach the same canonical dataset value as 'Fiber optic'.
    """
    return _NON_ALNUM.sub(" ", value.strip().lower()).strip()


# Keys are in _normalize form; values are the dataset's canonical strings.
_CAT_CANON: dict[str, dict[str, str]] = {
    "gender": {"male": "Male", "m": "Male", "female": "Female", "f": "Female"},
    "Internet_Service": {
        "dsl": "DSL",
        "fiber": "Fiber optic",
        "fiber optic": "Fiber optic",
        "fibre": "Fiber optic",
        "fibre optic": "Fiber optic",
        "fiberoptic": "Fiber optic",
        "no": "No",
    },
    "Contract": {
        "month to month": "Month-to-month",
        "one year": "One year",
        "two year": "Two year",
    },
    "Payment_Method": {
        "mailed check": "Mailed check",
        "electronic check": "Electronic check",
        "e check": "Electronic check",
        "credit card": "Credit card (automatic)",
        "credit card automatic": "Credit card (automatic)",
        "bank transfer": "Bank transfer (automatic)",
        "bank transfer automatic": "Bank transfer (automatic)",
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


def _contains_phrase(tokens: list[str], phrase: str) -> bool:
    """True when phrase's tokens appear contiguously inside tokens."""
    phrase_tokens = phrase.split()
    span = len(phrase_tokens)
    return any(
        tokens[i : i + span] == phrase_tokens for i in range(len(tokens) - span + 1)
    )


def _suite_hit(tokens: list[str]) -> str | None:
    """Yes/No from whole-word hits, refusing genuinely ambiguous phrasing."""
    on_hit = bool(_ON_SUITE.intersection(tokens))
    off_hit = bool(_OFF_SUITE.intersection(tokens))
    if on_hit and not off_hit:
        return "Yes"
    if off_hit and not on_hit:
        return "No"
    return None


def _canon_value(feature: str, value: str) -> str | None:
    """Map an LLM-copied user phrase to a canonical dataset value (or None).

    Exact lookup runs first; a whole-word containment pass then folds longer
    phrasings ('a fiber optic customer', 'on a month-to-month contract') onto
    the canon entry, longest key first so 'credit card automatic' wins over
    'credit card'. Single-letter keys stay safe because matching is per word
    ('male' never fires inside 'female').
    """
    needle = _normalize(value)
    tokens = needle.split()
    if feature in _CAT_CANON:
        table = _CAT_CANON[feature]
        if needle in table:
            return table[needle]
        for key in sorted(table, key=lambda k: len(k.split()), reverse=True):
            if _contains_phrase(tokens, key):
                return table[key]
        return None
    if feature == "Senior_Citizen":
        return _suite_hit(tokens)
    if feature in YN_FEATURES:
        if any(_contains_phrase(tokens, phrase) for phrase in _NOT_INTERNET):
            return "No internet service"
        if any(_contains_phrase(tokens, phrase) for phrase in _NOT_PHONE):
            return "No phone service"
        if _contains_phrase(tokens, "not subscribed"):
            return "No"
        return _suite_hit(tokens)
    if feature in NUMERIC_FEATURES:
        digits = re.search(r"\d+(?:\.\d+)?", value)
        return digits.group(0) if digits else None
    return value.strip()


_NUMBER_RUN = re.compile(r"\d+(?:[.,]\d+)*")


def _grounded_number(value: str, user_text: str) -> bool:
    """Return True when the user's own words carry this numeric value.

    The digits-only grammar for numeric slots forces a fabricated filler
    (typically "0") when the model mis-slots a stray fact there; such
    numbers must not enter the classifier profile.
    """
    try:
        wanted = float(value.replace(",", ""))
    except ValueError:
        return False
    return any(
        float(run.replace(",", "")) == wanted for run in _NUMBER_RUN.findall(user_text)
    )


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

    def _canonicalise(
        self, request: FeatureRequest
    ) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """Map the request's filters to dataset values; report the misses."""
        canonical: dict[str, str] = {}
        problems: list[tuple[str, str]] = []
        for feature, value in request.filters.items():
            clean_value = _canon_value(feature.value, str(value))
            if clean_value is None:
                problems.append((feature.value, str(value)))
            else:
                canonical[feature.value] = clean_value
        return canonical, problems

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
        canonical, problems = self._canonicalise(request)
        if problems:
            # One bounded repair attempt: echo the menu grammar back and let
            # the model re-place the facts, then canonicalise again.
            problem_view = [f"{feature}={value!r}" for feature, value in problems]
            try:
                repaired = self._llm.extract_json(
                    repair_messages(user_text, candidate, problem_view)
                )
                repaired_request = FeatureRequest.model_validate(repaired)
            except ValueError:
                repaired_request = None
            if repaired_request is not None and not repaired_request.out_of_scope:
                canonical, problems = self._canonicalise(repaired_request)
        if problems:
            return TurnReply(
                f"I could not read a valid value for {problems[0][0]} — no "
                "prediction was made. Try words like 'yes', 'Month-to-month', "
                "'Fiber optic', or numeric amounts.",
                None,
            )
        # Numeric grounding: drop numeric filters the user's words do not
        # support — a digits-only slot otherwise lets the model inject a
        # fabricated filler ("0") when it mis-slots a stray fact.
        for name in list(canonical):
            if name in NUMERIC_FEATURES and not _grounded_number(
                canonical[name], user_text
            ):
                del canonical[name]
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

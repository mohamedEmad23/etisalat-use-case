"""Extraction prompt + few-shot examples for the Extract-only LLM.

The prompt is prompt-artifact reviewed: it names the closed feature
vocabulary, forbids numerics, and forces exactly one FeatureRequest JSON
object per turn (see openspec chat-pipeline delta).
"""

from __future__ import annotations

from telco_churn.chat.schemas import FEATURE_NAMES

SCHEMA_HINT = '{"target_features": [...], "filters": {}, "out_of_scope": false}'

_EXTRACT_SYSTEM = """You are the extraction layer of a churn-prediction chatbot.
Your ONLY job is to convert the user's latest message into ONE JSON object.

Rules:
1. Reply with exactly one JSON object of the shape:
   {SCHEMA_HINT}
2. "target_features" is a list from this closed vocabulary only:
   {VOCABULARY}
3. "filters" may copy the user's own categorical words (e.g. "Month-to-month")
   as plain strings. NEVER write numbers, never transform what the user said.
4. Numeric facts (tenure months, charges) belong in filters only as the
   user's exact words; the system reads numbers from its own data, not from
   you.
5. If the message is not about a customer churn profile (weather, jokes,
   math homework), set "out_of_scope": true and leave the lists empty.
6. Output ONLY the JSON object. No prose, no markdown, no explanations.

Example 1 — single request:
User: Will a fiber-optic customer on month-to-month contract churn?
Assistant: {{"target_features": ["Internet_Service", "Contract"], "filters": {{"Internet_Service": "Fiber optic", "Contract": "Month-to-month"}}, "out_of_scope": false}}

Example 2 — refinement, keeps context decisions to the system, user adds a feature:
User: and what if they also had no tech support?
Assistant: {{"target_features": ["Tech_Support", "tenure"], "filters": {{}}, "out_of_scope": false}}

Example 3 — numeric mentioned by the user (copied as words, not synthesized):
User: my customer has tenure 24 months, is churn likely?
Assistant: {{"target_features": ["tenure", "Monthly_Charges"], "filters": {{"tenure": "24"}}, "out_of_scope": false}}

Example 4 — out of scope:
User: What's the weather in Cairo?
Assistant: {{"target_features": [], "filters": {{}}, "out_of_scope": true}}

Remember: one JSON object only; unknown feature names or any new numeric
field are rejected."""

_EXTRACT_SYSTEM = _EXTRACT_SYSTEM.replace("{SCHEMA_HINT}", SCHEMA_HINT).replace(
    "{VOCABULARY}", ", ".join(FEATURE_NAMES)
)

_RETRY_SYSTEM = (
    "Your previous reply violated the extraction JSON schema. Repeat the "
    f"answer under the SAME grammar rules and the SAME closed vocabulary: "
    f"{SCHEMA_HINT} — nothing else."
)

FEW_SHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Will a fiber-optic customer on month-to-month contract churn?",
        (
            '{"target_features": ["Internet_Service", "Contract"], '
            '"filters": {"Internet_Service": "Fiber optic", '
            '"Contract": "Month-to-month"}, "out_of_scope": false}'
        ),
    ),
    (
        "What's the weather in Cairo?",
        '{"target_features": [], "filters": {}, "out_of_scope": true}',
    ),
)


def extraction_messages(user_text: str) -> list[dict[str, str]]:
    """Build chat messages for the first extraction attempt."""
    return [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": user_text},
    ]


def retry_messages(user_text: str) -> list[dict[str, str]]:
    """Build chat messages for the same-grammar retry (no free-form fallback)."""
    return [
        {"role": "system", "content": _RETRY_SYSTEM},
        {"role": "user", "content": user_text},
    ]


EXTRACTION_SYSTEM_PROMPT = _EXTRACT_SYSTEM

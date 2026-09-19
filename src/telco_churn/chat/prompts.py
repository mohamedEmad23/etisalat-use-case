"""Extraction prompt + few-shot examples for the Extract-only LLM.

The prompt is prompt-artifact reviewed: it names the closed feature
vocabulary, forbids numerics, and forces exactly one FeatureRequest JSON
object per turn (see openspec chat-pipeline delta).
"""

from __future__ import annotations

import json

from telco_churn.chat.schemas import FEATURE_NAMES, FEATURE_VALUE_MENUS

SCHEMA_HINT = '{"target_features": [...], "filters": {}, "out_of_scope": false}'

_MENU_LINES = "\n".join(
    f"   {name}: {' | '.join(FEATURE_VALUE_MENUS[name])}"
    for name in FEATURE_NAMES
    if name in FEATURE_VALUE_MENUS
)
_MENU_LINES += (
    "\n   tenure / Monthly_Charges / Total_Charges: bare digits only "
    "(examples: 24, 40.50) — no words, no units, no currency symbols"
)

_EXTRACT_SYSTEM = """You are the extraction layer of a churn-prediction chatbot.
Your ONLY job is to convert the user's latest message into ONE JSON object.

Rules:
1. Reply with exactly one JSON object of the shape:
   {SCHEMA_HINT}
2. "target_features" is a list from this closed vocabulary only:
   {VOCABULARY}
3. "filters" restate facts the user actually stated. Each categorical feature
   accepts ONLY the values listed below — pick the closest one. Never invent a
   filter the user did not state, and never transform what the user said.
   Allowed values:
{MENUS}
4. Numeric facts (tenure months, charges) go into their numeric slot as
   bare digits (e.g. "about 24 months" becomes 24; "$40.50" becomes 40.50).
   Numeric slots never contain words or units; the system reads numbers
   from its own data, not from you.
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

_EXTRACT_SYSTEM = (
    _EXTRACT_SYSTEM.replace("{SCHEMA_HINT}", SCHEMA_HINT)
    .replace("{VOCABULARY}", ", ".join(FEATURE_NAMES))
    .replace("{MENUS}", _MENU_LINES)
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


_REPAIR_SYSTEM = """Your previous extraction put values in slots they cannot map to.
Re-emit the SAME facts as ONE JSON object of the shape:
{SCHEMA_HINT}

Rules:
- Use ONLY the allowed values below; numeric slots take bare digits only.
- Drop any filter the user did not actually state; never invent features.
- Leave out anything you cannot map.

Allowed values:
{MENUS}
Output ONLY the JSON object."""

_REPAIR_SYSTEM = _REPAIR_SYSTEM.replace("{SCHEMA_HINT}", SCHEMA_HINT).replace(
    "{MENUS}", _MENU_LINES
)


def repair_messages(
    user_text: str, previous: dict[str, object], problems: list[str]
) -> list[dict[str, str]]:
    """Build chat messages for the bounded repair retry after a canon miss."""
    return [
        {"role": "system", "content": _REPAIR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User said: {user_text}\n"
                f"Your previous extraction: {json.dumps(previous)}\n"
                f"Invalid slots: {'; '.join(problems)}\n"
                "Re-emit the corrected JSON now."
            ),
        },
    ]


EXTRACTION_SYSTEM_PROMPT = _EXTRACT_SYSTEM

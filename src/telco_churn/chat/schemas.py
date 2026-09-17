"""Extract-only LLM contract: schema for the one feature request per turn.

Glossary terms (CONTEXT.md): Extract-only LLM, FeatureRequest. This module is
the grammar the LLM must satisfy; nothing numeric can ever pass through it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# The 19 non-identifier dataset features in their CSV-header (trimmed)
# spelling. customerID and the Churn label are excluded by definition.
class FeatureName(StrEnum):
    """Closed vocabulary: the only feature names an extract may ever mention."""

    GENDER = "gender"
    SENIOR_CITIZEN = "Senior_Citizen"
    IS_MARRIED = "Is_Married"
    DEPENDENTS = "Dependents"
    TENURE = "tenure"
    PHONE_SERVICE = "Phone_Service"
    DUAL = "Dual"
    INTERNET_SERVICE = "Internet_Service"
    ONLINE_SECURITY = "Online_Security"
    ONLINE_BACKUP = "Online_Backup"
    DEVICE_PROTECTION = "Device_Protection"
    TECH_SUPPORT = "Tech_Support"
    STREAMING_TV = "Streaming_TV"
    STREAMING_MOVIES = "Streaming_Movies"
    CONTRACT = "Contract"
    PAPERLESS_BILLING = "Paperless_Billing"
    PAYMENT_METHOD = "Payment_Method"
    MONTHLY_CHARGES = "Monthly_Charges"
    TOTAL_CHARGES = "Total_Charges"


FEATURE_NAMES: tuple[str, ...] = tuple(member.value for member in FeatureName)


class FeatureRequest(BaseModel):
    """Schema of the single LLM output per turn — zero numeric fields.

    ``out_of_scope=True`` short-circuits prediction; ``filters`` restates
    profile facts as equality clauses whose values the LLM takes from the
    user's own words (no synthesis); ``target_features`` names what to
    highlight. Any numeric/unknown field breaks strict validation, which is
    the intended reject.
    """

    model_config = ConfigDict(extra="forbid")

    target_features: Annotated[
        list[FeatureName],
        Field(default_factory=list, description="Requested feature columns"),
    ]
    filters: Annotated[
        dict[FeatureName, str],
        Field(
            default_factory=dict,
            description="Feature equality clauses from the utterance",
        ),
    ]
    out_of_scope: Annotated[
        bool, Field(default=False, description="Not a churn-profile question")
    ]


def is_schema_candidate(obj: object) -> bool:
    """True when the object would validate against the extraction schema."""
    if not isinstance(obj, dict):
        return False
    try:
        FeatureRequest.model_validate(obj)
        return True
    except ValueError:
        return False

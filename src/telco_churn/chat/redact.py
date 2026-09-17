"""PII redaction scrubber applied to everything the system logs."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(
    r"(?<![\w-])\+?\d[\d\s().-]{7,}\d(?!\d)"  # 7+ digit clusters with separators
)
_CUST_ID = re.compile(r"\d{4}[A-Z]{5}")  # e.g. 7590-VHVEG style suffix part


def redact(text: str) -> str:
    """Scrub emails, phone_nr clusters, customer-ID tokens from `text`."""
    out = _EMAIL.sub("[EMAIL]", text)
    out = _CUST_ID.sub("[CUSTOMER_ID]", out)
    out = _PHONE.sub("[PHONE]", out)
    return out

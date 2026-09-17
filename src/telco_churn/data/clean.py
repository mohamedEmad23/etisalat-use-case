"""Tasks 2.2–2.3 — Cleaning and feature engineering.

Documented rules (data-pipeline spec):
- Total_Charges → numeric BEFORE any split. 11 blank rows (all tenure = 0, all
  Churn = No) get 0.0: tenure 0 means no billing cycle has completed yet, so no
  total charge exists. Deterministic pre-split coercion, never imputed from
  other rows.
- Sentinel strings "No phone service" / "No internet service" are kept distinct
  from plain "No" (they carry real semantics about add-on availability); only
  whitespace variants are trimmed.
- Senior_Citizen 0/1 → bool.
- avg_monthly = Total_Charges / tenure, defined only where tenure > 0; the
  tenure-0 rows get null rather than a fabricated value.
"""

from __future__ import annotations

import polars as pl

TOTAL_CHARGES = "Total_Charges"
TENURE = "tenure"
SENIOR_CITIZEN = "Senior_Citizen"
AVG_MONTHLY = "avg_monthly"


def clean(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the documented cleaning to the ingested frame (ingest output)."""
    return engineer_avg_monthly(coerce_total_charges(normalize_categories(frame)))


def normalize_categories(frame: pl.DataFrame) -> pl.DataFrame:
    """Trim whitespace on every string column and map Senior_Citizen 0/1 to bool."""
    string_cols = [name for name, dtype in frame.schema.items() if dtype == pl.String]
    out = frame.with_columns(
        pl.col(name).str.strip_chars().alias(name) for name in string_cols
    )
    return out.with_columns(pl.col(SENIOR_CITIZEN).cast(pl.Boolean))


def count_blank_total_charges(frame: pl.DataFrame) -> int:
    """Number of raw rows whose string Total_Charges is empty/whitespace."""
    if frame.schema[TOTAL_CHARGES] != pl.String:
        return int(frame.select(pl.col(TOTAL_CHARGES).is_null().sum()).item())
    return int(
        frame.select(
            (
                pl.col(TOTAL_CHARGES).is_null()
                | (pl.col(TOTAL_CHARGES).str.strip_chars() == "")
            ).sum()
        ).item()
    )


def coerce_total_charges(frame: pl.DataFrame) -> pl.DataFrame:
    """Coerce Total_Charges to numeric pre-split under the documented blank-row rule.

    Depending on schema inference the blanks arrive either as empty strings or
    as nulls of a Float64 column; both are the documented blank rows.

    Raises if a blank row violates the documented rule (tenure must be 0), so a
    changed dataset fails loudly instead of being silently imputed.
    """
    if frame.schema[TOTAL_CHARGES] == pl.String:
        blank = pl.col(TOTAL_CHARGES).is_null() | (
            pl.col(TOTAL_CHARGES).str.strip_chars() == ""
        )
        violations = frame.filter(blank & (pl.col(TENURE) != 0))
        if violations.height:
            raise ValueError(
                f"{violations.height} blank Total_Charges row(s) with tenure != 0 violate "
                "the documented blank-row rule (tenure must be 0)"
            )
        return frame.with_columns(
            pl.when(blank)
            .then(0.0)
            .otherwise(pl.col(TOTAL_CHARGES).str.strip_chars())
            .cast(pl.Float64)
            .alias(TOTAL_CHARGES)
        )

    blank = pl.col(TOTAL_CHARGES).is_null()
    violations = frame.filter(blank & (pl.col(TENURE) != 0))
    if violations.height:
        raise ValueError(
            f"{violations.height} null Total_Charges row(s) with tenure != 0 violate "
            "the documented blank-row rule (tenure must be 0)"
        )
    return frame.with_columns(pl.col(TOTAL_CHARGES).fill_null(0.0).alias(TOTAL_CHARGES))


def engineer_avg_monthly(frame: pl.DataFrame) -> pl.DataFrame:
    """avg_monthly = Total_Charges / tenure where tenure > 0, else null."""
    return frame.with_columns(
        pl.when(pl.col(TENURE) > 0)
        .then(pl.col(TOTAL_CHARGES) / pl.col(TENURE))
        .otherwise(None)
        .cast(pl.Float64)
        .round(4)
        .alias(AVG_MONTHLY)
    )

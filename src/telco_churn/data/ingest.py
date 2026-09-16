"""Task 2.1 — Faithful raw ingestion: header trim, shape assert, customerID drop at load."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from telco_churn.config import AppConfig

CUSTOMER_ID_COLUMN = "customerID"
EXPECTED_ROWS = 7_043
EXPECTED_COLUMNS = 21


def load_churn_csv(
    path: str | Path | None = None, *, df: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Load the raw Telco CSV exactly as provided and apply load-time hygiene.

    Rules (data-pipeline spec):
    - trim every header name (the raw file has `Senior_Citizen ` with a trailing space)
    - assert the documented 7,043 x 21 shape
    - drop customerID at load: it is never a model feature and never in payloads
    """
    if df is None:
        if path is None:
            settings = AppConfig()  # type: ignore[call-arg]  # pydantic-settings populates from env
            path = settings.data_csv_path
        frame = pl.read_csv(path)
    else:
        frame = df
    return prepare_raw_frame(frame)


def prepare_raw_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply load-time hygiene to an already-read frame (used by tests with inline fixtures)."""
    trimmed = frame.rename(
        {name: name.strip() for name in frame.columns if name != name.strip()}
    )
    if trimmed.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Unexpected dataset shape {trimmed.shape}; expected "
            f"({EXPECTED_ROWS}, {EXPECTED_COLUMNS})"
        )
    if CUSTOMER_ID_COLUMN not in trimmed.columns:
        raise ValueError(f"Missing required column {CUSTOMER_ID_COLUMN!r}")
    return trimmed.drop(CUSTOMER_ID_COLUMN)

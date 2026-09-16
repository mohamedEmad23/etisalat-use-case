"""Unit tests for the data pipeline (tasks 2.1–2.4) against the real CSV."""

from __future__ import annotations

import polars as pl
import pytest

from telco_churn.data.clean import clean, count_blank_total_charges
from telco_churn.data.ingest import load_churn_csv
from telco_churn.data.split import stratified_folds, stratified_split

SEED = 42


@pytest.fixture(scope="module")
def raw() -> pl.DataFrame:
    return load_churn_csv()


@pytest.fixture(scope="module")
def cleaned(raw: pl.DataFrame) -> pl.DataFrame:
    return clean(raw)


# --- 2.1 ingestion ---


def test_ingest_shape_and_schema(raw: pl.DataFrame) -> None:
    assert raw.shape == (7_043, 20)  # 21 columns minus dropped customerID
    assert "customerID" not in raw.columns
    assert "Senior_Citizen" in raw.columns  # trailing space trimmed, addressable


def test_ingest_is_deterministic(raw: pl.DataFrame, cleaned: pl.DataFrame) -> None:
    assert load_churn_csv().equals(raw)
    assert clean(raw).equals(cleaned)


# --- 2.2 cleaning ---


def test_blank_total_charges_rows_are_eleven(
    raw: pl.DataFrame, cleaned: pl.DataFrame
) -> None:
    assert count_blank_total_charges(raw) == 11
    blanks = cleaned.filter(pl.col("tenure") == 0)
    assert blanks.height == 11
    assert set(blanks["Total_Charges"].to_list()) == {0.0}
    assert set(blanks["Churn"].to_list()) == {"No"}


def test_total_charges_fully_numeric(cleaned: pl.DataFrame) -> None:
    assert cleaned.schema["Total_Charges"] == pl.Float64
    assert cleaned["Total_Charges"].null_count() == 0


def test_senior_citizen_mapped_to_bool(cleaned: pl.DataFrame) -> None:
    assert cleaned.schema["Senior_Citizen"] == pl.Boolean
    assert set(cleaned["Senior_Citizen"].unique().to_list()) <= {True, False}


def test_whitespace_trimmed_and_sentinels_kept(
    raw: pl.DataFrame, cleaned: pl.DataFrame
) -> None:
    security = set(cleaned["Online_Security"].unique().to_list())
    # sentinel kept distinct from plain "No" (add-on availability semantics)
    assert "No internet service" in security
    payment = set(cleaned["Payment_Method"].unique().to_list())
    assert "Mailed check" in payment and " Electronic check" not in payment


# --- 2.3 feature engineering ---


def test_avg_monthly_values(cleaned: pl.DataFrame) -> None:
    row = cleaned.filter(pl.col("tenure") > 0).head(1)
    expected = round(row["Total_Charges"][0] / row["tenure"][0], 4)
    assert row["avg_monthly"][0] == pytest.approx(expected)
    assert cleaned.filter(pl.col("tenure") == 0)["avg_monthly"].null_count() == 11
    assert cleaned.filter(pl.col("tenure") > 0)["avg_monthly"].null_count() == 0


# --- 2.4 split ---


def test_split_is_deterministic_and_leak_free(cleaned: pl.DataFrame) -> None:
    p1 = stratified_split(cleaned, seed=SEED)
    p2 = stratified_split(cleaned, seed=SEED)
    assert p1.train.equals(p2.train)
    assert p1.test.equals(p2.test)
    assert p1.train.height + p1.test.height == cleaned.height


def churn_rate(d: pl.DataFrame) -> float:
    """Share of churners; float() avoids polars' over-broad mean() typing."""
    return float((d["Churn"] == "Yes").sum()) / d.height


def test_split_churn_rate_matches_dataset(cleaned: pl.DataFrame) -> None:
    p = stratified_split(cleaned, seed=SEED)
    rate = churn_rate
    dataset_rate = rate(cleaned)
    # integer-count stratification bounds the achievable closeness per partition
    assert abs(rate(p.train) - dataset_rate) <= 0.02
    assert abs(rate(p.test) - dataset_rate) <= 0.02


def test_five_stratified_folds(cleaned: pl.DataFrame) -> None:
    p = stratified_split(cleaned, seed=SEED)
    folds = stratified_folds(p.train, seed=SEED)
    assert len(folds) == 5
    val_sizes = {val.height for _, val in folds}
    assert max(val_sizes) - min(val_sizes) <= 1
    for _, val in folds:
        assert abs(churn_rate(val) - 0.2654) < 0.02
    # folds are deterministic
    assert stratified_folds(p.train, seed=SEED)[0][1].equals(folds[0][1])


# --- 2.5 documented coercion failure mode ---


def test_coercion_rejects_blank_row_with_tenure(raw: pl.DataFrame) -> None:
    bad = raw.filter(pl.col("Total_Charges").is_null()).with_columns(
        pl.lit(5).alias("tenure")
    )
    if bad.height:  # blank rows exist as nulls (Float64 inference)
        from telco_churn.data.clean import coerce_total_charges

        with pytest.raises(ValueError, match="documented blank-row rule"):
            coerce_total_charges(bad)

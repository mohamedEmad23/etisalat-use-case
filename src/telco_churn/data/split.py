"""Task 2.4 — Deterministic stratified 80/20 split + seeded 5-fold fold builder.

Leakage guardrail (data-pipeline spec): no data-dependent transform is fitted
here — this module only partitions row indices. Every downstream fit happens on
the train portion (or train folds), never on the combined frame.

Polars owns the data; sklearn is used only for deterministic index selection —
the same policy the model seam uses (single pandas/numpy boundary at fit time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.model_selection import (  # type: ignore[import-untyped]
    StratifiedKFold,
    train_test_split,
)

CHURN = "Churn"
TEST_SIZE = 0.2
N_FOLDS = 5


@dataclass(frozen=True)
class Partition:
    """A leak-free 80/20 train/test partition — disjoint index ranges, row order preserved."""

    train: pl.DataFrame  # 80% of the rows, stratified on Churn
    test: pl.DataFrame  # 20% of the rows, stratified on Churn


def stratified_split(
    frame: pl.DataFrame,
    *,
    seed: int,
    test_size: float = TEST_SIZE,
) -> Partition:
    """Deterministic stratified 80/20 split of a churn-labelled frame."""
    pos = frame.select(
        pl.col(CHURN).cast(pl.Int8, strict=False).fill_null(0)
    ).to_series()
    train_idx, test_idx = train_test_split(
        range(frame.height),
        test_size=test_size,
        stratify=pos,
        random_state=seed,
    )
    train = frame[list(train_idx)]
    test = frame[list(test_idx)]
    return Partition(train=train, test=test)


def stratified_folds(
    frame: pl.DataFrame,
    *,
    seed: int,
    n_folds: int = N_FOLDS,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Seeded stratified K-fold over the train portion: (fold_train, fold_val)."""
    pos = frame.select(
        pl.col(CHURN).cast(pl.Int8, strict=False).fill_null(0)
    ).to_series()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (frame[list(train_idx)], frame[list(val_idx)])
        for train_idx, val_idx in skf.split(np.zeros(frame.height), pos)
    ]

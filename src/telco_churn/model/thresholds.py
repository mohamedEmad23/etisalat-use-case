"""Task 3.3 — Operating-point selection: F1(churn)-max subject to precision floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRECISION_FLOOR = 0.55


@dataclass(frozen=True)
class OperatingPoint:
    """Threshold plus the train-CV metrics that justified it."""

    threshold: float
    f1: float
    precision: float
    recall: float


def _curve(
    y: np.ndarray, proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-proba, kind="stable")
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    precision = tp / (tp + fp)
    recall = tp / max(int(y.sum()), 1)
    denom = precision + recall
    f1 = np.where(
        denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0
    )
    return precision, recall, f1


def select_operating_point(
    y: np.ndarray, proba: np.ndarray, *, precision_floor: float = PRECISION_FLOOR
) -> OperatingPoint:
    """Max-F1 threshold over the score curve; floor enforced on precision.

    `proba` must come from out-of-fold TRAIN predictions (spec: train CV only).
    Raises when the floor is unattainable anywhere on the curve (spec: report).
    """
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if proba.shape != y.shape:
        raise ValueError("y and proba shapes differ")
    precision, recall, f1 = _curve(y, proba)
    eligible = (precision >= precision_floor) & (recall > 0)
    if not eligible.any():
        raise ValueError(
            f"precision floor {precision_floor:.2f} unattainable on this score curve — "
            "reporting rather than silently lowering the floor"
        )
    order = np.argsort(-proba, kind="stable")
    sorted_proba = proba[order]
    idx = np.flatnonzero(eligible)
    best = idx[np.argmax(f1[idx] * 1_000_000 + precision[idx])]
    return OperatingPoint(
        threshold=float(sorted_proba[best]),
        f1=float(f1[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
    )

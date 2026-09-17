"""Task 3.8 — Reproducible churn artifact: everything needed to reproduce AUC.

Holds the fitted pipeline, calibrator, operating-point threshold, feature list,
the 3-model comparison table and full metadata (seed, imbalance mechanism,
hyperparameters). Saved with joblib under the configured model_registry_dir;
a reload reproduces test AUC within ±0.01 (spec 3.8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from telco_churn.data.split import stratified_split

LABEL = "Churn"
POS = "Yes"
NEG = "No"
AUC_TOLERANCE = 0.01


def frame_to_xy(frame: pl.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Polars → pandas boundary used by every model seam call."""
    df = frame.to_pandas()
    y = (df[LABEL] == POS).astype(int).to_numpy()
    X = df.drop(columns=[LABEL])
    return X, y


@dataclass
class ChurnArtifact:
    """Serialisable bundle of model + threshold + calibrator + provenance."""

    pipeline: Pipeline
    threshold: float
    calibrator: IsotonicRegression | None
    feature_list: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    comparison: list[dict[str, Any]] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Raw model probabilities."""
        return self.pipeline.predict_proba(X)[:, 1]

    def calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability served to the chat surface."""
        raw = self.predict_proba(X)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict(raw)

    def feature_names(self) -> list[str]:
        """Ordered feature list used at fit time."""
        return self.feature_list

    def save(self, path: Path) -> Path:
        """Persist to joblib; returns the resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        (path.parent / (path.stem + ".meta.json")).write_text(
            json.dumps(
                {
                    "threshold": self.threshold,
                    "metadata": self.metadata,
                    "metrics": self.metrics,
                    "comparison": self.comparison,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path


def load_artifact(path: Path) -> ChurnArtifact:
    """Reload a saved artifact."""
    artifact = joblib.load(path)
    if not isinstance(artifact, ChurnArtifact):
        raise TypeError(f"{path} is not a ChurnArtifact")
    return artifact


def evaluate_against(
    artifact: ChurnArtifact, frame: pl.DataFrame, *, seed: int
) -> float:
    """Re-create the seeded split and score the artifact on TEST — reproduction check."""
    partition = stratified_split(frame, seed=seed)
    X_test, y_test = frame_to_xy(partition.test)
    proba = artifact.predict_proba(X_test)
    return float(roc_auc_score(y_test, proba))

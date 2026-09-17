"""Task 3.5 — Per-prediction top-3 SHAP drivers, mapped back to raw features.

SHAP is computed on the transformed feature space (TreeExplainer for the GBMs,
LinearExplainer for logistic), then transformed columns are aggregated back to
the ORIGINAL dataset columns so a driver name means the same thing to the chat
layer as it does in the data dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Driver:
    """One churn driver: the raw feature and its signed contribution."""

    feature: str
    contribution: float


class TopDrivers:
    """Extracts exactly k top drivers for one prediction from a fitted pipeline."""

    def __init__(self, pipeline: Pipeline, k: int = 3) -> None:
        self.pipeline = pipeline
        self.k = k
        self.pre = pipeline.named_steps["preprocess"]
        self.estimator = pipeline.named_steps["model"]
        self._raw_columns: list[str] = [str(c) for c in self.pre.feature_names_in_]

    def _explainer_for(self, background: np.ndarray) -> Any:
        name = type(self.estimator).__name__
        if name in ("XGBClassifier", "LGBMClassifier"):
            return shap.TreeExplainer(self.estimator)
        return shap.LinearExplainer(self.estimator, background)

    def top3(self, X_row: pd.DataFrame) -> list[Driver]:
        """Return exactly k (feature, contribution) drivers for one row."""
        transformed = self.pre.transform(X_row)
        explainer = self._explainer_for(transformed)
        values = explainer.shap_values(transformed)
        if isinstance(values, list) and len(values) == 2:
            values = values[1]
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:  # new-style shap output (n, features, classes) → class 1
            arr = arr[0, :, 1]
        arr = arr.reshape(-1)
        names = [str(n) for n in self.pre.get_feature_names_out()]
        contrib_by_raw: dict[str, float] = {c: 0.0 for c in self._raw_columns}
        for tname, val in zip(names, arr, strict=True):
            contrib_by_raw[self._raw_for(tname)] += float(val)
        ranked = sorted(contrib_by_raw.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [Driver(feature=f, contribution=c) for f, c in ranked[: self.k]]

    def _raw_for(self, transformed_name: str) -> str:
        """Map 'num__tenure' / 'cat__Contract_Month-to-month' → raw column."""
        body = transformed_name.split("__", 1)[-1]
        best = ""
        for col in self._raw_columns:
            if (body == col or body.startswith(col + "_")) and len(col) > len(best):
                best = col
        return best or body

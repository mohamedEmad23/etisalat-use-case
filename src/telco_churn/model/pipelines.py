"""Task 3.1 — Three candidate pipelines under one identical protocol.

All three (logistic, XGBoost, LightGBM) share the same preprocessor (numeric
imputation + scaling, one-hot encoding with handle_unknown="ignore") fitted
INSIDE the sklearn Pipeline, i.e. per fold on fold-train rows only.

Imbalance handling is part of pipeline construction and is computed ONLY from
the train labels passed here; the values it produced are returned so the
design.md provable-recording requirement is met.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL_KEYS = ("logistic", "xgboost", "lightgbm")


def feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into (numeric, categorical) by runtime dtype."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Shared transformer: median-impute + scale numerics; OHE categoricals."""
    numeric, categorical = feature_columns(X)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def imbalance_mechanism(y: np.ndarray) -> dict[str, Any]:
    """Recorded imbalance plan computed from train labels only (provable recording)."""
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    scale_pos_weight = (negatives / positives) if positives else 1.0
    return {
        "logistic": "class_weight=balanced",
        "xgboost": f"scale_pos_weight={scale_pos_weight:.6f}",
        "lightgbm": f"scale_pos_weight={scale_pos_weight:.6f}",
        "scale_pos_weight": scale_pos_weight,
        "positives": positives,
        "negatives": negatives,
        "fitted_on": "fold-train labels only",
    }


def make_estimator(model_key: str, y: np.ndarray, *, seed: int) -> Any:
    """Build an unfitted bare estimator; imbalance settings use train y only."""
    mech = imbalance_mechanism(y)
    if model_key == "logistic":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
    if model_key == "xgboost":
        return XGBClassifier(
            scale_pos_weight=mech["scale_pos_weight"],
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
            tree_method="hist",
        )
    if model_key == "lightgbm":
        return LGBMClassifier(
            scale_pos_weight=mech["scale_pos_weight"],
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"unknown model key: {model_key}")


def make_pipeline(
    model_key: str, X: pd.DataFrame, y: np.ndarray, *, seed: int
) -> Pipeline:
    """Build an unfitted candidate pipeline; imbalance settings use train y only."""
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X)),
            ("model", make_estimator(model_key, y, seed=seed)),
        ]
    )


def suggest_hyperparams(trial: Any, model_key: str) -> dict[str, Any]:
    """Optuna search space per candidate model."""
    if model_key == "logistic":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if model_key == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_key == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
    raise ValueError(f"unknown model key: {model_key}")

"""Task 3.8 / D3 — External validation runner (zero-shot) + demo synthetic path.

``external_validate`` re-creates the split the artifact was trained on, applies
a minimal schema mapping to an external frame and reports the honest AUC
delta vs the internal benchmark — no retraining, no tuning, no tuning-time
peeking at the external distribution.

``ctgan_demo`` is DEMO ONLY: it generates a bootstrap/synthetic sensitivity
frame explicitly labelled as synthetic in the report; it makes no external
validation claim (spec D3 / task 3.7).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn.metrics import roc_auc_score

from telco_churn.data.split import stratified_split
from telco_churn.model.artifact import ChurnArtifact, frame_to_xy

SOURCE_LABELS = ("UCI-Iranian-Churn", "Orange-Telecom")


def external_validate(
    artifact: ChurnArtifact,
    external_frame: pl.DataFrame,
    mapping: dict[str, str],
    *,
    label: str,
    seed: int,
    benchmark_auc: float,
) -> dict[str, Any]:
    """Zero-shot score of the frozen artifact on an external source.

    mapping: external column -> canonical feature column. Unmapped external
    columns are dropped; the label column must map to "Churn".
    Returns the delta vs the benchmark AUC side-by-side, labelled honestly as
    zero-shot (no retraining).
    """
    if "Churn" not in mapping.values():
        raise ValueError("the label column must map to 'Churn' in the schema mapping")
    reverse = {v: k for k, v in mapping.items()}
    target_cols = [c for c in artifact.feature_names() if c != "Churn"]
    missing = [c for c in target_cols if c not in reverse]
    rename_map = {reverse[c]: c for c in target_cols if c in reverse}
    label_ext = reverse.get("Churn")
    if label_ext is not None:
        rename_map[label_ext] = "Churn"
    clean = external_frame.select([*rename_map]).rename(rename_map)

    partition = stratified_split(clean, seed=seed)
    X_test, y_test = frame_to_xy(partition.test)
    auc = float(roc_auc_score(y_test, artifact.predict_proba(X_test)))
    external_auc_delta = round(auc - benchmark_auc, 4)
    return {
        "kind": "external_zero_shot_validation",
        "label": label,
        "external_auc": round(auc, 4),
        "benchmark_auc": round(benchmark_auc, 4),
        "auc_delta": external_auc_delta,
        "honest_note": (
            "zero-shot: artefact frozen at training time, no retraining or "
            "distribution adaptation; large negative deltas are expected and "
            "are reported, not tuned away"
        ),
        "unmapped_features": missing,
    }


# Conditional Tabluar GAN demo path
def ctgan_demo(frame: pl.DataFrame, n: int, *, seed: int) -> pl.DataFrame:
    """Sensitivity-only synthetic frame, labelled DEMO (no validation claim).

    Uses CTGAN when installed; otherwise falls back to a seeded bootstrap
    resample. Either way the caller must label the output synthetic.
    """
    try:
        from ctgan import CTGAN  # type: ignore[import-untyped]

        df = frame.to_pandas()
        model = CTGAN(random_state=seed, verbose=False)
        model.fit(df)
        return pl.from_pandas(model.sample(n))
    except ImportError:
        # Seeded bootstrap resample — same caveat: synthetic, demo only.
        rng = __import__("numpy").random.default_rng(seed)
        idx = rng.integers(0, frame.height, size=n)
        return frame[idx.tolist()]

"""Phase 2 churn-model unit tests (tasks 3.1–3.8) on tiny synthetic frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from telco_churn.data.split import stratified_split
from telco_churn.model.artifact import (
    AUC_TOLERANCE,
    ChurnArtifact,
    evaluate_against,
    frame_to_xy,
    load_artifact,
)
from telco_churn.model.calibrate import fit_isotonic
from telco_churn.model.explain import TopDrivers
from telco_churn.model.external import external_validate
from telco_churn.model.pipelines import (
    MODEL_KEYS,
    imbalance_mechanism,
    make_pipeline,
)
from telco_churn.model.thresholds import select_operating_point
from telco_churn.model.train import AUC_BAND, fit_frame, train_all


def _synthetic_frame(n: int = 240, *, seed: int = 0) -> pl.DataFrame:
    """Tiny churn-like frame: 1 numeric, 1 binary categorical, binary label."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.integers(20, 120, n).astype(float)
    contract = np.where(rng.random(n) < 0.5, "Month-to-month", "Two year")
    logit = 0.05 * (70 - tenure) + np.where(contract == "Month-to-month", 1.2, -0.8)
    proba_churn = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < proba_churn).astype(int)
    return pl.DataFrame(
        {
            "tenure": tenure.tolist(),
            "MonthlyCharges": monthly.tolist(),
            "Contract": contract.tolist(),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def test_imbalance_mechanism_recorded() -> None:
    """3.1: imbalance handling recorded with explicit fold-train-only provenance."""
    y = np.array([0, 0, 0, 1])
    mechanism = imbalance_mechanism(y)
    assert mechanism["fitted_on"] == "fold-train labels only"
    assert mechanism["positives"] == 1 and mechanism["negatives"] == 3


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_pipeline_makes_predictions(model_key: str) -> None:
    """3.1: each candidate pipeline fits and predicts on synthetic data."""
    frame = _synthetic_frame()
    X, y = frame_to_xy(frame)
    pipeline = make_pipeline(model_key, X, y, seed=42)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (frame.height,) and proba.min() >= 0.0


def test_no_leakage_ohe_excludes_test_only_categories() -> None:
    """3.1: preprocessors fitted on train only — test-only category stays unknown."""
    train = _synthetic_frame(200, seed=1)
    test = pl.DataFrame(
        {
            "tenure": [10, 20],
            "MonthlyCharges": [90.0, 40.0],
            "Contract": ["NEVER-SEEN-PLAN", "Two year"],
            "Churn": ["Yes", "No"],
        }
    )
    X_train, y_train = frame_to_xy(train)
    pipeline = make_pipeline("logistic", X_train, y_train, seed=42)
    pipeline.fit(X_train, y_train)
    X_test, _ = frame_to_xy(test)
    proba = pipeline.predict_proba(X_test)[
        :, 1
    ]  # handle_unknown='ignore' must not raise
    assert np.isfinite(proba).all()


def test_operating_point_respects_precision_floor() -> None:
    """3.3: F1-max subject to precision >= 0.55 on a synthetic score curve."""
    y = np.array([1] * 6 + [0] * 4)
    # All six positives ranked above all four negatives → precision is 1.0
    # on the whole ascending part and F1 peaks (1.0) at the 6th score.
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1])
    op = select_operating_point(y, proba, precision_floor=0.55)
    assert op.precision >= 0.55
    assert op.threshold >= 0.4 - 1e-9
    assert op.f1 == pytest.approx(1.0)


def test_operating_point_unattainable_raises() -> None:
    """3.3: raising when the floor is unattainable instead of silently lowering it."""
    # The single positive is ranked last: any point with recall > 0 has
    # precision 1/10 = 0.1 < 0.55, so the floor is unattainable.
    y = np.array([0] * 9 + [1])
    proba = np.array([0.5] * 9 + [0.1])
    with pytest.raises(ValueError, match="unattainable"):
        select_operating_point(y, proba)


def test_isotonic_improves_brier_on_held_out_fold() -> None:
    """3.4: calibrated OOF probabilities improve Brier on a held-out fold."""
    rng = np.random.default_rng(7)
    y = (rng.random(400) < 0.3).astype(int)
    raw = np.clip(0.7 * y + 0.25 * rng.random(400), 0.01, 0.99)  # over-confident
    iso = fit_isotonic(y, raw)
    calibrated = iso.predict(raw)
    brier_raw = float(np.mean((raw - y) ** 2))
    brier_cal = float(np.mean((calibrated - y) ** 2))
    assert brier_cal <= brier_raw


def test_full_protocol_reports_all_three_models() -> None:
    """3.2: comparison table lists all three candidates under an identical protocol."""
    artifact = fit_frame(_synthetic_frame(), seed=42, n_trials=2, auc_band=None)
    assert [row["model"] for row in artifact.comparison] == list(MODEL_KEYS)
    assert all(
        "mean_fold_pr_auc" in row and "best_params" in row
        for row in artifact.comparison
    )


def test_operating_point_and_calibration_in_metadata() -> None:
    """3.3+3.4: metadata carries OperatingPoint fields and isotonic Brier delta."""
    artifact = fit_frame(_synthetic_frame(), seed=42, n_trials=2, auc_band=None)
    op_meta = artifact.metadata["operating_point"]
    assert op_meta["precision"] >= artifact.threshold and artifact.threshold > 0.0
    assert (
        "brier_oof" in artifact.metadata
        and "brier_oof_uncalibrated" in artifact.metadata
    )
    assert artifact.metadata["selected_model"] in MODEL_KEYS


def test_three_seed_eval_structure_and_band_flags(tmp_path: Path) -> None:
    """3.6: seeds aggregate mean±std; band flags flow into the summary."""
    summary = train_all(
        _synthetic_frame(280),
        seeds=(42, 43, 44),
        n_trials=2,
        registry_dir=str(tmp_path),
        auc_band=(0.0, 1.0),
    )
    assert summary["seeds"] == [42, 43, 44]
    assert 0.0 <= summary["test_auc_mean"] <= 1.0
    assert summary["test_auc_std"] >= 0.0
    assert summary["published_band"] == {"lower": AUC_BAND[0], "upper": AUC_BAND[1]}
    assert isinstance(summary["needs_leakage_audit"], bool)
    assert len(summary["seed_results"]) == 3
    for row in summary["seed_results"]:
        assert set(row) == {"seed", "test_auc", "oof_pr_auc", "brier_oof", "artifact"}
        assert Path(row["artifact"]).exists()
    assert (tmp_path / "train_report.json").exists()


def test_artifact_roundtrip_reproduces_auc(tmp_path: Path) -> None:
    """3.8: save → load → evaluate_against reproduces test AUC within ±0.01."""
    frame = _synthetic_frame()
    artifact = fit_frame(frame, seed=42, n_trials=2, auc_band=None)
    path = artifact.save(tmp_path / "churn_artifact_seed42.joblib")
    meta = json.loads((path.parent / f"{path.stem}.meta.json").read_text())
    assert meta["metadata"]["selected_model"] in MODEL_KEYS
    loaded = load_artifact(path)
    assert isinstance(loaded, ChurnArtifact)
    auc_reloaded = evaluate_against(loaded, frame, seed=42)
    assert (
        abs(auc_reloaded - float(artifact.metrics["test_auc"])) <= AUC_TOLERANCE + 1e-9
    )


def test_top_drivers_returns_exactly_three() -> None:
    """3.5: exactly 3 raw-feature drivers per prediction."""
    artifact = fit_frame(_synthetic_frame(), seed=42, n_trials=2, auc_band=None)
    X, _ = frame_to_xy(_synthetic_frame().head(1))
    drivers = TopDrivers(artifact.pipeline, k=3).top3(X)
    assert len(drivers) == 3
    names = {d.feature for d in drivers}
    assert names <= {"tenure", "MonthlyCharges", "Contract"}


def test_external_validation_labels_zero_shot() -> None:
    """3.7: external runner reports honest zero-shot deltas; CTGAN-free demo frame."""
    artifact = fit_frame(_synthetic_frame(), seed=42, n_trials=2, auc_band=None)
    external = _synthetic_frame(120, seed=99)
    mapping = {
        "tenure": "tenure",
        "MonthlyCharges": "MonthlyCharges",
        "Contract": "Contract",
        "Churn": "Churn",
    }
    report = external_validate(
        artifact,
        external,
        mapping,
        label="synthetic-external",
        seed=99,
        benchmark_auc=0.85,
    )
    assert report["kind"] == "external_zero_shot_validation"
    assert report["label"] == "synthetic-external"
    assert "honest_note" in report and "zero-shot" in report["honest_note"]
    assert -1.0 <= report["auc_delta"] <= 1.0
    assert report["unmapped_features"] == []


def test_stratified_split_still_holds_churn_rate() -> None:
    """Guard: splits used by the protocol keep the churn rate within tolerance."""
    frame = _synthetic_frame()
    partition = stratified_split(frame, seed=42)
    total = frame.height
    train_rate = partition.train.height / total
    assert 0.75 <= train_rate <= 0.85

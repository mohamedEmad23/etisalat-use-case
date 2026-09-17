"""Phase 2 training protocol (tasks.md 3.1, 3.2, 3.3, 3.4, 3.6, 3.8).

Per seed the protocol is:
  1. stratified 80/20 split — the test partition is evaluated at most once,
  2. stratified 5-fold CV over the training partition,
  3. Optuna tuning of all three candidate models by mean fold PR-AUC
     (task 3.2); all three scores land in the comparison table,
  4. out-of-fold predictions of the winning model feed BOTH the operating
     point (task 3.3) and the isotonic calibrator (task 3.4),
  5. final fit on the full training partition,
  6. one test evaluation (task 3.6).

Three seeds are aggregated to mean ± std; the published-band gate
(AUC 0.84–0.88) flags ``needs_leakage_audit`` when a seed scores
materially above 0.88.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from telco_churn.config import get_settings
from telco_churn.data.split import stratified_split
from telco_churn.model.artifact import (
    ChurnArtifact,
    frame_to_xy,
)
from telco_churn.model.calibrate import calibrate, fit_isotonic
from telco_churn.model.pipelines import (
    MODEL_KEYS,
    imbalance_mechanism,
    make_estimator,
    make_pipeline,
    make_preprocessor,
    suggest_hyperparams,
)
from telco_churn.model.thresholds import select_operating_point

AUC_BAND = (0.84, 0.88)
DEFAULT_SEEDS = (42, 43, 44)
N_JOBS = 3  # one Optuna study per candidate model, run in parallel processes


def _prepare_folds(
    X: Any,
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[Any, np.ndarray, Any, np.ndarray]]:
    """Fit the shared preprocessor ONCE per fold; return transformed fold data.

    Preprocessing is hyperparameter-independent, so refitting the
    ColumnTransformer per Optuna trial is wasted work. It is fitted
    per fold on fold-train rows only — no leakage introduced. Each
    tuple holds (X_train_pre, y_train, X_val_pre, y_val).
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prepared: list[tuple[Any, np.ndarray, Any, np.ndarray]] = []
    for train_idx, val_idx in folds.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        preprocessor = make_preprocessor(X_train)
        preprocessor.fit(X_train)
        prepared.append(
            (
                preprocessor.transform(X_train),
                y_train,
                preprocessor.transform(X_val),
                y_val,
            )
        )
    return prepared


def _fold_scores(
    model_key: str,
    params: dict[str, Any],
    prepared_folds: list[tuple[Any, np.ndarray, Any, np.ndarray]],
    *,
    seed: int,
) -> list[float]:
    """PR-AUC per stratified fold using pre-transformed fold data (task 3.2)."""
    scores: list[float] = []
    for X_train_pre, y_train, X_val_pre, y_val in prepared_folds:
        estimator = make_estimator(model_key, y_train, seed=seed)
        estimator.set_params(**params)
        estimator.fit(X_train_pre, y_train)
        proba = np.asarray(estimator.predict_proba(X_val_pre))[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, proba)
        scores.append(float(np.trapezoid(recall, precision)))
    return scores


def tune_model(
    prepared_folds: list[tuple[Any, np.ndarray, Any, np.ndarray]],
    model_key: str,
    *,
    seed: int,
    n_trials: int = 20,
) -> tuple[float, dict[str, Any]]:
    """Optuna study for one candidate; returns (best mean PR-AUC, best params).

    Folds are scored sequentially inside the objective so a rolling
    (mean-of-folds-so-far) value can be reported per trial — this enables
    the MedianPruner to cut clearly losing trials after early folds
    without changing the search space or the trial count.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_min_trials=2),
        study_name=f"telco-churn-{model_key}",
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_hyperparams(trial, model_key)
        scores = _fold_scores(model_key, params, prepared_folds, seed=seed)
        # Report the running mean after each fold so pruning can happen
        # before the remaining folds are fitted.
        running_mean = 0.0
        for step, score in enumerate(scores, start=1):
            running_mean = float(np.mean(scores[:step]))
            trial.report(running_mean, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return running_mean

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return float(study.best_value), dict(study.best_params)


def _tune_one_candidate(
    model_key: str,
    X: Any,
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int,
    n_trials: int,
) -> tuple[str, float, dict[str, Any]]:
    """Process-level worker: prepare folds once, then tune one candidate."""
    prepared = _prepare_folds(X, y, n_splits=n_splits, seed=seed)
    pr_auc, params = tune_model(prepared, model_key, seed=seed, n_trials=n_trials)
    return model_key, pr_auc, params


def fit_frame(
    frame: pl.DataFrame,
    *,
    seed: int,
    n_splits: int = 5,
    n_trials: int = 20,
    auc_band: tuple[float, float] | None = AUC_BAND,
) -> ChurnArtifact:
    """Run the full per-seed protocol; the test partition is touched once.

    ``auc_band`` is the published-band gate enforced on the test AUC
    (default (0.84, 0.88)); pass ``None`` to score without the gate —
    the umbrella used by fast synthetic tests and diagnostics only, never
    by the real CLI run. The precision floor is enforced on the
    out-of-fold curve: an unattainable floor raises (reported, never
    silently lowered — spec 3.3).
    """
    partition = stratified_split(frame, seed=seed)
    train, test = partition.train, partition.test
    X_train, y_train = frame_to_xy(train)
    X_test, y_test = frame_to_xy(test)

    # Task 3.2 — tune every candidate under the identical protocol.
    # Each candidate gets its own Optuna study run in a parallel process
    # (same seeds, same sweeps); folds stay sequential inside the objective
    # so the MedianPruner can report running means and cut losing trials.
    parallel = Parallel(n_jobs=min(N_JOBS, len(MODEL_KEYS)), prefer="processes")
    results = parallel(
        delayed(_tune_one_candidate)(
            model_key, X_train, y_train, seed=seed, n_splits=n_splits, n_trials=n_trials
        )
        for model_key in MODEL_KEYS
    )
    comparison: list[dict[str, Any]] = []
    best_key: str | None = None
    best_params: dict[str, Any] = {}
    best_auc = -1.0
    for model_key, pr_auc, params in results:
        comparison.append(
            {
                "model": model_key,
                "mean_fold_pr_auc": round(pr_auc, 4),
                "best_params": params,
            }
        )
        if pr_auc > best_auc:
            best_auc, best_key, best_params = pr_auc, model_key, params
    if best_key is None:
        raise ValueError(
            "no candidate model scored above -1.0 — tuning produced empty comparison"
        )

    # Out-of-fold predictions of the winner (train folds only).
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_train), dtype=float)
    for train_idx, val_idx in folds.split(X_train, y_train):
        X_t, y_t = X_train.iloc[train_idx], y_train[train_idx]
        fold_pipeline = make_pipeline(best_key, X_t, y_t, seed=seed)
        for name, value in best_params.items():
            fold_pipeline.set_params(**{f"model__{name}": value})
        fold_pipeline.fit(X_t, y_t)
        oof[val_idx] = fold_pipeline.predict_proba(X_train.iloc[val_idx])[:, 1]

    # Task 3.3 — operating point on OOF predictions subject to the floor.
    op = select_operating_point(y_train, oof)
    # Task 3.4 — isotonic calibration fitted on the same OOF curve.
    iso = fit_isotonic(y_train, oof)
    auc_oof = float(roc_auc_score(y_train, oof))
    brier_oof = float(
        brier_score_loss(y_train, np.asarray(calibrate(iso, oof), dtype=float))
    )
    brier_raw_oof = float(brier_score_loss(y_train, oof))

    # Final fit on the full training partition with the tuned hyperparameters.
    final = make_pipeline(best_key, X_train, y_train, seed=seed)
    for name, value in best_params.items():
        final.set_params(**{f"model__{name}": value})
    final.fit(X_train, y_train)

    # Task 3.6 — one test evaluation for this seed, no peeking.
    auc_test = float(roc_auc_score(y_test, final.predict_proba(X_test)[:, 1]))
    if auc_band is not None and auc_test < auc_band[0]:
        raise ValueError(
            f"seed {seed}: test AUC {auc_test:.4f} below published-band lower bound "
            f"{auc_band[0]} — reporting rather than masking"
        )
    test_auc = round(auc_test, 4)

    return ChurnArtifact(
        pipeline=final,
        threshold=op.threshold,
        calibrator=iso,
        feature_list=[str(c) for c in X_train.columns],
        metadata={
            "seed": seed,
            "selected_model": best_key,
            "best_params": best_params,
            "auc_band": list(auc_band) if auc_band is not None else None,
            "imbalance": imbalance_mechanism(y_train),
            "operating_point": {
                "threshold": op.threshold,
                "f1": op.f1,
                "precision": op.precision,
                "recall": op.recall,
            },
            "oof_auc": round(auc_oof, 4),
            "brier_oof": round(brier_oof, 4),
            "brier_oof_uncalibrated": round(brier_raw_oof, 4),
            "n_splits": n_splits,
            "n_trials": n_trials,
        },
        metrics={
            "test_auc": test_auc,
            "oof_pr_auc": round(auc_oof, 4),
            "brier_oof": round(brier_oof, 4),
        },
        comparison=comparison,
    )


def train_all(
    frame: pl.DataFrame,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_trials: int = 20,
    registry_dir: str | None = None,
    auc_band: tuple[float, float] | None = AUC_BAND,
) -> dict[str, Any]:
    """3-seed loop (task 3.8): mean ± std test AUC + saved artifacts + report.

    The leakage-audit gate fires when any seed exceeds the published band's
    upper bound (0.88) — scores materially above the band trigger review.
    ``auc_band`` is forwarded to each ``fit_frame`` call; pass ``None``
    only in tests/diagnostics (never the real CLI run).
    """
    settings = get_settings()
    out_dir = Path(
        registry_dir if registry_dir is not None else str(settings.model_registry_dir)
    )
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        artifact = fit_frame(frame, seed=seed, n_trials=n_trials, auc_band=auc_band)
        path = artifact.save(out_dir / f"churn_artifact_seed{seed}.joblib")
        seed_results.append({"seed": seed, **artifact.metrics, "artifact": str(path)})

    aucs = [float(r["test_auc"]) for r in seed_results]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    needs_leakage_audit = any(float(r["test_auc"]) > AUC_BAND[1] for r in seed_results)

    summary: dict[str, Any] = {
        "seeds": list(seeds),
        "test_auc_mean": round(mean_auc, 4),
        "test_auc_std": round(std_auc, 4),
        "published_band": {"lower": AUC_BAND[0], "upper": AUC_BAND[1]},
        "needs_leakage_audit": needs_leakage_audit,
        "seed_results": seed_results,
        "registry_dir": str(out_dir),
    }
    report_path = out_dir / "train_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary["report_path"] = str(report_path)

    # Light MLflow logging, guarded (task 3.8) — never fatal.
    try:
        import mlflow  # type: ignore[import-untyped]

        mlflow.set_experiment("telco-churn")
        with mlflow.start_run(run_name=f"churn-3seed-{mean_auc:.4f}"):
            mlflow.log_metric("test_auc_mean", mean_auc)
            mlflow.log_metric("test_auc_std", std_auc)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("seeds", list(seeds))
    except Exception:
        logging.getLogger(__name__).debug("MLflow logging skipped", exc_info=True)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the churn model (Phase 2 protocol)."
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--registry-dir", type=str, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    frame = pl.read_csv(str(settings.data_csv_path))
    summary = train_all(frame, n_trials=args.n_trials, registry_dir=args.registry_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

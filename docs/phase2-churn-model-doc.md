# Phase 2 — Churn Model: Documentation, Results, and Design Rationale

Branch: `p2/churn-model` — tasks `3.1–3.8` of `openspec/changes/add-telco-churn-poc/tasks.md` (§3, all complete).

## 1. Aim / Goal

Build the churn classifier behind the whole PoC: three candidate pipelines, an Optuna-tuned selection by mean 5-fold CV PR-AUC, an operating point with a precision floor, isotonic calibration, per-prediction SHAP top-3 drivers, a 3-seed evaluation honest to the published AUC band (0.84–0.88), external validation hooks, and a reproducible-on-reload artifact. Non-negotiables carried over from Phase 1 / design.md:

- The classifier computes Churn probability — the LLM never does.
- Imbalance handling (class weights / scale-pos-weight) is fitted on fold-train labels only, inside the sklearn Pipeline.
- The test set is touched at most once per seed; it is never consumed during tuning.
- Above-band AUC (> 0.88) triggers a mandatory leakage audit; below-band (< 0.84) fails loudly rather than being masked.

## 2. Code written (`src/telco_churn/model/`)

| File | Purpose | Why this shape |
| --- | --- | --- |
| `pipelines.py` | `MODEL_KEYS`, `feature_columns` (numeric/categorical split by dtype), `make_preprocessor`, `make_estimator`, `make_pipeline`, `imbalance_mechanism`, `suggest_hyperparams` | One shared `ColumnTransformer` (median imputer + scaler / OHE `handle_unknown='ignore'`); per-model estimators with imbalance built in; `scale_pos_weight` / class weights derive from fold-train labels only and the mechanism is recorded in metadata (leakage test proves it) |
| `thresholds.py` | `OperatingPoint`, `select_operating_point` with `PRECISION_FLOOR=0.55` | Max F1 on OOF predictions subject to precision ≥ 0.55; raises `ValueError` ("unattainable … reporting rather than silently lowering the floor") if no threshold satisfies it |
| `calibrate.py` | `fit_isotonic`, `calibrate` | Isotonic regression fitted on train-fold OOF predictions, so chat-time probability is calibrated without touching test |
| `explain.py` | `Driver`, `TopDrivers` | SHAP (`TreeExplainer` for XGB/LGBM, `LinearExplainer` for logistic), aggregated from transformed columns back to raw columns, exactly top-3 per prediction |
| `artifact.py` | `ChurnArtifact` dataclass, `save` / `load_artifact`, `evaluate_against`, `frame_to_xy` | Reproducibility: persists pipeline + threshold + calibrator + feature list + metadata; reloading and re-scoring the seeded split must reproduce the AUC within ±0.01 |
| `train.py` | `fit_frame`, `train_all`, `_prepare_folds`, `_fold_scores`, `tune_model`, `_tune_one_candidate`, CLI `main` | The whole per-seed protocol (split → tune 3 models → OOF of winner → operating point → isotonic → final fit → one test eval → band gate); 3-seed aggregation, `runs/` artifacts, `train_report.json`, guarded light MLflow logging |
| `external.py` | `external_validate`, `ctgan_demo`, `SOURCE_LABELS` | Zero-shot UCI-Iranian-Churn / Orange-Telecom validation with a minimal schema mapping and honest AUC deltas; CTGAN path is demo-only (`ImportError` → seeded bootstrap), explicitly NOT a validation claim |

Tests: `tests/unit/test_churn_model.py` — 12 tests on tiny synthetic frames (n≈240): imbalance recorded, per-model fit/proba, no-leakage on unseen categories, threshold floor + unattainable path, isotonic Brier improvement, all-three-model comparison table, 3-seed structure, artifact save/load ±0.01, exactly-3 drivers, external zero-shot payload, split rate.

Processing-speed work (later commits): `make_estimator` split out of `make_pipeline` so tuning uses raw estimators; `_prepare_folds` pre-computes fold matrices; per-model Optuna studies run in parallel processes via joblib.

## 3. Results (real CSV run, user-executed)

Command: `PYTHONPATH=src .venv/bin/python -m telco_churn.model.train --n-trials 20`

| Seed | test_auc | oof_pr_auc | brier_oof |
| --- | --- | --- | --- |
| 42 | 0.8437 | 0.8476 | 0.1330 |
| 43 | 0.8487 | 0.8463 | 0.1336 |
| 44 | 0.8437 | 0.8473 | 0.1324 |

- `test_auc_mean = 0.8454`, `test_auc_std = 0.0024`
- `needs_leakage_audit = false`; every seed inside the published band 0.84–0.88
- Artifacts: `runs/churn_artifact_seed{42,43,44}.joblib`, `runs/train_report.json`
- Reading: stable across seeds (tiny std), calibration improved Brier on OOF, and the score sits honestly in band — no overfitting signature, so no leakage audit required.

## 4. Runtime efficiency work

### The 6 candidate optimizations considered

1. **Pre-fit the preprocessor once per fold** — OHE/scaler/imputer do not depend on model hyperparameters; refitting them inside every one of ~300 trial×fold fits is pure waste.
2. **Optuna pruners (MedianPruner)** — report a running mean fold score per step; trials clearly losing after the first folds stop early without reducing the number of trials or sweep ranges.
3. **Parallelize per-model Optuna studies** — one study per model key, run in parallel processes with joblib; folds stay sequential inside each objective (required for pruning to see ordered means).
4. **Native GBM early stopping** — LightGBM/XGBoost validation-fold early stopping to end bad configs early, not fewer epochs allowed.
5. **Parallelize Optuna trials themselves** (`study.optimize(n_jobs=)`) — rejected: parallel trials break reproducibility of the seeded TPESampler.
6. **Reduce trials / folds / sweep ranges** — rejected by constraint: the user explicitly asked for speed without touching sweeps, epochs, or folds; also memory was verified a non-issue (~7043 rows, 32 GB slot).

### Chosen and implemented: #1 + #2 + #3

- `_prepare_folds(X, y, *, n_splits, seed)`: StratifiedKFold(5, shuffle, seed); per fold, fit the preprocessor once and transform train+val once.
- `_fold_scores`: fits the bare estimator (`make_estimator`) on pre-transformed fold-matrices and scores PR-AUC (`precision_recall_curve` + `np.trapezoid`).
- `tune_model`: MedianPruner (`n_startup_trials=4`), running-mean reporting per fold step; losing trials are pruned.
- `_tune_one_candidate` + `fit_frame`: the three model studies run in parallel (`Parallel(n_jobs=min(3, len(MODEL_KEYS)), prefer="processes")`, `N_JOBS=3`), full fold coverage retained.

Consequence to record: pruning can change which trials finish, so best params may differ slightly from a fully un-pruned run — that is expected and documented here, and #4 (early stopping) was deliberately NOT taken because it would shift best_params semantics beyond pruning alone.

## 5. Cross-phase agreements honored

- Published-band honesty (AUC 0.84–0.88; above → leakage audit): verified live, `needs_leakage_audit` false.
- Test set consumed exactly once per seed (single `roc_auc_score` after final fit).
- Imbalance mechanism recorded in artifact metadata with `fitted_on: fold-train labels only`.
- Reproducible artifact per task 3.8; reload + `evaluate_against` reproduces within `AUC_TOLERANCE = 0.01`.

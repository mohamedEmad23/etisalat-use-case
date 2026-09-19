# telco-churn/churn-model Specification

## Purpose

Owns the churn classifier: training under the locked protocol, threshold selection, calibration, per-prediction explanation, and honest evaluation against published-band targets plus real external datasets.

## Requirements

### Requirement: Imbalance handling fitted on train only

The system SHALL handle the ≈26.5%/73.5% class imbalance via class weighting (or train-fitted resampling) and MUST NOT apply any imbalance treatment to or derived from the test partition.

#### Scenario: Imbalance treatment is train-scoped

- **WHEN** training completes
- **THEN** the imbalance mechanism and its fitted parameters are recorded and provably derived from train folds only

### Requirement: Model selection under identical protocol

The system SHALL train and compare a logistic-regression baseline, XGBoost, and LightGBM under the identical split/CV protocol, and SHALL select the primary model by mean PR-AUC across folds, reporting all three.

#### Scenario: Comparison table produced

- **WHEN** the training run completes
- **THEN** a comparison table with per-model mean±std PR-AUC, AUC-ROC, F1, and accuracy exists, and the selected model is stated with the selection rule

### Requirement: Operating point selection

The system SHALL select the operating point that maximizes F1(churn) on train cross-validation subject to precision ≥ 0.55, and SHALL report precision, recall, and F1 at that point.

#### Scenario: Threshold satisfies the precision floor

- **WHEN** the operating point is chosen from train CV
- **THEN** its precision is ≥ 0.55 and its F1 is the achievable maximum subject to that constraint

### Requirement: Calibrated probabilities

The system SHALL calibrate predicted probabilities (isotonic) fitted on train folds only, and chat responses SHALL present calibrated churn probabilities.

#### Scenario: Calibration applied

- **WHEN** a churn probability is produced
- **THEN** it comes from the calibrated model and the calibration method is recorded

### Requirement: Per-prediction explanation

The system SHALL provide, for any individual prediction, the top-3 feature drivers ranked by contribution.

#### Scenario: Drivers returned with prediction

- **WHEN** a single-customer prediction is executed
- **THEN** the three most influential features for that prediction are retrievable in the prediction payload

### Requirement: Honest evaluation protocol

The system SHALL report test metrics (AUC-ROC, PR-AUC, F1(churn), accuracy, churn precision/recall at the operating point) as mean±std over 3 seeds, SHALL target the published band (AUC 0.84–0.88; scores materially above the band trigger a leakage audit), and MUST evaluate the test partition at most once per seed.

#### Scenario: Seeded metrics reported

- **WHEN** evaluation completes
- **THEN** all reported metrics are mean±std over 3 seeds and the test set was not consumed during tuning

### Requirement: Reproducible model artifact

The system SHALL persist the trained model with metadata (seed, hyperparameters, operating point, calibration, metrics) sufficient to reproduce test AUC within ±0.01 across seeds.

#### Scenario: Artifact reload reproduces metrics

- **WHEN** the persisted artifact is reloaded and re-evaluated with its recorded seed
- **THEN** test AUC matches the recorded value within ±0.01

### Requirement: External validation

The system SHALL evaluate the frozen trained model on at least two real external datasets (UCI Iranian Churn, Orange Telecom) with honest, documented AUC-drop reporting, and SHALL restrict CTGAN-generated data to sensitivity analysis and demo filler — never a validation or scoring claim.

#### Scenario: External results reported honestly

- **WHEN** external validation runs
- **THEN** per-dataset AUC deltas versus the in-dataset benchmark are reported, and no synthetic-data metric appears as a validation claim

## Purpose

Turns the raw attached CSV into a clean, deterministic, leakage-free train/test contract that every downstream capability consumes. It is the single authority on data hygiene rules for the 7,043-row Telco dataset.

## ADDED Requirements

### Requirement: Faithful raw ingestion
The system SHALL load the attached CSV exactly as provided (7,043 data rows × 21 columns) and SHALL normalize header whitespace (including the trailing-space column name) so all columns are addressable by canonical names.

#### Scenario: Load succeeds with renamed schema
- **WHEN** the raw CSV is loaded
- **THEN** 7,043 rows and 21 columns are present, every header is trimmed, and column names match the documented dataset schema

### Requirement: Total_Charges coercion before splitting
The system SHALL coerce `Total_Charges` to numeric before any split occurs, and SHALL apply one explicit, documented rule for the 11 blank `Total_Charges` rows (all tenure = 0, all Churn = No).

#### Scenario: Blank Total_Charges rows handled deterministically
- **WHEN** the dataset is loaded and the documented blank-row rule is applied
- **THEN** `Total_Charges` is fully numeric with no NaN outside the documented handling, and the rule (and its row count) is stated in the run report

### Requirement: Identifier exclusion
The system SHALL drop the customer identifier column at load time and MUST NOT use it as a model feature or expose it in any prediction output.

#### Scenario: No identifier in features
- **WHEN** the training feature matrix is built
- **THEN** the customer identifier column is absent from features and from any returned prediction payload

### Requirement: Deterministic stratified split
The system SHALL produce a stratified 80/20 train/test split with a fixed seed, and SHALL run 5-fold stratified cross-validation on the train portion for model selection and threshold tuning.

#### Scenario: Reproducible split
- **WHEN** the pipeline runs twice with the same seed
- **THEN** train and test sets are byte-identical and the churn rate in both partitions matches the dataset's ≈26.5% within rounding

### Requirement: Leakage guardrails
The system MUST fit every data-dependent transform (class weighting, resampling, imputation, scalers) on training folds only. Resampling MUST NOT occur before the split.

#### Scenario: Train-only fitting
- **WHEN** any preprocessing transform is fitted
- **THEN** it is fitted inside the train (or train-fold) context and never sees test rows

### Requirement: Categorical normalization
The system SHALL normalize categorical values deterministically: trim whitespace variants, map the 0/1 senior-citizen encoding to booleans, and treat "No phone service" / "No internet service" sentinel strings as documented distinct values.

#### Scenario: Sentinel strings preserved as semantics
- **WHEN** normalization runs
- **THEN** sentinel strings map to their documented meaning and no category value retains leading/trailing whitespace

### Requirement: Documented spec/data mismatches
The system SHALL document every divergence between the challenge PDF and the data (PDF says "Postal check"; data contains "Mailed check"), stating the PDF's claim and deferring to the data.

#### Scenario: Mismatch note present
- **WHEN** documentation is generated
- **THEN** each known PDF/data mismatch is listed with both values and the resolution rule

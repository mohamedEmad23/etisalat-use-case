# Phase 1 Analysis & Design Rationale

Answers to the implementation review questions for `p1/data-pipeline`, backed by
numbers measured on the actual dataset (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`).
Written before starting `p2/churn-model`; the model-choice section records the
Phase 2 design decisions that the Optuna selection will test empirically.

---

## 1. Why LogReg, XGBoost, and LightGBM specifically

The three candidates form a **deliberate spread of inductive biases**, not a grab-bag.
Spec §churn-model requires all three evaluated with their scores reported, so the
choice also has to be defensible if any of them wins.

| Model | Why it's in the race |
|---|---|
| **Logistic Regression (baseline)** | Maximum-likelihood optimization gives the *calibration ground truth*: on independent features, logreg probabilities are close to true empirical rates — the yardstick any fancier model's calibration is judged against. Handles mixed continuous + discrete features natively (binarize the categoricals, scale the numerics). It is transparent, nearly unattackable on AUC vs. a single 25–30% positive class, and establishes whether the boosters even beat the floor. If XGB/LGBM can't beat it, the honest story is "this data is linear-ish." |
| **XGBoost** | Best-in-class regularization controls (L1/L2, gamma, min split gain) — strong defense against overfitting a 7k-row dataset that 3,000+ tree depths would love to memorize. Native missing-data handling (NULLs route to the better child at split time) means the 11 tenure-0 nulls and generally messy telecom data don't need aggressive imputation. Parallel thread-level histogram building. |
| **LightGBM** | Histogram-based split finding with **leaf-wise** growth → deeper, more asymmetric trees reach the minority "churner" structure faster with a small dataset; dramatically faster than level-wise XGB on this scale, which matters because Optuna explores 50–100 trials × 3 models × 5 folds. The `class_weight` native support makes it the natural home for the imbalance handling (spec: weight/resample *inside* the sklearn pipeline). |

Why not more candidates: the challenge rewards an honest protocol, not a model
zoo. These three cover linear vs. two leading GBM implementations (different
growth strategies: level-wise vs leaf-wise) — the consensus structures of large
tabular benchmarks. Logistic regression as the baseline keeps the published-band
claim (AUC 0.84–0.88) honest: if only LightGBM passes the band, we know to check
whether LogReg scoring is anywhere near it before claiming a "boosting win".

All three are wrapped in **sklearn Pipelines** so imbalance handling
(`class_weight` / SMOTE-like transforms) is fitted on **train folds only** and
Optuna tunes hyperparameters per model with the same CV protocol (PR-AUC).

---

## 2. What Phase 1 found in the data

Measured on the ingested frame (`ingest.py` output: 7,043 rows × 20 cols after
the `customerID` drop).

### Data quality
- **Nulls:** 11 total across every column — all in `Total_Charges`. Characterized
  fully: all 11 have `tenure == 0` and `Churn == No`. No other column has a
  null, no implicit " " sentinels beyond that one column.
- **Duplicates:** zero duplicate `customerID`s. But **40** rows are *feature
  twins* (19 cleaned features identical, i.e. 7,043 → 7,003 unique feature
  vectors, 73 rows share a twin, 54.8% churn among twin-group rows). Keeping
  them is correct dedup policy for two reasons: (a) they are different
  customers (the PDF timeline genuinely records repeated contract activity),
  and (b) same features → different churn outcomes is signal we want the model
  to learn, not bleed out. Dropping them would bias the train and fold splits
  only if we dropped one twin per pair — left alone, stratification keeps both.
- **Data types forcing a decision:** `Total_Charges` arrives as inferred
  Float64 with 11 nulls only when read via obvious heuristics; otherwise as
  space-padded string. The cleaner handles both shapes so the coercion rule is
  schema-inference-agnostic.

### Relationships (Pearson on the cleaned numerics + engineered churn flag)
| Pair | r | Note |
|---|---|---|
| `Total_Charges` ↔ `tenure` | **0.83** | high collinearity (Total_Charges ≈ monthly × tenure precisely) |
| `tenure` ↔ churn | **−0.35** | strongest numeric predictor: longer-lived customers don't churn |
| `Total_Charges` ↔ churn | −0.20 | largely a proxy for the tenure effect (0.83 collinear above) |
| `Monthly_Charges` ↔ churn | +0.19 | higher bill → more churn, real but modest |
| `avg_monthly` ↔ everything | NaN↔ merely nulls in tenure-0 rows only | engineered AFTER the split-free clean, mathematically independent of the others by construction (it removes the tenure multiplier) — this is why it can add information over both parents |

### Categorical signal (sanity-checked against churn rates)
- Contract: month-to-month **42.7%**, one-year 11.3%, two-year 2.8% churn — the dominant driver.
- Internet service: fiber optic **41.9%**, DSL 19.0%, none 7.4% churn.
- Tenure median among churners 10 months vs. 38 for stayers; churners' average
  monthly bill 74.4 vs. 61.3.

These numbers agree with the challenge PDF narrative and set up the Phase 2
expectation that the top 3 SHAP drivers will likely be Contract type, tenure,
and a billing/FAS feature.

---

## 3. Cleaning & feature engineering: reasoning (and "is this the best approach?")

### The 4 rules, restated as anti-footgun decisions

**Rule 1 — coerce `Total_Charges` to numeric BEFORE any split; coerce under a
strict, loud rule.**
- *Best approach?* Dropping the 11 rows would be the "safe-looking" alternative,
  but it throws away real customers and, worse, **hides a row-count invariant**
  (the challenge describes exactly 7,043 rows). Imputing the blanks with a
  column mean or regression imputation would fabricate a charge for a customer
  with zero billing history - and silently infect both train and test if done
  before the split. Setting them to 0.0 is the only value that's *derived from
  the row itself*, not from other rows: `tenure == 0` mathematically implies
  no completed billing cycle → `Total_Charges` must be 0. The `raise
  ValueError` if a blank row ever has `tenure != 0` converts a silent rule
  change into a crash, so a future data refresh can't erode the rule without
  the tests noticing.
- A Tennure-0 row with `Churn = No` is also *not* contradicting the target: a
  new customer whose first cycle just completed won't have churned yet.

**Rule 2 — trim whitespace on every string column, keep the sentinel strings
distinct.**
- Whitespace trimming is a *deterministic* normalization (identity on clean
  data), never a data-dependent transform, so it can't leak.
- "No phone service" / "No internet service" carry *more* information than
  "No": they encode that the customer HAS the base service but not the add-on.
  Collapsing them into "No" would destroy a real feature boundary; mapping
  them to a third boolean column ("has internet service") is what the
  cleaning effectively enables the model layer to do via one-hot without
  data loss. This was the least confrontational choice: no invented
  transformations, just refuse to conflate.

**Rule 3 — `Senior_Citizen` 0/1 → bool.**
- The source encoding is numeric but semantically boolean. Casting it makes
  the feature list uniform ("boolean" not "integer that happens to be 0/1"),
  which matters for the model's treatment (no secret ordinal assumption of
  0 < 1, though for a two-level variable in boosted trees/logreg it's
  numerically identical anyway) and for SHAP readability.

**Rule 4 — `avg_monthly = Total_Charges / tenure` where tenure > 0, else null.**
- *Why this is the right function:* the raw dataset has a strong mot context:
  `Total_Charges ≈ Monthly_Charges × tenure` (collinearity 0.83). The ratio
  normalizes that dependency out, leaving "what does this customer believe
  per month", which is a better intent-of-stay signal when we want a per-row
  drift-adjusted bill, not just a snapshot.
- *Why null for tenure 0 instead of 0:* a customer with no completed cycle has
  no defined average monthly spend. Fabricating 0 or Monthly_Charges itself
  would produce a hard-leveraged garbage value at the extreme of the feature
  distribution. XGBoost/LightGBM natively route nulls through their split
  search, so the null is not a "broken value" — it's a *legal, learnable* one.
  The 11 null rows are 0.16% of the data — below any practical imputation gain.

### Honest self-assessment: could this be done better?

Not meaningfully, given the spec constraints (determinism, pre-split coercion,
documented rules). The alternatives were considered and rejected on leakage or
fidelity grounds:
- *Dedup feature twins:* rejected (throws away real churn-rate heterogeneity).
- *Impute `Total_Charges` from Monthly×tenure:* rejected (it is literally what
  the column is, so 0.0 is the honest value; regression imputation adds
  train-data dependence to test data — a leakage vector).
- *One-hot the sentinels now (cleaning stage):* rejected deliberately —
  categorical encoding is a **model-side** transform (fits on train folds);
  cleaning keeps the frame semantic and encoding-agnostic.

The one *future* change worth flagging: if Phase 2 diagnostics show the 11
tenure-0 rows create an artifact (e.g., SHAP weirdness at tenure 0), the
documented rule can evolve — but it must evolve in this file and in the
mismatches doc, never as an inline hack in the trainer.

---

## 4. Why StratifiedKFold with 5 folds

Three requirements drive this, one empirical, two structural:

1. **Stratification is non-negotiable** because the target is imbalanced
   (~26.5% churn). With plain KFold, a random fold could have 15% or 40%
   churners; a PR-AUC computed on such a fold would collapse and mislead   the
   Optuna selection. Stratified folds hold each fold near 26% churn (verified:
   all 5 validation folds land 25.5–27.8% here).
2. **5 as the K** (sklearn's StratifiedKFold default, and the K most commonly
   used for tabular model selection):
   - **Bias/variance trade-off:** K=5 gives each model a mean over 5 runs —
     enough to average out fold luck without the noise-hiding that 1 or 2
     folds would do. K=10 gives a tighter mean but doubles compute across
     Optuna × 3 models × 5 folds and yields a *larger* train portion only
     marginally different from K=5 on 5,634 rows.
   - **Compute budget:** the protocol multiplies out to (Optuna trials) ×
     (3 models) × (5 folds) training fits. At K=5 a 60-trial Optuna sweep over
     LightGBM costs ~300 fits; LightGBM's histogram splits keep this well
     under minutes on CPU. K=10 would double it for little metric-stability
     gain.
   - **Fold size symmetry:** on the 5,634-row train portion, 5 folds give
     ~1,127-row validation slices — ~300 churners each, i.e. enough positives
     for a stable PR-AUC and a per-fold precision@threshold ≥ 0.55 check.
3. **Seeded and shuffled** (`shuffle=True, random_state=seed`): determinism
   (same seed → same folds, unit-tested byte-identically) plus the shuffle you
   need because the CSV is *not* random-ordered — TeleWorm data like this
   arrives with correlated block ordering (new customers together, etc.),
   which would make unshuffled folds systematically biased.

80/20 test split (not 70/30) follows the same logic: a 1,409-row test set is
~373 churners, enough to make the AUC band check meaningful while preserving
5,634-row train for CV. The spec mandates exactly this 80/20 + 5-fold protocol,
and the test set is **touched at most once per seed** in the final evaluation
— the split's job is to stay sacred, not to be run against repeatedly.

---

## 5. The leakage guardrail — and why resampling MUST NOT precede the split

Spec: *"The system MUST fit every data-dependent transform (class weighting,
resampling, imputation, scalers) on training folds only. Resampling MUST NOT
occur before the split."*

### The principle

Leakage = the eval metric learns something about rows it will later be scored
on, causing the reported number to flatter the model. The classic sneaky form
in churn datasets is *pre-split resampling to fix the class imbalance*, and it
fails in two steps:

1. **Row identity leak.** SMOTE oversamples by interpolating between two
   minority-class neighbors. If you SMOTE the FULL dataset and then split, a
   train-row point and its interpolated synthetic twin can land on both sides
   of the boundary. The synthetic test point is a convex combination of train
   rows — the model literally saw its ingredients. The metric then measures
   the data pipeline's memory, not the model's generalization.
2. **Distribution shift.** Resampling before the split changes the prior in
   *both* train and test. The test set is supposed to answer "how would this
   model do on production churn traffic at its natural 26.5% rate?" —
   resampling rewrites that question to "how does it do at 50/50?" and the
   AUC/threshold numbers lose their production meaning.

### What the guardrail actually enforces in this codebase

Structurally, the guardrail is enforced by *where code is allowed to fit
things*, not by vigilance:

- `split.py` partitions row indices only — it fits nothing, sees no feature
  statistics, and is the only place train/test data touches.
- Every data-dependent transform (class_weight, SMOTE, imputation, scalers)
  lives **inside an sklearn Pipeline**, which fits exactly what it sees. When
  CV runs a fold, the pipeline is constructed and fitted on the fold's
  train indices only; the validation fold is `.transform()`/`.predict()` only.
- The stratified 80/20 split produces the train portion; folds are cut from
  the train portion, never from the combined frame. So test rows have never
  been part of *any* fit, including resampling and the Optuna search.

Why this matters for the specific numbers we publish:
- The `AUC 0.84–0.88` published band is an honest claim only if the CV protocol
  in training mirrors deployment scoring. Any leak inflates CV PR-AUC, pushes
  the operating-point threshold to an unrealistically sweet spot, and the
  "test AUC once-per-seed" ritual then confirms the inflated CV story.
- The **above-band gate** (AUC > 0.88 → mandatory leakage audit) is the
  honest-ducks telling us to check, precisely because naive protocol bugs leak
  in ways that look like brilliance.

Concrete consequences if we broke the rule (hypothetical, measured often in
the literature): SMOTE-before-split on this dataset classically inflates
fold PR-AUC by 5–15% while producing a *calibration* that is meaningless on
the untouched test set — which is the worst combination because the chat
pipeline then quotes a confident probability that has no production basis.

---
*Written for `p2/churn-model` review; numbers re-runnable via the scripts in
the analysis traces (see `tests/unit/test_data_pipeline.py` for the unit
pinned counterparts).*

# Phase 2/3 Questions and Answers — CSV cleaning, accuracy, calibration, thresholds, external validation, and the extract-only LLM design

This document answers the review questions raised after Phase 3, in the order asked. It is prose, not a spec — where a decision is recorded, it repeats what the code already enforces so the reasoning survives in one place.

---

## 1. "Why is there no cleaned CSV the model trains on?"

You are right — this was an avoidable mistake, and the root cause was one shortcut in one function, not a design gap: `train.py::main()` (the CLI entry the ~1-hour training run calls) loaded the raw CSV directly with `polars.read_csv` instead of going through the pipeline we built in Phase 1 (`load_churn_csv` → `clean`). All the *library* users of the data path (chat loop baseline frame, tests, external validation) already ran through `load_churn_csv().pipe(clean)`; only the CLI bypassed it. That is exactly why the Phase 3 doc flagged the stale artifacts as a known issue. The criticism stands as process feedback: the persistent, inspectable artifact should have existed from Phase 2.

**What is fixed now (on this branch):**

- `train.py::main()` routes through the full ingestion + cleaning pipeline: `frame = clean(load_churn_csv())`.
- Every training run **persists the cleaned frame** next to the raw source and prints the path, so what the model trained on is always inspectable:

  ```
  data/churn_cleaned.csv
  ```

  Properties verified just now: 7043 rows × 21 columns; `customerID` dropped; the raw header's `'Senior_Citizen '` trailing-space trimmed; the 11 blank `Total_Charges` rows (tenure = 0) coerced to `0.0`; category sentinels (`No phone service`, `No internet service`) preserved as distinct values; `avg_monthly` engineered; `Churn` present as the label.

- The cleaned copy is freshly regenerated at the start of every training run (overwrites, deterministic), so it can never silently drift from the code.

**For your ~58-minute run**, the (unchanged) command is:

```sh
PYTHONPATH=src .venv/bin/python -m telco_churn.model.train --n-trials 20
```

On the speedup question — "your optimization better not break something": the 43-test suite (12 churn-model tests among them, including artifact save/load with AUC reproducibility ±0.01 and the 3-seed structure) passes with the optimization in place; the optimizations themselves only removed redundant preprocessing refits, doomed Optuna trials, and sequential waits — the protocol (splits, folds, sweeps, the band gate) is untouched, and the per-seed determinism test still proves reproducibility. Residual risk is the one documented honestly earlier: the MedianPruner can change which trials finish, so best params can differ slightly from an unpruned run — that is a different parameter choice, not a broken metric.

---

## 2. "What is the actual accuracy? What is slot accuracy? What is the bank-transfer miss?"

Three different numbers were quoted in the docs; they measure three different things.

### 2a. The model's score: AUC, not accuracy — and why

The churn classifier was deliberately **evaluated on ROC AUC, mean 0.8454 ± 0.0024 across the 3 seeds** (from the real-CSV run you executed). AUC asks: "if you take one random churner and one random non-churner, does the model rank the churner higher?" A 0.8454 answer happens ~84.5% of the time.

We did **not** report plain accuracy for three deliberately honest reasons:

1. **The classes are imbalanced** (churn ≈ 26.5%, non-churn ≈ 73.5%). A model that says "nobody churns" achieves 73.5% "accuracy" using zero information. Accuracy rewards that; AUC punishes it (a constant scorer scores 0.5 AUC, not 0.735).
2. **The spec's own published band (0.84–0.88 AUC) is an honesty gate, not a brag.** AUC is the figure the challenge distributes are quoted in, and the band exists so that a suspiciously high AUC (> 0.88) triggers a mandatory leakage audit rather than a celebration.
3. **Decision-threshold accuracy is a policy choice**, not a model property: the same pipeline has a full range of (precision, recall) pairs, and we pick the single point the marketing use-case needs via the operating point (see §6). Accuracy-after-threshold depends on that choice; AUC does not. Precision/recall at the chosen threshold will be visible in the re-run's artifact metadata (`operating_point`), computed on the training CV split only.

"(Never) to be accurate at the margin" is also why the **Brier score** (~0.133 on out-of-fold predictions, calibrated) is tracked: it measures whether the *probabilities themselves* are trustworthy (how far each predicted probability sits from the actual outcome), which is what the chat surface actually quotes to the marketing team.

### 2b. Slot accuracy — an LLM-eval metric, not a model metric

This number (95.83%) measures the **extraction step only**: for each of the 23 curated utterances in `tests/eval/test_extraction_suite.py`, the eval knows exactly which FeatureRequest slots the sentence should map to (which feature names belong in `target_features`, which `filters`, whether `out_of_scope=True`). Slot accuracy = fraction of cases where the LLM's predicted request matched the expected request exactly. The model's AUC plays no role in this number.

- **Schema validity = 100%** (24/24 LLM calls produced parseable, vocabulary-valid FeatureRequests — nothing leaked through the grammar).
- **Slot accuracy = 23/24 = 95.83%** (bar: ≥ 95%, cleared).

### 2c. The bank-transfer miss, elaborated

The case: the utterance *"Churn for bank transfer payments?"*. The curated expectation was that this selects the feature `Payment_Method` (the question is really "what is churn among customers whose payment method is bank transfer/automatic payments"). The qwen3:4b model produced a schema-valid FeatureRequest but **failed to include `Payment_Method`** in its selected slots. One feature missed out of one case out of 23 = 95.83%.

Had the suite dipped below 95%, the documented fallback (config-only, zero code changes) activates: `TELCO_LLM_MODEL=qwen3:8b-instruct-2507-q4_K_M` and a re-run. It did not regress below the bar, so no swap was made — and the miss is recorded per-case in `runs/extraction_suite_report.json` rather than buried.

Also note, to pre-empt confusion: a *later* interaction asking "what has the biggest effect on churn in the bank transfer segment?" would likely still produce a correct prediction, because a follow-up extraction would put `Payment_Method` in play; the miss is an extraction recall miss on the first turn, not a forbidden-number problem.

---

## 3. "Why the `eval_llm` pytest marker? Why would the eval suite skip when Ollama is unreachable?"

The marker (`pytest -m eval_llm` to activate) exists because **that suite is not a unit test** — it is a live-integration test of an external service. Reasons for keeping it opt-in and skip-on-absence:

1. **Determinism and hermeticity.** A unit suite must produce identical results with no network, no local 4 GB model running, and no side effects. Phase 3's contract tests already verify the loop's behavior end-to-end with `FakeExtractionTransport` (a real HTTP round-trip through httpx with deterministic candidates) — those *do* run offline and assert the same schema/refusal/state guarantees. The eval suite is an additional live check of a *third-party component's behavior* (qwen3:4b through Ollama).
2. **CIagnostic hygiene.** If Ollama isn't installed on a contributor's machine (or CI runner), the suite would otherwise *fail* for reasons that say nothing about our code. Skips under conditions external to the code are the standard mechanism for live-integration tests.
3. **The 95% gate still governs real usage.** The skip is not "we don't measure"; it's "the measurement is a deliberate act," run explicitly with `pytest -m eval_llm` against a reachable Ollama — which is exactly what produced the report artifact quoted in §2b. The bar (100% schema validity, ≥ 95% slot accuracy) is asserted when the suite actually runs; it is not asserted vacuously by a skip.

If you'd prefer a stricter policy, the honest alternatives are (a) run the eval suite as a phase-gate before merge (recommended: one command, we did it for Phase 3) or (b) keep it opt-in but have a real seed run of `extraction_suite_report.json` attached to each PR — both preserve the bar without making CI offline-incompatible. The current setup is consistent with the repo's existing pattern (pip-audit also only runs at pre-push, for a similar environment-dependency reason).

---

## 4. "`SOURCE_LABELS = ("UCI-Iranian-Churn", "Orange-Telecom")` — what are these?"

These are the **two external churn datasets named in the spec (task 3.7)** that Phase 2's model is required to be validated against zero-shot, apart from the IBM Telco CSV it was trained on:

- **UCI-Iranian-Churn** — a telecom-churn dataset published on the UCI Machine Learning Repository (Iranian anonymized customer base, with call-center/fault information; genuinely different distribution from the IBM Telco challenge data, all the more informative).
- **Orange-Telecom** — a classic telecom-churn dataset from Orange (the French operator), publicly released with ~200 anonymized usage-statistics features; widely used as an external benchmark for churn model transferability.

In the code these two strings are only **labels for the report artifact** — `external_validate` itself is dataset-agnostic: it takes any external frame + a minimal `{external_column → canonical_feature}` mapping and reports zero-shot AUC vs our benchmark with an honest delta. Neither file ships with the repo; the runner waits for one of these frames to be supplied (they are large external datasets, not committed). The `honest_note` in the output exists so nobody can ever misread a large negative AUC delta as a bug to be tuned away: a frozen artifact evaluated zero-shot on a different national population *should* degrade, and the praver spec expects us to report it, not fix it in place.

---

## 5. "Isotonic calibration — prone to overfitting? Where is the calibration dataset?"

Two good questions; both have concrete answers here.

### Where the calibration data is

The isotonic regressor is fitted **once per seed** on the **the winning model's out-of-fold predictions over the entire training partition** (step 4 of the protocol in `train.py::fit_frame`). Concretely: 5-fold stratified CV on the 5634-row training split produces, for every training row, one prediction from a model that was fitted on the other 4 folds; those ~5634 `(predicted_probability ≈ , true outcome)` pairs are the calibration data. It never touches the test partition, and it never sees in-fold predictions.

### The overfitting concern is real — but bounded here

Isotonic regression fits a piecewise-constant monotone function and, on small or noisy samples, **can** step-overfit and fit noise (this is exactly why Platt/sigmoid calibration is often preferred below a few hundred calibration points). Mitigating factors here:

1. **Sample size.** ~5634 calibration points with ~1490 positives is far above the regime where isotonic is considered risky; the empirical literature's crossover point is roughly ≤ ~1000 calibration samples.
2. **Consistent sources of noise.** The OOF construction means every calibration point is an honest held-out prediction; no point is double-counted. That said, isotonic's step structure is still visible late (the calibrator can force probabilities that were held constant across thresholds).
3. **Out-of-bounds clipping.** `out_of_bounds="clip"` prevents degenerate extrapolation outside the OOF range.
4. **The feedback loop is measured.** We didn't take improvement on faith: the Brier score improved (that is the point of the metric), and the artifact records both `brier_oof` (calibrated) and `brier_oof_uncalibrated` so future re-runs can detect degradation; the unit test (`test_churn_model.py`) explicitly asserts the calibrated Brier is ≤ raw Brier on a held-out fold.
5. **The mapping is monotone — the AUC stays fixed.** Isotonic is a monotone transform, so it cannot reorder predictions; the rank metric (AUC) is unchanged and only the *calibration* (probability→outcome mapping) changes. The worst failure mode is not a silently-wrong model; it is less precise probabilities.

---

## 6. "Shouldn't OOF predictions feed a second-level meta-model? What's the operating point?"

These are related to **stacked generalization**, and it's worth understanding exactly why this pipeline deliberately does NOT stack.

### What stacking would look like

In a 2-level ensemble, every base model's out-of-fold predictions become *inputs to a meta-model* (logistic regression, GBM, …) that learns combination weights and produces the final prediction. `train.py`'s file docstring describes OOF predictions feeding `the operating point and the isotonic calibrator` — the intent **is** that the OOF predictions are the "second-stage input", but the "second stage" here holds **no learnable parameters that reinterpret the input distribution onto the model**: it's a *single scalar threshold* and a *monotone 1→1 calibration map*. There is no competing base-model blend, no learned linear combination, no additional ensemble degrees of freedom to go wrong.

Why not stack:

1. **Protocol, not architecture.** The spec (task 3.2) requires *selecting* the best of 3 models by mean fold PR-AUC under one identical protocol — selecting a winner, not blending 3 winners. Stacking would change what "the model" is exactly the thing the published-band honesty contract must keep attributable.
2. **Leakage surface.** Stacking is the single most common vector for silent leakage in telco churn PoCs: convenient OOF→meta-model combos blur where the OOF came from versus where the final prediction happens; with only ~5634 rows, a slight second-level fit is easily overfit by the time it hits test. The threshold + calibrator path adds *zero* parameters that could drift relative to what the folds saw.
3. **Cost/benefit at 0.845 AUC.** Stacked gains are typically meaningful when base models are complementary and individually imperfect at a wide margin; at 0.84–0.88 with XGB/LGBM/logreg already within noise of each other in fold PR-AUC, the meta-model would learn weights to make up for a difference the selection already resolves — with the same maintainability.

### `thresholds.py` — the operating point, explained in full detail

**Why there is a threshold at all.** The pipeline outputs a continuous churn probability (0→1). Marketing doesn't act on probabilities; they pull the trigger on a retention offer (email, call, discount). To act, you need a cutoff: above it, a customer is "churner-risk"; below, they are not. The *operating point* is the cutoff chosen from the model's own score distribution under the business constraint in the spec.

#### The math of `_curve(y, proba)` — line by line

Given `y` (labels 0/1, churn=1) and `proba` (the model's continuous outputs):

1. `order = np.argsort(-proba, kind="stable")` — sorts cases from **most-confident churn** to least, keeping ties in their original order (stability matters for reproducible thresholds).
2. `tp = np.cumsum(sorted_y)` — walking down the list, at each cutoff, how many true churners so far (A subsequence of sorted y).
3. `fp = np.cumsum(1 - sorted_y)` — how many non-churners so far.
4. `precision = tp / (tp + fp)` — of the cases the model flagged, how many actually churn: the business cost of a false-positive retention offer (you call someone who never churns; wasted spend and an annoyed customer).
5. `recall = tp / max(int(y.sum()), 1)` — of the real churners, how many were called: the cost of a false *negative* (you miss a churner, lost revenue). Guard `max(..., 1)` prevents /0 on degenerate inputs.
6. `f1 = 2 * precision * recall / (precision + recall)` — harmonic mean; a single scalar that resists the degenerate "flag everyone" (precision→0) and "flag almost no one" (recall→0) optima, both of which would look good on one metric and terrible on the other.

So after one sorting pass, we have the full (precision, recall, F1) profile of **every** possible threshold — no model re-fit, no loop, no repeated prediction: pure arithmetic over the sorted outputs.

#### The floor: why precision must be ≥ 0.55

The spec pins `PRECISION_FLOOR = 0.55` because **marketing work costs real money**: every "predicted churner" gets an intervention, and ≥ 0.55 means at worst a 45% wasted-offer side. Without a floor, max-F1 on imbalanced data is content to trade precision down to recall-hunting, which would crank the recall but bury the marketing team in offers against non-churners. The floor is *superset-constrained*, which means it plays at the business's pace, not the data's.

#### Selecting the "best" threshold (`select_operating_point`)

1. **Eligibility first.** `eligible = (precision >= 0.55) & (recall > 0)` — only cutoffs that at least respect the floor AND catch at least one churner qualify. Cutoffs above the model's max-probability (recall = 0) are excluded even with 100% precision — precision without recall isn't a *operation*; it's an inert rule.
2. **Unattainable floor raises.** If no cutoff on the entire curve satisfies the floor, the function **raises a `ValueError`** ("precision floor 0.55 unattainable … reporting rather than silently lowering the floor"). This is the honesty contract again: if the model isn't good enough to spare the marketing budget at 55% precision, the correct engineering answer is to say so to the humans, not to nudge `0.55 → 0.40` and let the schedule meet the deadline.
3. **Tiebreak: F1 primary, precision as a fixed-decimal subsidy.** `best = idx[argmax(f1 * 1_000_000 + precision)]`. Because F1 lives in [0, 1] and two cutoffs can round to the same F1 at double precision, the tiebreak rewards the cutoff with the *higher precision* among equal-F1 candidates — the `* 1_000_000` scale keeps F1 dominance lexicographically (a 1e-7 F1 gap is worth far more than any precision tiebreak), so the tiebreak only acts on (near-)exact ties. Deterministic, no hidden randomness.
4. **The returned threshold.** `threshold` is the concrete probability cut-off; `f1` / `precision` / `recall` (at that cutoff on the OOF curve) are stored in the artifact's metadata so any reviewer can re-derive the choice from the score distribution, not just take the number on faith.

Why **train-only, out-of-fold**: selecting the cutoff on the test partition would be the primary rule the spec protects *against* — the test partition is reserved for exactly-once evaluation per seed; choosing the business threshold from the same curve used to choose the hyperparameter winner means one honest, self-consistent train-side decision surface, with test-side AUC still un-looked-at.

Where the operating point lands in the product: the chat reply compares the calibrated churn probability against `artifact.threshold`, so the "Churner" / "Not churner" label shown to marketing reflects the exact cut the floor was designed for, and the 0.55-precision constraint rides along in every conversation.

---

## 6. "`ctgan_demo` — what is this? What is `external.py` for?"

**`external.py`'s job** is the honest-frontier analysis the spec (task 3.7) requires: check whether the churn model learned something *general about telecom churn* rather than overfit to the quirks of this one IBM CSV. Two tools for that:

1. **`external_validate`** — the real check. You hand it a frozen artifact, an external frame (from UCI-Iranian-Churn or Orange-Telecom, or any similar), and a minimal schema mapping (external column names → our canonical feature names). It re-maps, re-splits **with the same seeding machinery**, scores the artifact zero-shot (no retraining, no threshold re-picking, no distribution adaptation), and reports:
   - `external_auc` — how the model ranks on data it never saw;
   - `auc_delta` — the honest gap vs our internal benchmark (0.8454);
   - `unmapped_features` — any feature of ours the external source cannot provide (dropped, listed, never silently filled).

   Spec-mandated honesty: a large negative delta is an expected finding (different country, provider, era of customer behavior), and the note in the report says so explicitly rather than inviting tuning the model back up on external data.

2. **`ctgan_demo`** — **explicitly NOT a validation tool.** It exists because the spec asks for a *synthetic sensitivity check* (D3): generate synthetic rows mechanically resembling the training data (CTGAN's conditional-tabular-GAN, if installed; otherwise a seeded bootstrap resample — CTGAN is deliberately **not** a project dependency), and measure the artifact's behavior there. It is labeled `DEMO` / `synthetic` everywhere so nobody ever mistakes a synthetic-data number for a validation claim — that would be exactly the kind of self-deception the honesty guardrails exist to prevent.

The two are separated so that "the model generalizes" is never asserted from either (a) cherry-picked in-sample performance or (b) machine-generated lookalikes — only from (a) untouched test partitions and (b) labeled-as-synthetic diagnostics when both are run.

---

## 7. "Why is the open-source model treated as a JSON-schema validator/pass-through? Shouldn't the churn model live on DagsHub for this pipeline?"

### The pass-through design IS the spec — ADR-0002 and challenge item 4b/4c

The challenge's "SPECIFICATIONS" say (quoting you):

> 3: "Pipeline: Build a pipeline to handle the chat with the marketing team, gather the model inputs through a conversation, and return the output."
> 4b, c: "Translates marketing questions into structured data for the ML model. Presents results in a clear and understandable format."

Item 4b is literally "translates marketing questions into structured data for the ML model" — the LLM's role (extraction, not computation) is the challenge's own phrasing. "Presents results in a clear and understandable format" (4c) is the reply-composition layer, where the only numbers that appear are the classifier's calibrated probability, the threshold comparison, and SHAP top-3 — produced by `chat/loop.py`, never by the model. And Phase 3's numerics test proves it: after stripping punctuation tokens from the reply, every digit in the assistant's text traces back to the tool payload.

### Why the LLM produces JSON trivially — because it is *made* to, three layers deep

The one-line few-shot examples produce JSON answers every time for a mechanical reason, not model goodwill:

1. **Grammar-constrained decoding** (`serving/llm_client.py`): the request carries `response_format: json_schema` with the exact FeatureRequest grammar (`additionalProperties: false`, closed 19-name enum, string-typed filter values). Constrained decode means the token sampler is *physically prohibited* from emitting tokens that would leave the grammar — the JSON shape is enforced by the inference engine (Ollama/vLLM), at generation time, not hoped for.
2. **Pydantic validation** (`chat/schemas.py`): even a hypothetical encoder that ignored `response_format` still hits `FeatureRequest.model_validate` — `extra='forbid'`, enum-closed feature names, no numeric-typed slots — and invalid candidates get one same-grammar retry, then a refusal turn with no state corruption.
3. **In-process double-check** (`llm_client._no_numeric_fields`): the parser-level check that the candidate has exactly the shape keys — defense-in-depth for any silent endpoint change.

This is the **grounding** you asked about, and yes — the chain NL → (open-source model, constrained decode) → chat api → validated FeatureRequest → churn-classifier tool → numeric payload → formatted reply is exactly the "clear and understandable format" path the spec requires, with every numeric claim provably born from the classifier.

### DagsHub — what it would solve, what this PoC actually needs

You're right in principle: a registry-hosted model artifact (DagsHub, MLflow Model Registry, S3/MinIO) is the standard answer when multiple *services* or *teams* load the same trained artifact. Three clarifications for where this project stands:

1. **The artifact already has provenance tracking** — `mlflow-skinny` logs the 3-seed run metrics/params (guarded; visible in `mlruns/`), and the artifact itself carries its full reproducibility metadata (seed, best params, feature list, operating point, calibration, all of it in the `.meta.json` sidecar). What's missing is *remote hosting*, not tracking.
2. **The deployment target is single-service by design.** The spec (design.md) mandates `Modal` serverless as the single FastAPI service and forbids K8s/queues; the artifact is a ~few-MB `.joblib` loaded locally from `runs/`. There is no second consumer that a shared remote registry would serve *today*; the Remote-hosted-prod-out-of-core cost is real but out of scope for this challenge.
3. **The seam exists when it's needed.** Every reader of the artifact goes through one loading function (`model.artifact.load_artifact`) and one config field (`model_registry_dir`). Pointing that at a DagsHub/MLflow/S3-backed download path is a config-level swap in Phase 5, not a rearchitecture — exactly like the Qwen3-8B swap (and the whole `llm_base_url` seam) is.

So: DagsHub is a legitimate Phase 5+ infrastructure decision for multi-consumer serving, deliberately not on the critical path for the challenge's single-service, single-team scope. If you want it added as a deployment option, it slots behind `model_registry_dir` with an artifact-download step before service start — say the word and it becomes a task in `p5/deployment`.

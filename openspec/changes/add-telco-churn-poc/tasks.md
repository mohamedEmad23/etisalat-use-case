# Tasks — add-telco-churn-poc

## 1. Phase 0 — Scaffolding

- [ ] 1.1 Create `etisalat-use-case/` uv project (pyproject.toml, src layout, Python 3.12+); verify `uv sync` and `uv run pytest` succeed with zero collected tests
- [ ] 1.2 Add dependencies (polars, scikit-learn, lightgbm, xgboost, optuna, shap, fastapi, uvicorn, mlflow, pytest, httpx); verify import smoke test passes
- [ ] 1.3 Create module skeleton `src/telco_churn/{data,model,chat,serving,api}`, `tests/{unit,eval,integration}`, `docs/`, `deploy/`; verify package imports and tree matches design.md layout
- [ ] 1.4 Implement env-driven config (LLM base URL, model id, bearer token with fail-fast when unset); verify unit test asserts startup failure without token

## 2. Phase 1 — Data pipeline (`telco-churn/data-pipeline`)

- [x] 2.1 Implement CSV ingestion (trim header names incl. `Senior_Citizen ` trailing space, assert 7,043×21 shape, drop customerID at load); verify unit test on the real CSV
- [x] 2.2 Implement cleaning (Total_Charges→numeric coercion pre-split with documented 11-blank-tenure-0-row rule, categorical trimming, 0/1→bool, sentinel strings kept distinct); verify unit tests including blank-row count == 11
- [x] 2.3 Implement `avg_monthly` engineering (Total_Charges/tenure); verify unit test values
- [x] 2.4 Implement deterministic stratified 80/20 split + 5-fold stratified fold builder (seeded); verify determinism test (same seed → identical partition) and train/test churn-rate match
- [x] 2.5 Write `docs/data-mismatches.md` (PDF "Postal check" vs CSV "Mailed check", named-column deltas); verify document reviewed against dataset

## 3. Phase 2 — Churn model (`telco-churn/churn-model`)

- [x] 3.1 Implement three candidate pipelines (logistic baseline, XGBoost, LightGBM) with imbalance handling inside sklearn Pipeline fitted on train folds only; verify leakage unit test
- [x] 3.2 Implement Optuna tuning + selection by mean 5-fold CV PR-AUC, reporting all three models; verify report artifact lists all three scores
- [x] 3.3 Implement operating-point selection (F1(churn)-max on train CV subject to precision ≥ 0.55); verify unit test with synthetic score curves
- [x] 3.4 Implement isotonic calibration fitted on train-fold predictions; verify Brier improvement on held-out validation fold
- [x] 3.5 Implement SHAP top-3 driver extraction per prediction; verify unit test returns exactly 3 drivers for a sample row
- [x] 3.6 Implement 3-seed evaluation (mean±std, published-band check AUC 0.84–0.88, above-band → leakage-audit gate, test set touched ≤ once per seed); verify eval run artifact produced
- [x] 3.7 Implement external validation runner (UCI Iranian Churn + Orange Telecom, minimal schema mapping, honest AUC deltas) + CTGAN sensitivity/demo-only path; verify report artifact labels synthetic use correctly
- [x] 3.8 Save reproducible artifact (model + calibrator + threshold + feature list + seed/config metadata) with light MLflow logging; verify artifact reload reproduces AUC within ±0.01

## 4. Phase 3 — Chat pipeline (`telco-churn/chat-pipeline`)

- [ ] 4.1 Implement Pydantic FeatureRequest schema (no numeric fields, closed feature-name enum); verify unit tests reject unknown features and numeric injection
- [ ] 4.2 Write extraction prompt + few-shot examples; verify prompt artifact review
- [ ] 4.3 Implement ~150-line tool loop (extract → grammar-constrained decode → validate → predict → compose response with numerics interpolated from payload); verify unit tests incl. malformed-candidate retry path
- [ ] 4.4 Implement in-memory multi-turn session state; verify two-turn refinement accumulates features without restatement
- [ ] 4.5 Implement PII redaction scrubber for logs; verify unit test scrubs emails/phones/customer IDs
- [ ] 4.6 Implement out-of-scope refusal path (no fabricated prediction); verify unit test
- [ ] 4.7 Build curated extraction eval suite (all 19 features + multi-intent + out-of-scope); verify suite runs against local Ollama with 100% schema validity and ≥95% slot accuracy, report artifact saved; on failure execute documented Qwen3-8B swap and re-run

## 5. Phase 4 — Chat API (`telco-churn/chat-api`)

- [ ] 5.1 Wire FastAPI `POST /chat` + `GET /health` with bearer auth; verify httpx tests: 401 unauthenticated, 200 authenticated round trip, health reflects LLM reachability
- [ ] 5.2 Implement stable JSON error contract (global handler, no stack traces to clients); verify test on forced internal error
- [ ] 5.3 Implement OWASP API Security Top 10 (2023) checklist + controls (rate limiting, secure defaults, inventory doc); verify checklist reviewed and rate-limit test passes
- [ ] 5.4 Build perf budget harness (p95 end-to-end, first-token latency instrumentation); verify harness produces measured numbers locally
- [ ] 5.5 Write docker-compose local stack (api + ollama); verify one-command end-to-end chat succeeds on CPU

## 6. Phase 5 — Deployment + docs (`telco-churn/deployment`)

- [ ] 6.1 Write `deploy/modal_app.py` (FastAPI + vLLM containers, pinned versions, scaledown_window ≈ 1800 s); verify staged `modal deploy` succeeds
- [ ] 6.2 Measure demo-GPU budgets (p95 ≤ 5 s, first token ≤ 1 s, VRAM headroom ≥ 20%); verify `docs/perf-report.md` records measured numbers
- [ ] 6.3 Write RunPod 3090 stop-pod fallback runbook; verify runbook reviewed (start/stop preserves storage)
- [ ] 6.4 Harden container (non-root, no secrets in image, resource limits, pinned base); verify image inspection audit
- [ ] 6.5 Write `docs/architecture.md` (Mermaid diagrams from design.md) + `etisalat-use-case/README.md`; verify docs complete and link-consistent
- [ ] 6.6 Append operations record to `wiki/log.md`; verify log entry exists (no wiki entries per user directive)

## 7. Final verification

- [ ] 7.1 Full unit suite green in CI mode (mocked LLM, no GPU): `uv run pytest tests/unit -q` passes with zero warnings
- [ ] 7.2 `openspec validate --changes` green and every spec scenario traceable to at least one test or delivered artifact

## Why

The client challenge (`etisalat-use-case/Telco Challenge for Data Science and AI.pdf`, IBM Telco churn spec) requires an end-to-end deliverable — churn classification model, natural-language chat pipeline over it, serving API, architecture diagram, and documentation — under hard constraints: open-source LLMs only, $0–20 budget, no local GPU, 6–7-day window. All architectural decisions were grilled and locked with the user (decision ledger D1–D14; ADR-0001, ADR-0002). This change turns those locked decisions into a buildable spec; nothing exists yet, so this change creates the system from zero.

## What Changes

- New self-contained Python project under `etisalat-use-case/` with a `uv`-managed environment (Python 3.12+), Polars, scikit-learn + LightGBM, Optuna, SHAP, MLflow (light), pytest.
- Data pipeline implementing the locked protocol: CSV ingestion (renamed Kaggle schema, trailing-space header), `Total_Charges` numeric coercion, 11 blank tenure-0 rows handling, stratified 80/20 split, leakage guardrails, `avg_monthly` engineering, documented spec/data mismatches.
- Churn classifier: LightGBM primary (+ logistic baseline, XGBoost comparison), threshold = F1-max at precision ≥ 0.55 on train CV, isotonic calibration, SHAP top-3 drivers per prediction, honest published-band targets (AUC 0.84–0.88, PR-AUC ≥ 0.65, F1 0.60–0.65, acc 0.82–0.86).
- Extract-only chat pipeline: Qwen3-4B-Instruct-2507 emits grammar-constrained feature requests (no numeric fields in schema); hand-rolled ~150-line tool loop; numerics string-interpolated from tool output only; curated extraction test suite (100% schema validity, ≥ 95% slot accuracy).
- FastAPI chat API: `POST /chat` + `GET /health`, bearer auth, PII redaction, in-memory sessions, OWASP API Security Top 10 (2023) checklist, measured budgets (p95 ≤ 5 s, first token ≤ 1 s).
- Serving: vLLM on Modal Starter (primary, scale-to-zero, `scaledown_window` ≈ 1800 s during demo windows), Ollama 4B Q4 local CPU dev, RunPod 3090 documented fallback; docker-compose for the local stack.
- External validation on UCI Iranian Churn + Orange Telecom (honest AUC-drop reporting); CTGAN restricted to sensitivity analysis + demo filler.
- Documentation: Mermaid architecture diagram, README, spec/data-mismatch notes, `wiki/log.md` operation records.

## Capabilities

### New Capabilities

- `telco-churn/data-pipeline`: Ingestion, cleaning, split protocol, and leakage guardrails for the 7,043-row dataset.
- `telco-churn/churn-model`: Training, threshold selection, calibration, per-prediction explanation, and honest evaluation of the churn classifier.
- `telco-churn/chat-pipeline`: NL question → schema-constrained feature request → tool execution → prediction → NL response, plus the curated extraction test suite.
- `telco-churn/chat-api`: The FastAPI surface (`/chat`, `/health`), auth, PII handling, session state, and measurable perf/security budgets.
- `telco-churn/deployment`: Packaging and deployment of api + LLM serving to Modal, with Ollama local dev and RunPod fallback paths.

### Modified Capabilities

(none — greenfield change; no existing specs)

## Impact

- **New code**: `etisalat-use-case/` subtree (`pyproject.toml`, `src/telco_churn/`, `tests/`, `deploy/`, `docs/`). No behavior changes to the existing wiki; `wiki/log.md` gets append-only entries.
- **Dependencies** (uv-managed): LightGBM, XGBoost, Optuna, SHAP, Polars, FastAPI, Uvicorn, Modal SDK, MLflow, pytest; vLLM in the cloud image only; Ollama as a dev-side service.
- **External services**: Modal (primary GPU serving), RunPod (documented fallback), Hugging Face Hub (model weights), Ollama (local dev).
- **Budget**: expected $0; hard cap $20. Timeline: 6 phases over 6–7 days.
- **Docs surface**: openspec artifacts, Mermaid diagram, `etisalat-use-case/README.md`, `wiki/log.md`.

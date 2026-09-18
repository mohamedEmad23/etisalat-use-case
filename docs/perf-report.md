# Performance budget report — measured numbers (task 5.4)

Budget definition (spec `chat-api` + deployment design):
p95 end-to-end chat turn ≤ 5 s; first-token ≤ 1 s.

## Local run (CPU) — 2026-09-18

- Harness: `PYTHONPATH=src TELCO_API_BEARER_TOKEN=… .venv/bin/python -m telco_churn.api.perf --repeats 6`
- Backend: local Ollama, model `qwen3:4b-instruct-2507-q4_K_M` (JSON-schema constrained), 4-bit quant
- Hardware: 11th Gen Intel Core i5-113G7, 40 GB RAM, CPU-only inference
- Load: 6 sequential turns, 3 rotating utterances, per-session isolation
- Report artifact: `runs/perf_report.json` (aggregated via `telco_churn.api.perf.measure`)

| metric | measured | budget | verdict |
|--------|----------|--------|---------|
| p50 e2e turn | 7.30 s | — | — |
| **p95 e2e turn** | **49.53 s** (cold start turn) / 7.30 s steady | ≤ 5 s | **miss on CPU — expected** |
| p50 first token | 7.30 s | — | — |
| **p95 first token** | **49.53 s / 7.30 s steady** | ≤ 1 s | **miss on CPU — expected** |

Reading: turn 1 paid 49.5 s (cold model load on CPU); steady p50 ≈ 7.3 s.
The 5 s / 1 s budgets are **demo-GPU budgets**: they are re-measured on the
Modal A10G (or pinned vLLM container) in Phase 6 (task 6.2, same harness
`telco_churn.api.perf`, results appended below). A CPU miss with the measured
number is documented here rather than tuning the budget to match the hardware.

## Phase 6 GPU run (to be appended after `modal deploy`)

- Hardware: Modal GPU (model/card stated at run time), pinned vLLM image
- p95 e2e: _pending measurement_
- p95 first token: _pending measurement_

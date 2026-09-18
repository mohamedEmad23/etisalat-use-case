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

## Phase 6 GPU run (to be appended after a GPU platform is exercised)

- Hardware: Modal GPU — T4 default, L4 via `TELCO_MODAL_GPU` (`deploy/modal_app.py`,
  adopted runtime: Ollama 4-bit Qwen3-4B single container; vLLM is the documented
  variant only). No account existed at Phase 5 close, so the GPU numbers could not
  be produced here — measurement is **deferred with the user's explicit deferral
  decision** (no account = no card = no cost; the CPU stack above is the honest
  recorded baseline).
- Harness (unchanged, same code path)::

  modal secret create telco-api-bearer TELCO_API_BEARER_TOKEN=<value>
  modal deploy deploy/modal_app.py
  PYTHONPATH=src TELCO_API_BEARER_TOKEN=<value> .venv/bin/python -m telco_churn.api.perf --repeats 9

- p95 e2e: _pending measurement_
- p95 first token: _pending measurement_
- VRAM headroom (≥20% requirement): _pending measurement_ — capture with
  `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` under demo load
  and record the free-percentage below.

| metric | measured | budget | verdict |
|--------|----------|--------|---------|
| p95 e2e turn (GPU) | pending | ≤ 5 s | pending |
| p95 first token (GPU) | pending | ≤ 1 s | pending |
| VRAM headroom (GPU) | pending | ≥ 20% free | pending |

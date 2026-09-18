# Proposal — add-demo-surface

## Why

The challenge deliverable is a PoC the marketing team must *see and use* — the PDF (p.1) requires "the marketing team to interact with the model through an LLM-powered chatbot using natural language" and deliverable 2.c asks for "clear documentation on how the marketing team can use the chatbot". Today the only interface is a curl-only JSON API with docs disabled: recruiters cannot demonstrate the system to themselves, and nothing visibly proves the qwen3:4b model is actually running and producing valid answers. The cheapest compliant option — one self-contained HTML page served by FastAPI itself (same origin, no second service, no JS build tooling) — closes that gap at near-zero cost. In the same step we adopt the deployment runtime decision deferred from Phase 4 review: Ollama (llama.cpp) + qwen3:4b in a Modal T4/L4 GPU container as the primary demo runtime, with vLLM demoted to a documented concurrency fallback and Modal itself deferrable (no account → no card → no cost; the local CPU compose stack remains a legitimate, fully transparent PoC demonstration).

## What Changes

- **New capability `demo-surface`**: FastAPI serves a single-file HTML demo client at `GET /demo` (inline CSS + vanilla JS only — no framework, no build step, no CDN/external assets). The page talks to the existing `POST /chat` (bearer token entered by the user, session identifier, existing `ChatIn`/`TurnOut` contract), renders churn probability / churn label / top-3 SHAP drivers from the tool payload, and renders the structured error contract (`error.code`, `error.message`) verbatim on failures.
- **`GET /health` extended** with the configured model identity (`llm_model`, `llm_backend`) so the page can display exactly which open-source model is answering.
- **Transparency mandate**: the demo path contains **no mocked LLM backend** — every recruiter-visible answer must originate from the actually-running open-source model (Ollama + `qwen3:4b-instruct-2507-q4_K_M`); a gated smoke check records model identity + one real turn + latency into `runs/demo_smoke_report.json`, and the user accepts the demo first through the HTML page. Hermetic unit suites keep deterministic test doubles (no GPU, no live LLM in CI).
- **Deployment runtime amendment (ADR-0001 amended)**: primary demo runtime becomes a single Modal GPU container (T4 default, L4 alternative) running the uvicorn API alongside an in-container Ollama server, weights cached in a Modal Volume (no re-download per cold start), artifacts served from a runs Volume, bearer token via Modal Secret, `scaledown_window ≈ 1800 s`. vLLM becomes a documented variant reserved for concurrent multi-user demos. Explicit deferral clause: without a Modal account, the local CPU compose stack (real Ollama, real qwen3:4b) remains the documented legitimate demo.
- **`docs/chatbot-usage.md`** added — the marketing-team usage guide (challenge deliverable 2.c); OWASP checklist gains the `/demo` route rows (API5 surface, API9 inventory).

## Capabilities

### New Capabilities

- `telco-churn/demo-surface`: the visible demo client — self-contained HTML page served by FastAPI, demo-to-`/chat` contract, real-model transparency (identity surfaced, no mock backend in served paths), and the usage documentation for the marketing team.

### Modified Capabilities

- `telco-churn/deployment`: primary LLM runtime changes from "vLLM-class runtime" to Ollama (llama.cpp) + qwen3:4b 4-bit inside the same Modal container as the API (weights in a Modal Volume); vLLM re-scoped to a documented concurrency fallback; local-parity requirement reworded so served demo paths always use the real LLM while unit tests stay GPU-less doubles; new demo-first deployment smoke requirement.
- `telco-churn/chat-api`: `Health endpoint` requirement extended to report configured model identity; `OWASP API Security Top 10 (2023) checklist` requirement updated so the documented inventory and surface analysis include `GET /demo`.

## Impact

- `src/telco_churn/api/` — new `static/demo.html` asset + `/demo` route in `create_app`; `/health` response fields.
- `deploy/modal_app.py` — written for the amended runtime (single container: ollama subprocess + uvicorn; T4 default / L4 alternative; Modal Volumes for weights and `runs/`; Secret for the bearer token; documented vLLM variant).
- `docs/adr/0001-modal-serverless-over-free-tier-vms.md` — amended (runtime decision).
- `docs/security-checklist.md` — `/demo` rows; `docs/chatbot-usage.md` — new.
- `tests/unit/test_demo_surface.py` (new), `tests/eval/test_demo_smoke.py` (new, gated `eval_llm`), `runs/demo_smoke_report.json` (generated).
- Dependency: deployment/chat-api deltas amend requirements introduced by `add-telco-churn-poc` (apply after, or concurrently with, that change being merged/archived).
- No dependency changes; no breaking API changes; `POST /chat` contract unchanged.

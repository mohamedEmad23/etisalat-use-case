# Design: add-demo-surface

## Context

The API (branch `p4/chat-api`) exposes `POST /chat` + `GET /health` only. Motivation is in proposal.md — Why. Current constraints that shape the approach:

- `src/telco_churn/api/app.py` builds the app in `create_app()` with docs disabled (`docs_url=None`); handlers + rate-limit middleware are installed there. A static route belongs in the same factory.
- The repo is Python-only (AGENTS.md: no JS/TS, no build tooling); the single-service rule (design.md of add-telco-churn-poc: "one FastAPI service + one LLM runtime") forbids a second hosted frontend or CDN assets.
- `LlmClient` already speaks an OpenAI-compatible endpoint; `TELCO_LLM_BASE_URL` fully selects the runtime (Ollama local, vLLM cloud, fake transport in unit tests only).
- ADR-0001 (docs/adr/0001-modal-serverless-over-free-tier-vms.md) currently names vLLM as the primary serving runtime on Modal Starter. User decision (proposal): amend to Ollama (llama.cpp) + Qwen3-4B-Instruct-2507 4-bit as the demo default, vLLM documented-only for concurrent use, with an explicit "defer Modal" clause.
- Transparency mandate (user): the served demo must talk to the real configured model — no mock/fake backend anywhere on a demo/serving path; hermetic unit tests keep deterministic test doubles (GPU-less CI), with live verification via the gated smoke check and user acceptance through the page.
- Modal-specific facts (verified against Modal docs): `scaledown_window` range is 2 s–20 min (demo target ≈1800 s = the 20-min cap); GPU functions need a payment method on file; T4 ≈ $0.59/h ≈ covered by the $30/month Starter free compute.

## Goals / Non-Goals

**Goals:**

- A recruiters-usable demo surface: one self-contained HTML page served by the FastAPI app itself at `GET /demo`, with zero external requests.
- Model-identity transparency surfaced on the page via `/health`.
- Deployment runtime amendment: single-container Ollama-on-Modal pattern with volume-cached weights, vLLM demoted to documented variant, deferral documented.
- A gated smoke check that records a real end-to-end turn (`runs/demo_smoke_report.json`) as deployment evidence.

**Non-Goals:**

- No JS/TS build system, no npm, no bundler, no external CSS/JS/fonts (zero CDN).
- No frontend framework, no SPA router, no state persistence beyond the page session (bearer token lives in page memory only, never in `sessionStorage`/`localStorage`).
- No new Python dependencies.
- No change to the extraction contract, the classifier, or any phase-1–3 modules.

## Decisions

**D1 — Single-file demo page at `src/telco_churn/api/static/demo.html`, served by `FileResponse` from `create_app()`.**
Rationale: the page must ship with the service (single origin, single service, no second deployment); inline CSS + vanilla JS keep it dependency-free and reviewable. The route is `GET /demo` with `include_in_schema=False` so it stays out of the OpenAPI inventory we publish for OWASP review (the inventory is documented separately — see chat-api delta).
Alternative considered: serving a directory with `StaticFiles` — rejected (mounts a wildcard, more surface for one file); embedding HTML in a Python string — rejected (unreviewable, no editor tooling).

**D2 — Bearer token is entered by the user in the page and held in memory only.**
Rationale: the shell must never embed a secret (gitleaks would flag it; committing a token would violate the secrets policy). The page keeps the token in a JS variable for the tab's lifetime; a page refresh clears it. The demo page itself is an unauthenticated static shell with no embedded data — justified exposure documented in the OWASP inventory row.
Alternative considered: `sessionStorage` persistence — rejected (token survives tab closure in same window; unnecessary for a demo).

**D3 — `/health` gains model identity fields (`llm_model`, `llm_backend`) read from `get_settings()` at request time.**
Rationale: recruiters must see which model produced the answers; reading config at request time (not at import) keeps the pattern already used by `require_auth` (env may change between requests) and makes artifact/runtime drift visible on the page itself.
Alternative considered: baking identity into the HTML at build time — rejected (stale the moment the runtime changes; contradicts transparency).

**D4 — `deploy/modal_app.py` uses a single-container pattern: one GPU function that runs both the API and `ollama serve`.**
Rationale (user-locked): Ollama (llama.cpp) + Qwen3-4B-Instruct-2507 4-bit is adequate for a single-user ~50-token extraction turn; vLLM's batching/PagedAttention advantage appears only under concurrent multi-user load (research evidence in b30), and its JIT compile hurts cold starts. One container = one scale-to-zero unit = idle cost zero.
Shape: one `@modal.web_server(port=8000)` GPU function (`gpu="T4"` default, L4 selectable via env) that starts `ollama serve` on 127.0.0.1:11434, waits for readiness, `ollama pull`s the model if missing, then execs uvicorn on 8000. Weights persist in `modal.Volume("ollama-models")` mounted at `/root/.ollama`; artifact joblib files ride in a volume mounted at `/srv/app/runs`; the bearer token comes from a `modal.Secret` (never committed). `scaledown_window=1800` (Modal's 20-minute cap).
Alternative considered: two containers (API CPU + Ollama GPU) — rejected: doubles cold-start coordination and breaks the "one FastAPI service + one LLM runtime" seam into a network hop we don't need for a demo; vLLM variant remains documented for the concurrent case.
Deferral: Modal is optional — with no account there is no card and no cost; the local compose stack is a legitimate PoC demonstration.

**D5 — Verification is two-layered: hermetic unit tests + a gated live smoke.**
Unit tests (`tests/unit/test_demo_surface.py`) cover the page shell (zero external URLs, no embedded token), the `/demo` route contract, and `/health` identity fields — all with test doubles, GPU-less. The live path is `tests/eval/test_demo_smoke.py` under the existing `eval_llm` marker: it must reach the real backend or it skips; when it runs it performs one real authenticated turn through `ChurnChatPipeline` and writes `runs/demo_smoke_report.json` (model identity, backend, turn latency, schema validity). Final acceptance is the user driving a real turn through the page — the transparency mandate's human gate.
Alternative considered: making unit tests hit the real model — rejected: violates the GPU-less CI constraint and makes green depend on a local daemon.

**D6 — This change amends `add-telco-churn-poc` deltas rather than forking them.**
The deployment delta in the parent change keeps its requirement headers; this change's MODIFIED blocks replace the vLLM-worded body text. Implementation ordering is coordinated at apply time: apply `add-telco-churn-poc` phases 5–6 first (or land them together on the same branch), then this change's tasks; `add-telco-churn-poc` tasks.md §6 wording is reconciled to the amended runtime in this change's tasks.

## Risks / Trade-offs

- [Artifact staleness: served page may render a payload computed from an older artifact than the repo's current training output] → the page always shows `/health` model identity; the smoke report records artifact identity per turn; regeneration is a single command (`telco_churn.model.train`).
- [Modal account not created; no card on file] → `modal_app.py` is written and reviewed locally but marked "ready, not exercised"; the deferral clause makes the local stack the legitimate demo path; no cost is incurred until the user opts in.
- [T4 VRAM (16 GB) vs 4-bit 4B model] → comfortable fit; VRAM headroom ≥20% is measured and recorded at deploy time per the deployment spec, not assumed.
- [Demo page cached stale in a browser] → identity surfacing (D3) makes drift visible on the page itself; `Cache-Control: no-store` on `/demo` keeps the shell fresh.
- [Smoke gate could pass against a wrong-backend] → the smoke report records backend + model identity read from live config, and the page displays the same fields, so a silent swap is detectable by inspection.

## Migration Plan

1. Implement on a feature branch (`p6/demo-surface`), created from `p5/deployment` lineage so history carries (branch placement is re-confirmed with the user at apply time).
2. Land order: `/demo` page + `/health` identity + unit tests + usage doc → `modal_app.py` + ADR amendment + deployment delta reconciliation → gated smoke.
3. Rollback: the demo route is additive and self-contained; reverting the branch removes it without touching phases 1–4 behavior. `modal_app.py` has no runtime effect unless explicitly deployed.

## Open Questions

None spec-blocking. Branch naming (`p6/demo-surface` vs folding into the `p5` cycle) is a user choice resolved at apply time; it does not affect the artifacts.

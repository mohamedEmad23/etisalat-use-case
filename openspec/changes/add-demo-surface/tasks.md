# Tasks: add-demo-surface

## 1. Demo surface

- [ ] 1.1 Create `src/telco_churn/api/static/demo.html` — self-contained page (inline CSS, vanilla JS, zero external requests, no frameworks/CDN/fonts): bearer-token field held in page memory only, session id input, message transcript, payload cards (churn probability, churn label, top-3 drivers), error banner rendering `{error.code, error.message}` for 401/429, and model-identity panel fed by `GET /health`. Verify: served HTML contains no `http(s)://` asset URL, no token literal, and no framework reference (grep + unit test).
- [ ] 1.2 Wire `GET /demo` in `create_app()` via `FileResponse` with `include_in_schema=False` and `Cache-Control: no-store`; extend `GET /health` to return `llm_model` and `llm_backend` read from `get_settings()` at request time. Verify: `tests/unit/test_demo_surface.py` — route serves the file, health exposes identity fields, page shell leaks nothing.
- [ ] 1.3 Write `docs/chatbot-usage.md` (quickstart for the marketing team: start the local stack, open `/demo`, enter token, ask a churn question, read the answer; aligns with challenge deliverable 2.c) and add the `GET /demo` unauthenticated-shell row to `docs/security-checklist.md` (OWASP API5/API9). Verify: doc cross-references the routes that exist in `app.py`; checklist row names the route and its justified exposure.
- [ ] 1.4 Full verification: `.venv/bin/python -m pytest -q` (zero new warnings) and `uv run pre-commit run --all-files` green after `git add -A`. Verify: both commands pass on the demo-surface commits.

## 2. Deployment runtime amendment

- [ ] 2.1 Create `deploy/modal_app.py` — single-container pattern: one GPU `@modal.web_server(port=8000)` function (T4 default, L4 via env) running `ollama serve` on 127.0.0.1:11434, readiness wait, pull-if-missing `ollama pull` of the configured model, then uvicorn on 8000; weights in `modal.Volume("ollama-models")` → `/root/.ollama`, artifacts volume → `/srv/app/runs`, bearer token via `modal.Secret`, `scaledown_window=1800`. Verify: `python -c "import ast; ast.parse(open('deploy/modal_app.py').read())"` parses; dry structure matches design D4; file marked "ready, not exercised — requires user's Modal account".
- [ ] 2.2 Document the vLLM variant for concurrent multi-user serving (when to switch, what changes in `deploy/modal_app.py`, `TELCO_LLM_BASE_URL` value) and the deferral clause (no account → local compose stack is the legitimate demo path) in `docs/adr/0001-modal-serverless-over-free-tier-vms.md` as an amendment. Verify: ADR status updated to amended-with-date, decision text matches deployment delta.
- [ ] 2.3 Reconcile `openspec/changes/add-telco-churn-poc/tasks.md` §6 wording to the amended runtime (Ollama single-container default, vLLM documented variant) so the two changes do not contradict. Verify: `grep -n "vLLM" openspec/changes/add-telco-churn-poc/tasks.md` shows only variant references, no primary-runtime claims.
- [ ] 2.4 Full verification repeat: pytest + pre-commit green on the deployment-amendment commits. Verify: both commands pass.

## 3. Transparency verification

- [ ] 3.1 Write `tests/eval/test_demo_smoke.py` under the `eval_llm` marker: reach the real backend (skip if unreachable), perform one real authenticated chat turn through `ChurnChatPipeline`, write `runs/demo_smoke_report.json` recording model identity, backend, turn latency, and schema validity; a failing backend produces a loud report, never a fabricated payload. Verify: gated run against local Ollama writes the report; unit-mode run skips cleanly.
- [ ] 3.2 Final system verification and user acceptance: full pytest zero-warnings, pre-commit green, then the user drives one real turn through `/demo` against the live local Ollama stack and confirms the answer, drivers, and model identity visible on the page. Verify: `runs/demo_smoke_report.json` exists with a real turn; user confirms acceptance in chat.

## 4. Coordination

- [ ] 4.1 Confirm branch placement with the user at apply time (suggested: `p6/demo-surface` created from the p5/deployment lineage so history carries; user decides fold vs. new branch). Verify: user's recorded choice in the session before the first implementation commit.

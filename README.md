# telco-churn — IBM Telco (Etisalat) churn challenge

Churn classifier + extract-only LLM chat pipeline + FastAPI service. Built for the IBM Telco
(Etisalat) data-science challenge: a marketing team asks churn questions in natural language, an
open-source LLM converts each question into a **schema-validated feature request** (it never
produces numbers), and a gradient-boosting classifier computes the churn probability.

## What it does

| Phase | Capability | Where |
| --- | --- | --- |
| 1 | Data pipeline: ingest → clean → stratified split | `src/telco_churn/data/` |
| 2 | Churn model: Optuna-tuned logistic/XGBoost/LightGBM, operating point (F1-max @ precision ≥ 0.55), isotonic calibration, SHAP top-3 drivers, 3-seed eval in the honest AUC band 0.84–0.88 | `src/telco_churn/model/` |
| 3 | Chat pipeline: extract-only LLM, grammar-constrained JSON, sessions, PII redaction, refusals | `src/telco_churn/chat/` |
| 4 | Chat API: `POST /chat`, `GET /health`, bearer auth, error contract, rate limiting, OWASP checklist | `src/telco_churn/api/` |
| 5 | Deployment: single-container Modal (Ollama + API), local compose stack, RunPod fallback | `deploy/` |

The LLM's only channel into the system is the closed `FeatureRequest` schema (feature names +
filters, no numeric fields). Every number in a chat answer is interpolated from the classifier's
prediction payload — the LLM never generates numbers.

## Quickstart (local demo stack)

Prerequisites: Python 3.13 via [uv](https://docs.astral.sh/uv/), Docker (for the compose stack),
and Ollama running locally (`ollama serve`).

```sh
uv sync                                   # create .venv
# train (optional — artifacts ship in runs/):
PYTHONPATH=src .venv/bin/python -m telco_churn.model.train --n-trials 20
# serve:
TELCO_API_BEARER_TOKEN=local-demo-token \
.venv/bin/python -m uvicorn telco_churn.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Or one-command compose:

```sh
docker compose -f deploy/docker-compose.yml up --build
```

Then `GET /health` is open, and every other call needs the bearer header:

```sh
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TELCO_API_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "Will a fiber-optic customer on a month-to-month contract churn?"}'
```

## Documentation map

- `docs/architecture.md` — system diagram + data flow (challenge deliverable 3)
- `docs/chatbot-usage.md` — how the marketing team uses the chatbot (deliverable 2.c)
- `docs/perf-report.md` — measured latency budgets (CPU baseline; GPU procedure)
- `docs/security-checklist.md` — OWASP API Security Top 10 (2023) audit
- `docs/ops-log.md` — deployment operations record
- `docs/adr/` — architecture decisions (Modal serverless; extract-only grammar-constrained LLM)
- `docs/phase*-doc.md` — per-phase implementation rationale
- `openspec/` — spec-driven development artifacts (proposals, specs, tasks)

## Tests

```sh
.venv/bin/python -m pytest -q        # unit suite (GPU-less, deterministic doubles)
```

Live-model eval suites (`tests/eval/`) are opt-in via the `eval_llm` marker and run against a real
local Ollama endpoint — no mock is used on any served path.

## License & scope

Open-source-licensed models only (Qwen3, Apache-2.0). Single-service deployment by design —
no Kubernetes, no queues, no second frontend host.

# Phase 4 — Chat API: documentation of the implementation

Branch `p4/chat-api` (created from `p3/chat-pipeline`). Implements tasks 5.1–5.5 of
`openspec/changes/add-telco-churn-poc/tasks.md` §5 (chat-api), all marked `[x]`.

Commit: `ebdeea9` — `feat(api): phase 4 chat API — POST /chat + /health, bearer auth, error contract, rate limiting, perf harness, OWASP checklist, compose stack` (14 files, +812/−5).

---

## 1. Aim and contract

Expose the Phase 3 chat pipeline as a small, secured HTTP service so the PoC can be
reached over a network without importing Python:

| Task | Deliverable | Where |
|------|-------------|-------|
| 5.1 | `POST /chat` + `GET /health` with bearer auth | `src/telco_churn/api/app.py` |
| 5.2 | Stable error contract `{error: {code, message}}` | `src/telco_churn/api/errors.py` |
| 5.3 | OWASP API Security Top-10 checklist | `docs/security-checklist.md` |
| 5.4 | Perf p95 harness (measured, honest numbers) | `src/telco_churn/api/perf.py`, `docs/perf-report.md` |
| 5.5 | `docker-compose` (api + ollama) | `deploy/docker-compose.yml`, `deploy/Dockerfile`, `deploy/.dockerignore` |

Contract kept from Phase 3, unchanged: the LLM never produces numbers or predictions;
the classifier computes Churn probability; the chat loop interpolates numerics only
from tool output (`payload`).

## 2. File-by-file rationale (implementation steps)

### 2.1 `src/telco_churn/api/errors.py` — the stable error contract (task 5.2)

Single helper `error_body(code, message)` produces
`{"error": {"code": "...", "message": "..."}}`. A tiny `JsonError(status, code, message)`
exception class carries status with the body. Four stable helpers:

- `unauthorized()` → 401 / code `unauthorized`
- `rate_limited()` → 429 / code `rate_limited`
- `bad_request(msg)` → 400 / code `invalid_request`
- `internal_error()` → 500 / code `internal_error`, message
  *"internal error — details are withheld"*

`install_handlers(app)` registers three handlers:

1. `JsonError` → its own status + body;
2. `RequestValidationError` (FastAPI body validation) → 400 `invalid_request`
   ("request body failed validation") — the client never sees pydantic internals;
3. catch-all `Exception` → 500 `internal_error`, while `logger.exception` records the
   real traceback server-side behind a **redacted URL** (`redacted_url` strips query
   strings before logging).

Why this shape: the spec requires one stable error contract for every failure path;
stack traces, exception class names, and URLs with embedded tokens must never leave
the process.

### 2.2 `src/telco_churn/api/rate_limit.py` — API4/API6 control

- `RateLimiter(max_calls, window_seconds)`: fixed sliding window with a `deque` per
  key; `allow(key) -> bool` is the only decision surface (pure, unit-testable).
- `client_key(request)`: `sha256(token)[:16]` prefixed `token:` when a bearer token is
  present, else `addr:<host>`. We rate-limit **per credential** (hash, not raw token —
  the raw token is never logged), falling back to source address for anonymous calls
  (only `/health` is anonymous and it is not rate-limited).
- `RateLimitMiddleware(BaseHTTPMiddleware)`: scoped to `path='/chat'`; on overflow it
  answers 429 with the contract body and a `Retry-After` header.

Defaults live in config: `rate_limit_calls = 60`, `rate_limit_window_seconds = 3600`
(60 chat turns/hour/token — generous for a demo, honest ceiling for abuse).

### 2.3 `src/telco_churn/api/app.py` — the service (task 5.1)

- `create_app(*, artifact=None, llm=None, store=None, limiter=None)` builds the FastAPI
  app with **docs disabled** (`docs_url=None, redoc_url=None`) — minimal public surface.
  Every dependency can be injected for tests; defaults resolve from real config:
  artifact via `load_default_artifact()` (runs/churn_artifact_seed*.joblib), LLM via
  `LlmClient()` (base URL/model from settings), pipeline = `ChurnChatPipeline(artifact, llm, store)`.
- `POST /chat` with `response_model=TurnOut` and `Depends(require_auth)`:
  - `ChatIn(message, session_id)` — pydantic, both fields `min_length=1`; nothing else
    is accepted (API3 surface minimization).
  - The handler logs the (redacted) turn, then delegates to the Phase 3 loop.
- `GET /health` — **unauthenticated** (it exists to answer "is this up and is the LLM
  reachable"); returns `{"service": "ok", "llm": "reachable"|"unreachable"}` using the
  new `LlmClient.reachable()`.
- `require_auth`: reads `Authorization` header, requires `Bearer ` prefix, compares
  against `get_settings().api_bearer_token` with `hmac.compare_digest` (constant-time —
  no string-equality timing oracle). The token is read **at request time** via
  `get_settings()`, not memoized at import time — this matters for tests and for
  containers where the env is set after process start.
- Request logging: `logging.getLogger("telco_churn.api.requests")` writes each turn's
  session id and message through `chat.redact.redact()` — emails/phones/customer IDs
  never reach logs (verified by a caplog-based test).

### 2.4 `src/telco_churn/config.py` — rate-limit fields

Added only: `rate_limit_calls: int = 60`, `rate_limit_window_seconds: int = 3600`
(annotated "OWASP API4/API6 control; per-token in-process window"). Nothing else changed.

### 2.5 `src/telco_churn/serving/llm_client.py` — `reachable()`

New method: `GET {base_url}/models` with a hard 5 s timeout; catches
`httpx.HTTPError | httpx.InvalidURL | OSError` → `False`. Used only by `/health`, so
health checks stay honest and cheap (and never raise).

### 2.6 `src/telco_churn/api/perf.py` — the honest perf harness (task 5.4)

- Budgets as named constants: `BUDGET_P95_E2E_SECONDS = 5.0`,
  `BUDGET_FIRST_TOKEN_SECONDS = 1.0` (demo-GPU budgets from the design doc).
- `TimedTransport` wraps any httpx transport and records per-request
  **time-to-first-byte** (including `response.read()`), giving first-token/LLM-hop and
  end-to-end timings without monkey-patching the pipeline.
- `measure(pipeline, *, session_id, repeats)` instruments the pipeline's LLM client
  transport, runs `repeats` chat turns round-robin over three fixed sample utterances,
  and returns a report dict: `p50/p95` for both e2e and first-token, raw samples, and
  `budgets` with pass/fail booleans. Nothing is rounded upward; numbers are recorded
  with measured values (no "feels fast" adjectives).
- `report_path()` = `settings.model_registry_dir / "perf_report.json"`; `main()` CLI:
  `PYTHONPATH=src .venv/bin/python -m telco_churn.api.perf --repeats 9`.

### 2.7 Tests

`tests/unit/test_chat_api.py` (12 tests, no network — LLM behind
`FakeExtractionTransport`):

- `TestAuth`: missing header → 401 `unauthorized` contract body; wrong token → 401;
  correct token → 200 round-trip with `{session_id, text, payload}` and a payload
  containing `churn_probability`, `churn_label`, exactly 3 drivers.
- `TestHealth`: reachable LLM → `"llm": "reachable"`; dead transport → `"unreachable"`;
  `/health` needs no token.
- `TestErrorContract`: monkeypatched `pipeline.handle` raising `RuntimeError` with
  secret material → 500 `internal_error` and the secret **and** the class name absent
  from the response; missing `message` field → 400 `invalid_request`;
  out-of-scope candidate → 200 with `payload: null` and refusal text.
- `TestRateLimiting`: limiter with `max_calls=1` → first call 200, second 429 with
  contract body; window logic unit tests (blocked key, different key allowed).
- `TestRedactionLogging`: caplog proves `user.name@example.com`, `0805553434`,
  `7590VHVEG` are absent and `[EMAIL]`, `[PHONE]`, `[CUSTOMER_ID]` present in logs.

`tests/integration/test_perf_harness.py` — `eval_llm`-gated (skips without a reachable
Ollama): runs `measure(repeats=3)`, writes `runs/perf_report.json`, asserts the
budget constants and sample count; numbers are recorded regardless of pass/fail.

### 2.8 `docs/security-checklist.md` — OWASP API Security Top 10 (2023) (task 5.3)

Auditable table, one row per risk (API1–API10), each with the control and the test
that proves it. Highlights: API1 (no object store — session aggregates only),
API2 (bearer + `hmac.compare_digest` + env-only, min-8 chars, verified by tests),
API3 (two-field request surface), API4 (rate limiter + timeouts), API5 (two routes,
docs disabled), API8 (fail-fast token, generic errors; non-root container in Phase 5),
API10 (single grammar-constrained LLM seam), plus log-hygiene row verified by test.

### 2.9 `deploy/` — compose stack (task 5.5)

- `deploy/Dockerfile`: `python:3.13-slim`, non-root user `telco`, `uv sync --frozen
  --no-dev`, uvicorn `telco_churn.api.app:app` on port 8000.
- `deploy/docker-compose.yml`: `ollama` service (volume for models, healthcheck
  `ollama ls`) + `api` service (env: `TELCO_API_BEARER_TOKEN`, `TELCO_LLM_BASE_URL=http://ollama:11434/v1`,
  `TELCO_LLM_BACKEND=ollama`, `TELCO_API_HOST=0.0.0.0`; read-only mount of `../runs`;
  `depends_on: ollama service_healthy`). Header comment carries a curl smoke-test
  using a `${TELCO_API_BEARER_TOKEN}` reference (gitleaks-safe). Validated with
  `docker compose -f deploy/docker-compose.yml config --quiet`.
- `deploy/.dockerignore`: keeps the image lean (no data/, docs/, runs/, .venv, .git…).

## 3. Measured performance (honest numbers)

Local CPU run (11th Gen i5-113G7, 40 GB RAM, CPU-only qwen3:4b via Ollama):

```
PYTHONPATH=src TELCO_API_BEARER_TOKEN=local-cli-token .venv/bin/python -m telco_churn.api.perf --repeats 6
```

- steady-state p50 end-to-end ≈ **7.30 s** per chat turn
- cold-start first turn ≈ **49.5 s** (model load dominates)
- p95 e2e ≈ 49.53 s; p95 first-token ≈ 49.53 s (cold start sets the p95)
- budgets p95 ≤ 5.0 s / first-token ≤ 1.0 s → **fail on CPU** (expected: these are
  demo-GPU budgets; recorded honestly, re-measured on GPU in Phase 6 with the same
  harness — no number is being hidden)

Full numbers in `docs/perf-report.md` (hardware stated) and `runs/perf_report.json`.

## 4. Verification evidence

- `.venv/bin/python -m pytest -q` → **56 passed** (54 unit + eval-gated integration);
  4 benign third-party warnings (starlette deprecation via testclient, shap
  LightGBM TreeExplainer note) — no project warnings.
- `uv run pre-commit run --all-files` → **all 12 hooks Passed** (after fixing: 1
  gitleaks `curl-auth-header` fingerprint in the compose comment — replaced with an
  env-var reference; 2 ruff findings in `perf.py` — ISC004 parenthesized implicit
  concatenation, RUF046 redundant `int()`).

## 5. Deferred to later phases

- Full end-to-end image build + `docker compose up` (multi-GB Ollama model pull) —
  user-run; compose config validated statically.
- GPU perf re-measurement and Modal/RunPod deployment — Phase 5/6 (`p5/deployment`
  tasks 6.1–6.6) with the same `perf.py` harness.
- Frontend/UI — not in the current OpenSpec change (discussed separately with the
  user; see the phase-4 Q&A answer).

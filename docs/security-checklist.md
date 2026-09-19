# OWASP API Security Top 10 (2023) — Checklist & Controls

Auditable checklist for `telco-churn` FastAPI service (Phase 4, task 5.3).
Reviewed against the running service; every item is either addressed by a
control or explicitly justified as not applicable.

| # | OWASP (2023) | Control in this service | Verified by |
|---|--------------|-------------------------|-------------|
| API1 | Broken object level authorization | No object store at all: the API exposes only stateless churn predictions keyed by caller-supplied `session_id`; sessions carry aggregate profile words, never customer records. There are no customer records to address-level-access. | W8: `session.py` snapshot; no `/customers`-style routes exist |
| API2 | Broken authentication | Bearer token required on every non-`/health` request (`require_auth`), constant-time comparison (`hmac.compare_digest`), token from env `TELCO_API_BEARER_TOKEN` (min length 8, no insecure default), 401 before any LLM/model work is queued. | `tests/unit/test_chat_api.py::TestAuth::test_401_when_header_missing`, `::test_401_when_token_invalid` |
| API3 | Broken object property level authorization | Only two request fields accepted (`message`, `session_id`); pydantic rejects unknown/absent properties; nothing else is readable or writable through the API. | `::TestErrorContract::test_malformed_body_maps_to_invalid_request` |
| API4 | Unrestricted resource consumption | In-process fixed-window per-token limiter (`RateLimiter`, defaults 60 calls/3600 s) returns HTTP 429 with the stable error body + `Retry-After`. `httpx` timeouts bound LLM waits (600 s / 10 s connect; health probe 5 s). | `::TestRateLimiting::test_extra_call_is_429_with_contract_body` |
| API5 | Broken function level authorization | Surface is exactly `POST /chat` + `GET /health` + `GET /demo`; the demo route serves one static, data-free HTML shell (`include_in_schema=False`) and exposes no function beyond reading the page; no admin/debug routes; FastAPI docs disabled (`docs_url=None, redoc_url=None`). | `create_app` source; `tests/unit/test_demo_surface.py::TestDemoRoute` |
| API6 | Unrestricted access to sensitive business flows | The sensitive flow (churn inference) is the same token-gated, rate-limited `/chat` call; one token = one rate-limit budget; no bulk/download capabilities exist. | `::TestRateLimiting` |
| API7 | Server side request forgery | No endpoint accepts a URL or host; the only outbound call is to the configured LLM base URL from settings; user text reaches that client only as chat message content. No redirect-following client. | Code review: `LlmClient` fixed `base_url` from settings |
| API8 | Security misconfiguration | Fail-fast required bearer token (no default), docs UI disabled, generic error mapping (no stack trace replay), non-health errors auditable; container runs non-root (Phase 5 hardening). | `::TestErrorContract`; config validators in `config.py` |
| API9 | Improper inventory | Documented inventory: `POST /chat` (bearer-gated), `GET /health` (open, no data beyond liveness + model identity), `GET /demo` (open static shell — no embedded data, no token; justified exposure), kept in `src/telco_churn/api/app.py` and this document + `README.md` (Architecture section). | This file; `tests/unit/test_demo_surface.py::TestPageShellLeaksNothing`; spec `chat-api` inventory clause |
| API10 | Unsafe consumption of third party APIs | LLM consumed through the single `serving/llm_client.py` seam, grammar-constrained (`json_schema` decode + pydantic `extra='forbid'` + `_no_numeric_fields`), validated before any tool executor runs; refund paths to a failing backend have no fast retries and no secrets sent. | Phase 3 extraction suite; `::TestAuth` keeps `/clients`-heavy data away from the LLM |

Log hygiene: request logs are scrubbed via `chat.redact.redact` (emails,
7+ digit phone clusters, `NNNNAAAAA` usernames) before persisting —
`::TestRedactionLogging::test_log_line_is_scrubbed`. Exception messages are
logged server-side only; clients receive `internal_error` with details withheld.

Compliance status: every scenario in `openspec/specs/telco-churn/chat-api/spec.md`
(the synced main spec; the originating change is archived) is traceable to a test
in `tests/unit/test_chat_api.py` / `tests/unit/test_demo_surface.py` or a control
documented above; the performance-budget numbers are recorded separately in
`docs/perf-report.md`.

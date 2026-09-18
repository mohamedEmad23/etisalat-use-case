# chat-api — Delta

## MODIFIED Requirements

### Requirement: Health endpoint

The API SHALL expose `GET /health` reporting service liveness, LLM backend reachability, and the configured LLM model identity (model identifier and backend name).

#### Scenario: Healthy service check

- **WHEN** `/health` is called while the LLM backend is reachable
- **THEN** the response indicates both service and LLM status as available

#### Scenario: Degraded backend surfaced

- **WHEN** `/health` is called while the LLM backend is unreachable
- **THEN** the response marks the LLM status as unavailable without crashing the endpoint

#### Scenario: Model identity surfaced

- **WHEN** `/health` is called on a running deployment
- **THEN** the response includes the configured open-source model identifier and backend name

### Requirement: OWASP API Security Top 10 (2023) checklist

The API SHALL document an auditable checklist addressing all ten items of OWASP API Security Top 10 (2023) — including rate limiting against unrestricted resource consumption, no object-level data exposure, no SSRF surface, secure defaults for misconfiguration, and explicit inventory (documented endpoints/auth model, including the unauthenticated static demo shell `GET /demo` with its exposure justified: static page, no embedded data, `/chat` remaining bearer-gated) — and the checklist MUST be shipped in the docs.

#### Scenario: Checklist audit passes

- **WHEN** the security checklist is reviewed against the running service
- **THEN** every one of the ten items is either addressed by a control or explicitly justified as not applicable
- **AND** the documented inventory lists `POST /chat` (auth), `GET /health` (open), and `GET /demo` (open static shell, no embedded data)

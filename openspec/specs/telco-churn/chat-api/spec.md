# telco-churn/chat-api Specification

## Purpose

Expose the chat pipeline over HTTP with a minimal authenticated surface: one prediction endpoint, one health endpoint, bearer-token auth, audited security posture, and measured performance budgets. Must be independently usable as the public face of the service and auditable against OWASP API Security Top 10 (2023).

## Requirements

### Requirement: Chat endpoint

The API SHALL expose `POST /chat` accepting a user message and session identifier, returning a structured JSON prediction response (churn probability, calibrated confidence, top drivers, human-readable explanation).

#### Scenario: Successful prediction round trip

- **WHEN** a valid authenticated request posts "Will this customer churn? Contract is month-to-month, fiber internet, tenure 3 months."
- **THEN** the response is JSON with a churn probability, its drivers, and a readable explanation
- **AND** the HTTP status is 200

### Requirement: Health endpoint

The API SHALL expose `GET /health` reporting service liveness and LLM backend reachability.

#### Scenario: Healthy service check

- **WHEN** `/health` is called while the LLM backend is reachable
- **THEN** the response indicates both service and LLM status as available

#### Scenario: Degraded backend surfaced

- **WHEN** `/health` is called while the LLM backend is unreachable
- **THEN** the response marks the LLM status as unavailable without crashing the endpoint

### Requirement: Bearer authentication

The API SHALL require a bearer token on every non-health request and return HTTP 401 with a JSON error body when the token is missing or invalid.

#### Scenario: Unauthenticated request rejected

- **WHEN** `POST /chat` is called without an Authorization header
- **THEN** the response is HTTP 401 with a structured JSON error and no prediction is computed

### Requirement: PII redaction in logs

The API SHALL redact personally identifying information (names, email addresses, phone numbers, customer identifiers) from all request logs before persisting them.

#### Scenario: Redaction on logged input

- **WHEN** a request containing an email address or customer ID is logged
- **THEN** the persisted log line contains no unredacted email address or customer ID

### Requirement: OWASP API Security Top 10 (2023) checklist

The API SHALL document an auditable checklist addressing all ten items of OWASP API Security Top 10 (2023) — including rate limiting against unrestricted resource consumption, no object-level data exposure, no SSRF surface, secure defaults for misconfiguration, and explicit inventory (documented endpoints/auth model) — and the checklist MUST be shipped in the docs.

#### Scenario: Checklist audit passes

- **WHEN** the security checklist is reviewed against the running service
- **THEN** every one of the ten items is either addressed by a control or explicitly justified as not applicable

### Requirement: Measured performance budgets

The API SHALL meet, and report measured numbers for, the demo-GPU budgets: p95 end-to-end chat turn latency ≤5 s and first-token latency ≤1 s; budget verification results MUST be recorded in the docs with hardware and load conditions stated.

#### Scenario: Budget measurement recorded

- **WHEN** the performance budget run executes against the demo GPU configuration
- **THEN** measured p95 end-to-end and first-token latencies are recorded in the docs
- **AND** any budget miss is documented with the measured number, not the adjective

### Requirement: Structured JSON error contract

All error responses SHALL be structured JSON with a stable error shape; stack traces and internal details MUST NOT leak to clients.

#### Scenario: Internal error masked

- **WHEN** an unhandled exception occurs inside the pipeline during a request
- **THEN** the client receives a structured JSON error (stable code, generic message)
- **AND** the response body contains no stack trace

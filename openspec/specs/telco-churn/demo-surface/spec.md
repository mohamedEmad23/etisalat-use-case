# telco-churn/demo-surface Specification

## Purpose

Provides the visible demonstration surface for the marketing team and reviewers: a single self-contained HTML page served by FastAPI that converses with the existing authenticated `/chat` endpoint and transparently proves the real open-source model is running and producing valid answers.

## Requirements

### Requirement: Self-contained demo page

The service SHALL serve a single-file HTML demo client at `GET /demo` (excluded from the OpenAPI schema) that is fully self-contained: inline CSS, vanilla JavaScript only, no JavaScript frameworks or build step, and no requests to any origin other than the serving origin.

#### Scenario: Page loads with zero external assets

- **WHEN** a browser opens `/demo`
- **THEN** the response is HTTP 200 `text/html` and the rendered page makes no network requests to third-party origins

#### Scenario: Page shell leaks nothing

- **WHEN** the `/demo` HTML is fetched
- **THEN** it contains no bearer token, no customer data, and no model artifact contents

### Requirement: Demo chat over the authenticated API

The demo client SHALL converse with the existing `POST /chat` using the bearer token entered by the user, a user-visible session identifier, and the existing request/response contract; prediction payloads SHALL be rendered as churn probability, churn label, and top-3 drivers, and non-2xx responses SHALL render the structured JSON error contract verbatim.

#### Scenario: Successful prediction round trip

- **WHEN** a user pastes a valid bearer token and submits a churn question
- **THEN** the page POSTs to `/chat` with the Authorization header and renders the assistant text plus the payload's churn probability, churn label, and top-3 drivers

#### Scenario: Error contract surfaced

- **WHEN** `/chat` responds 401 (missing/invalid token) or 429 (rate limited)
- **THEN** the page displays the structured `error.code` from the JSON error body and never a stack trace

### Requirement: Real-model transparency

The served application SHALL contain no mock or fake LLM backend in any demo/serving path: the default application SHALL resolve the real OpenAI-compatible client against the configured `TELCO_LLM_BASE_URL`, the page SHALL display the configured model identity (model identifier + backend) obtained from `/health`, and a gated smoke check SHALL record the model identity, one real chat turn, and its latency into `runs/demo_smoke_report.json`.

#### Scenario: Health exposes model identity

- **WHEN** `/health` is called on a running stack
- **THEN** the response includes the configured open-source model identifier and backend name alongside reachability

#### Scenario: Demo smoke records a real turn

- **WHEN** the demo stack is up (local compose or serverless) and the LLM endpoint is reachable
- **THEN** the gated smoke check performs one real `/chat` turn and records the answering model identity, backend, and measured turn latency in `runs/demo_smoke_report.json`

#### Scenario: No fabricated answers when the backend is down

- **WHEN** the LLM backend is unreachable and a demo turn is attempted
- **THEN** the page shows the `unreachable` health status and the structured error contract — no mock-generated prediction appears anywhere in the demo path

### Requirement: Demo usage documentation

The docs SHALL include `docs/chatbot-usage.md`, a marketing-team usage guide covering: starting the stack (local compose command or deployed URL), opening `/demo`, entering the bearer token, example questions, and how to read the prediction payload — aligned with challenge deliverable 2.c.

#### Scenario: Marketing user self-serves

- **WHEN** a marketing user follows `docs/chatbot-usage.md` without reading source code
- **THEN** they complete one end-to-end demo chat turn successfully

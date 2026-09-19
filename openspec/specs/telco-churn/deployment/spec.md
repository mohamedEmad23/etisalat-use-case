# telco-churn/deployment Specification

## Purpose

Run the service where the budget allows: a scale-to-zero serverless GPU as primary with a documented cheap-GPU fallback, plus a faithful local development setup with no local GPU. Must be independently usable to start/stop the demo, to develop locally, and to hand off a documented fallback.

## Requirements

### Requirement: Serverless scale-to-zero serving

The primary deployment SHALL serve the API and the LLM on a scale-to-zero serverless platform with no standing GPU cost when idle. The LLM runtime SHALL be the OpenAI-compatible Ollama (llama.cpp) server running Qwen3-4B-Instruct-2507 (4-bit quantized) inside the same GPU container as the API (single-container pattern), with model weights cached in a platform volume so container restarts do not re-download them; a vLLM-class runtime SHALL remain documented as the variant reserved for concurrent multi-user demos. Deploying to the serverless platform MAY be deferred entirely: without a platform account, the documented local CPU stack running the real Ollama model SHALL remain the legitimate PoC demonstration.

#### Scenario: Idle cost is zero

- **WHEN** no demo window is active and the service is scaled down
- **THEN** no GPU compute is billed for the idle period

#### Scenario: Scale-up on demand

- **WHEN** a request arrives at the scaled-down deployment
- **THEN** the platform scales up, the service becomes reachable within the documented cold-start budget, and the request completes

#### Scenario: Weights restored from volume

- **WHEN** the serverless container starts
- **THEN** the Ollama server loads model weights from the platform volume rather than re-downloading them per cold start

#### Scenario: vLLM variant only on demand

- **WHEN** the demo must serve concurrent multi-user traffic
- **THEN** the documented vLLM variant is deployed behind the identical OpenAI-compatible endpoint with no application-layer changes

#### Scenario: Deferred platform path

- **WHEN** no serverless platform account exists
- **THEN** the documented demo path is the local CPU compose stack running the real Ollama model, at zero platform cost

### Requirement: Demo window management

The deployment SHALL provide a documented demo-window configuration: an idle scaledown window of approximately 1800 s during active demo periods, and a documented cold-start expectation of at most about 3 minutes from scale-up to first successful chat response.

#### Scenario: Demo window active

- **WHEN** the demo-window configuration is applied
- **THEN** consecutive requests within the window complete without cold-start pauses

#### Scenario: Cold start documented and bounded

- **WHEN** a fresh deployment scales up from zero
- **THEN** the documented cold-start procedure reports time-to-first-successful-chat of ≤ about 3 minutes

### Requirement: Local development parity

Local development SHALL run the same pipeline against a locally served open-source LLM (4-bit quantized) launched via a single compose command, with the application layer configuration-switchable between local and serverless backends. The local and serverless demo paths MUST use the real configured LLM — no mock backend in any served runtime. The unit test suite MUST run without any GPU or live LLM, using deterministic test doubles confined to test code.

#### Scenario: One-command local stack

- **WHEN** a developer runs the documented local compose command
- **THEN** the API, the LLM backend, and the churn model are all reachable locally and an end-to-end chat request succeeds on CPU

#### Scenario: GPU-free CI

- **WHEN** the unit test suite runs in CI with no GPU and no live LLM
- **THEN** all tests pass because LLM interactions use deterministic test doubles confined to test suites
- **AND** live-model behavior is verified separately by the gated demo smoke check and user acceptance through the demo page

#### Scenario: Demo path is real

- **WHEN** a chat turn is served by the demo stack (local or serverless)
- **THEN** the answer originates from the actually-running open-source model, and the smoke report records the model identity and turn latency

### Requirement: Documented GPU fallback

The deployment SHALL include a runbook for the documented fallback: a per-hour community-GPU cloud instance (RTX 3090 class) using the stop-pod storage-preserving pattern, with start/stop procedure and cost notes.

#### Scenario: Fallback exercised

- **WHEN** the fallback runbook is followed on the fallback provider
- **THEN** a 3090-class instance is started, the same service deploys to it, and stopping the pod preserves storage for the next start

### Requirement: Container hardening

The container image SHALL contain no secrets (secrets injected via environment at runtime), run as a non-root user, and declare resource limits; image build MUST be reproducible from the repository.

#### Scenario: Image audit

- **WHEN** the built image is inspected
- **THEN** it contains no secrets or credentials, its default user is non-root, and resource limits are declared in the deployment configuration

### Requirement: VRAM headroom budget

The deployed LLM configuration SHALL retain at least 20% VRAM headroom on the target GPU under demo load, and the measured headroom MUST be recorded in the docs.

#### Scenario: Headroom measured on target GPU

- **WHEN** the demo load profile runs on the target GPU
- **THEN** peak VRAM usage leaves ≥20% of device memory free
- **AND** the measured figure is recorded in the deployment docs

### Requirement: Demo-first deployment smoke

Every deployment (serverless or local stack) SHALL be smoke-verified before a demo: `GET /health` reachable plus one real authenticated chat turn through the demo page (or an equivalent authenticated scripted request when recording the artifact), with the smoke report written to `runs/demo_smoke_report.json` recording model identity, backend, turn latency, and schema validity.

#### Scenario: Smoke gate passes

- **WHEN** the deployment smoke run executes against a healthy deployment
- **THEN** `/health` reports reachable and the recorded turn is answered by the configured real model with measured latency

#### Scenario: Smoke gate fails loudly

- **WHEN** the smoke run cannot obtain a real model answer (backend unreachable or invalid response)
- **THEN** the smoke report marks the deployment not demo-ready — the failure is visible, not silently masked by a mock

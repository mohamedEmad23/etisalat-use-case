# deployment — Delta

## Purpose

Run the service where the budget allows: a scale-to-zero serverless GPU as primary with a documented cheap-GPU fallback, plus a faithful local development setup with no local GPU. Must be independently usable to start/stop the demo, to develop locally, and to hand off a documented fallback.

## ADDED Requirements

### Requirement: Serverless scale-to-zero serving

The primary deployment SHALL serve the API and the LLM (vLLM-class runtime, open-source model) on a scale-to-zero serverless platform with no standing GPU cost when idle.

#### Scenario: Idle cost is zero

- **WHEN** no demo window is active and the service is scaled down
- **THEN** no GPU compute is billed for the idle period

#### Scenario: Scale-up on demand

- **WHEN** a request arrives at the scaled-down deployment
- **THEN** the platform scales up, the service becomes reachable within the documented cold-start budget, and the request completes

### Requirement: Demo window management

The deployment SHALL provide a documented demo-window configuration: an idle scaledown window of approximately 1800 s during active demo periods, and a documented cold-start expectation of at most about 3 minutes from scale-up to first successful chat response.

#### Scenario: Demo window active

- **WHEN** the demo-window configuration is applied
- **THEN** consecutive requests within the window complete without cold-start pauses

#### Scenario: Cold start documented and bounded

- **WHEN** a fresh deployment scales up from zero
- **THEN** the documented cold-start procedure reports time-to-first-successful-chat of ≤ about 3 minutes

### Requirement: Local development parity

Local development SHALL run the same pipeline against a locally served open-source LLM (4-bit quantized) launched via a single compose command, with the application layer configuration-switchable between local and serverless backends; unit tests MUST run without any GPU or LLM.

#### Scenario: One-command local stack

- **WHEN** a developer runs the documented local compose command
- **THEN** the API, the LLM backend, and the churn model are all reachable locally and an end-to-end chat request succeeds on CPU

#### Scenario: GPU-free CI

- **WHEN** the unit test suite runs in CI with no GPU and no LLM available
- **THEN** all tests pass because LLM interactions are mocked

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

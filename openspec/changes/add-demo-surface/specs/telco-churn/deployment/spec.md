# deployment — Delta

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Demo-first deployment smoke

Every deployment (serverless or local stack) SHALL be smoke-verified before a demo: `GET /health` reachable plus one real authenticated chat turn through the demo page (or an equivalent authenticated scripted request when recording the artifact), with the smoke report written to `runs/demo_smoke_report.json` recording model identity, backend, turn latency, and schema validity.

#### Scenario: Smoke gate passes

- **WHEN** the deployment smoke run executes against a healthy deployment
- **THEN** `/health` reports reachable and the recorded turn is answered by the configured real model with measured latency

#### Scenario: Smoke gate fails loudly

- **WHEN** the smoke run cannot obtain a real model answer (backend unreachable or invalid response)
- **THEN** the smoke report marks the deployment not demo-ready — the failure is visible, not silently masked by a mock

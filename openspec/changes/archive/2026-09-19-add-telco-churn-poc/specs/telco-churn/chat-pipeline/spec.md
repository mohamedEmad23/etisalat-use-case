# chat-pipeline — Delta

## Purpose

Turn natural-language customer questions into churn predictions without the LLM ever inventing numbers. The LLM acts as a schema-constrained extractor: each user turn produces exactly one grammar-enforced feature request, the pipeline executes prediction with the extracted features plus session context, and every numeric in the reply is interpolated from the model's tool output. Must be independently usable for multi-turn chat sessions over the churn model.

## ADDED Requirements

### Requirement: Extract-only LLM contract

The chat pipeline SHALL treat the LLM as a feature extractor only: per user turn it SHALL request exactly one feature-selection object (target features, optional filters) via a schema that contains no numeric fields, and the pipeline MUST reject any LLM output that does not validate against that schema before it can affect a prediction. The pipeline SHALL enforce schema conformance with deterministic grammar-based decoding (constrained JSON) so that schema-invalid output cannot reach the tool executor.

#### Scenario: Single well-formed request per turn

- **WHEN** a user asks "Will a fiber-optic customer on month-to-month contract churn?"
- **THEN** the LLM emits exactly one schema-valid feature request naming the relevant features
- **AND** the request contains no numeric field values

#### Scenario: Malformed LLM output cannot corrupt state

- **WHEN** the constrained decoder rejects an LLM candidate (schema violation)
- **THEN** no partial or garbage features enter the session or the prediction
- **AND** the pipeline retries with the same grammar instead of degrading to free-form generation

### Requirement: Numerics sourced exclusively from tool output

The pipeline SHALL produce every numeric value shown to the user (churn probability, confidence band, top drivers' weights) by interpolating strings from the prediction tool's output; the LLM-generated segments of a response MUST NOT contain numbers derived by the LLM itself.

#### Scenario: Probability comes from the model artifact

- **WHEN** a prediction completes for a session turn
- **THEN** the probability and its supporting numerics in the reply are taken verbatim from the tool output payload
- **AND** the LLM is not asked to restate or transform numbers in its own generation

### Requirement: Multi-turn session state

The pipeline SHALL maintain in-memory per-session state that accumulates extracted features across turns, allowing the user to refine ("what if contract is two-year instead?") without restating prior facts; state MUST be keyed by a session identifier and scoped to the session lifetime.

#### Scenario: Refinement without restatement

- **WHEN** a first turn extracts Contract=month-to-month and a second turn says "and make tenure 24 months"
- **THEN** the second prediction uses the accumulated feature set from both turns

### Requirement: Out-of-scope refusal

When a user request does not map to churn prediction for a customer profile, the pipeline SHALL refuse politely, state what it can answer, and MUST NOT emit a fabricated prediction or probabilities.

#### Scenario: Off-topic question

- **WHEN** the user asks "What's the weather in Cairo?"
- **THEN** the response states the service only predicts churn from customer profile features and offers an example question

### Requirement: Extraction evaluation suite

The pipeline SHALL ship a curated test suite covering every one of the 19 non-identifier dataset features, multi-intent utterances, and out-of-scope utterances, asserting 100% schema validity of LLM feature requests and at least 95% slot accuracy (correct feature names selected) as the acceptance bar; failures MUST be either fixed or documented before release.

#### Scenario: Suite run on the serving model

- **WHEN** the extraction suite runs against the deployed LLM configuration
- **THEN** every request parses under the schema (100% validity)
- **AND** aggregate slot accuracy is ≥95% with per-case results recorded

### Requirement: Open-source LLM only

Any LLM used by the pipeline SHALL be an open-source-licensed model, runnable without closed-source or third-party LLM APIs; the LLM identity MUST be an external configuration value, not code logic.

#### Scenario: Configuration names a compliant model

- **WHEN** the serving configuration is resolved
- **THEN** the configured model resolves to an open-source-licensed artifact (e.g., Apache-2.0) served locally or on self-hosted infrastructure

### Requirement: Documented model swap path

The pipeline SHALL document a single-configuration change to swap the LLM to a larger open-source alternative (e.g., Qwen3-8B) with no code changes, to be exercised if extraction evaluation fails.

#### Scenario: Swap exercise

- **WHEN** the documented swap procedure is followed with the alternative model identifier
- **THEN** the pipeline runs and the extraction suite re-executes against the new model

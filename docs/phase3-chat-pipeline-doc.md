# Phase 3 — Extract-Only Chat Pipeline: Documentation, Results, and Design Rationale

Branch: `p3/chat-pipeline` — tasks `4.1–4.7` of `openspec/changes/add-telco-churn-poc/tasks.md` (§4, all complete).

## 1. Aim / Goal

Wire a conversational front-end around the Phase 2 classifier without ever giving the LLM the authority to compute anything. The agent scores zero churn numbers; it only converts free text into a schema-validated **FeatureRequest** (target features + optional filters, closed 19-name vocabulary), and everything numeric in the reply is interpolated from tool output (calibrated Churn probability, operating-point threshold, SHAP top-3 contributions). Non-negotiables enforced structurally:

- The LLM never produces numbers or predictions — only a schema-valid FeatureRequest.
- Schema-invalid output is rejected **before** it reaches the tool executor; a malformed candidate gets one same-grammar retry, never a free-form re-ask, and session state is untouched.
- Multi-turn session state is in-memory, keyed by session id, scoped to session lifetime (restart clears it — documented).
- Every non-churn-profile domain (weather, small talk) refuses with no fabricated prediction, and states what it *can* answer.
- PII (emails, phones, customer IDs) is scrubbed before anything is logged or persisted.
- Open-source-licensed LLM only; the model identity is external config, not code (`llm_model = qwen3:4b-instruct-2507-q4_K_M` by default, single-config swap documented below).

## 2. Code written (`src/telco_churn/chat/` + `src/telco_churn/serving/`)

| File | Purpose | Why this shape |
| --- | --- | --- |
| `chat/schemas.py` | `FeatureName` StrEnum (19 members), `FEATURE_NAMES`, `FeatureRequest`, `is_schema_candidate` | Closed enum = the cleaned dataset feature names (data wins naming: `Is_Married`, `Dual` per `docs/data-mismatches.md`). `extra='forbid'` + LAX mode rejects any unknown key (including numeric-injection attempts like `{"tenure": "24"}`); tests prove both rejection paths |
| `chat/prompts.py` | `SCHEMA_HINT`, `EXTRACTION_SYSTEM_PROMPT` (rules + four few-shots), `_RETRY_SYSTEM`, `extraction_messages` / `retry_messages` | System prompt teaches the exact JSON shape and the vocabulary; the retry prompt re-asserts the *same* grammar — malformed candidates get a constrained second attempt, not a free-form rewrite |
| `chat/loop.py` | `TurnReply`, `ChurnChatPipeline.handle`, `_canon_value`, `_baseline_frame`, `_profile_row`, `load_default_artifact` | The ~223-line tool loop: extract → validate (retry once on invalid) → canonicalize filters → update session state → build profile row from a cached baseline → `calibrated_proba` → threshold comparison → SHAP top-3 → compose reply. Numerics appear in text only via the payload dict; refusals set `payload=None` and touch no state |
| `chat/session.py` | `SessionState`, `SessionStore` (`get_or_create` / `clear` / `snapshot`) | Two-turn refinement accumulates canonical features without restatement (`{'Contract': 'Month-to-month', 'tenure': '24'}` observed live). `snapshot` returns a redacted copy |
| `chat/redact.py` | `redact` (email / 7+-digit phone /\`\d{4}[A-Z]{5}\` customer id) | Regex scrubber applied to snapshots/logs: `[EMAIL]`, `[PHONE]`, `[CUSTOMER_ID]` |
| `serving/llm_client.py` | `churn_schema()` (JSON Schema), `LlmClient.extract_json`, `FakeExtractionTransport` | The **single** LLM seam. Grammar-constrained decode via `response_format: json_schema` (vLLM-style; Ollama honors the schema on `openai`-compatible mode). Temperature 0.0. A `_no_numeric_fields` guard re-checks the parsed candidate in-process so the contract holds even on endpoints that ignore `response_format`. `FakeExtractionTransport` lets unit tests drive the loop with deterministic candidates over real HTTP plumbing |

Tests: `tests/unit/test_chat_pipeline.py` — 10 tests: schema contract (unknown feature, 3× numeric injection, non-string filter value, 19-name closed vocabulary) + loop behavior (malformed candidate → refusal with empty snapshot, unparseable value → refusal naming the feature, numerics-composed-from-payload proof, two-turn accumulation, out-of-scope refusal, PII scrub). `tests/eval/test_extraction_suite.py` — opt-in Ollama-gated suite (marker `eval_llm`), 23 cases (19 features + 2 multi-intent + 2 out-of-scope), per-case records, report artifact.

## 3. Results (extraction eval vs local Ollama, `qwen3:4b-instruct-2507-q4_K_M`)

Report: `runs/extraction_suite_report.json` (written by the gated eval suite).

| Metric | Result | Bar |
| --- | --- | --- |
| Test cases | 23 | all 19 features + 2 multi-intent + 2 OOS |
| Schema validity | **100%** (24/24 calls) | 100% required |
| Slot accuracy | **95.83%** (23/24) | ≥ 95% required |

The single documented miss: *"Churn for bank transfer payments?"* did not select `Payment_Method` in its target features — recorded per-case in the report, not hidden. Since accuracy clears the bar, no Qwen3-8B swap was needed. Full suite: **43/43 tests pass**; all 12 pre-commit hooks green.

## 4. Numerics discipline and refusal paths (how the contract is enforced)

1. **Schema gate first.** `extract_json` parses → `FeatureRequest.model_validate` → invalid ⇒ one retry with the same grammar ⇒ still invalid ⇒ refusal turn, state untouched. Invalid candidates never reach the predictor.
2. **No numeric fields exist in the schema.** `FeatureRequest` fields are only `target_features` (enum list), `filters` (enum → string), `out_of_scope` (bool). The in-process `_no_numeric_fields`/`extra='forbid'` checks back up the endpoint-side `json_schema` constraint.
3. **Canonicalization before state.** Every filter value must map (`_canon_value`) to a canonical string the profile row understands (yes/no + `No phone/network service` sentinels, category canon, numeric digits); an unmappable value ⇒ refusal naming that feature — no guesses, no silent coercion.
4. **Numbers only from payload.** The reply template interpolates `churn_probability` (calibrated), the operating-point threshold, and SHAP contributions — all produced by the artifact/train-side code. A unit test strips common punctuation tokens from the reply text and asserts every digit-group is accounted for by the payload values, proving the LLM text path never invents numbers.
5. **Refusals never fabricate.** Out-of-scope utterances yield a refusal that names the fiber-optic/month-to-month example of what IS answerable, `payload=None`. Failed extraction and unmappable values refuse the same way. Nothing in these paths touches session state or the predictor.

## 5. Known issue carried to next step (artifact staleness)

`train.py`'s `main()` currently loads the CSV with `pl.read_csv(settings.data_csv_path)` directly, bypassing `prepare_raw_frame`/`clean`. Artifacts produced by the v1 real-CSV run therefore contain `customerID` and the untrimmed `'Senior_Citizen '` column in `feature_list`. Fix (queued first on Phase 4): route `main()` through `load_churn_csv()` → `clean()`, then re-run `PYTHONPATH=src .venv/bin/python -m telco_churn.model.train --n-trials 20` so `runs/` artifacts match the cleaned schema the chat loop consumes. The chat pipeline's `_baseline_frame` is already built on the cleaned pipeline and is unaffected.

## 6. Cross-phase agreements honored

- Extract-only grammar-constrained LLM (ADR-0002): `json_schema` response_format + pydantic forbid + in-process double-check.
- Domain glossary terms used exactly: Churn probability (model-computed, never LLM), Churner / Not churner labels, FeatureRequest, Extract-only LLM.
- PII redaction for anything persisted; bearer-token and env config untouched (Phase 4 scope).
- Open-source LLM only; model identity external (`TELCO_LLM_MODEL`), documented single-config swap to Qwen3-8B (`TELCO_LLM_MODEL=qwen3:8b-instruct-2507-q4_K_M`) if slot accuracy regresses below 95%.

---
status: accepted
date: 2026-09-16
---

# Extract-only, grammar-constrained LLM; never generates numbers; no fine-tuning

The chat pipeline's correctness hinges on the LLM never hallucinating numbers. The open-source model (Qwen3-4B-Instruct-2507, Ollama tag `qwen3:4b-instruct-2507-q4_K_M`, 4-bit) has exactly one job: emit a single schema-constrained JSON feature request per turn — a grammar with **no numeric fields**, enforced by constrained decoding (vLLM structured outputs in cloud, Ollama JSON-schema mode in dev). All numbers in chat answers are string-interpolated from tool/model output. We do not fine-tune: prompting + few-shot is sufficient for slot accuracy ≥ 95% under grammar constraints, and a fine-tuning loop does not fit the $0–20 / 6–7-day budget. Qwen3-4B's weaker BFCL tool-calling score vs 8B-class models is neutralized by the grammar; the Qwen3-8B swap remains a documented fallback if the extraction suite fails.

## Considered Options

- **Prompted + grammar-constrained extract-only LLM (chosen)** — deterministic extraction, zero numeric hallucination surface, no training cost.
- **Fine-tune the LLM for extraction — rejected**: GPU cost, eval burden, no measured benefit at 7K-row/19-feature scale.
- **LLM free-forms answers containing computed numbers — rejected**: hallucination risk on the one axis the client cares about; violates "LLM never generates numbers."
- **LangChain/agent frameworks — rejected**: spec §6 simplicity constraint; a hand-rolled ~150-line tool loop does the job and stays auditable.

## Consequences

- The feature-request JSON schema is a public contract of the system; changing it changes the extraction eval suite.
- Failure modes are bisected: schema validity is guaranteed by the runtime (violations = bugs), chat quality reduces to slot accuracy (extraction correctness) plus classifier quality.
- If Qwen3-4B-Instruct-2507 (`qwen3:4b-instruct-2507-q4_K_M`) misses ≥ 95% slot accuracy, the fallback is Qwen3-8B under the same grammar (~2× VRAM, still fits T4/12 GB), not prompt surgery.

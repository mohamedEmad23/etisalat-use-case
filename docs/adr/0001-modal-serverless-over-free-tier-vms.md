---
status: accepted
date: 2026-09-16
---

# Modal serverless over free-tier VMs for LLM serving

The challenge requires serving an open-source LLM (Qwen3-4B-Instruct-2507, tag `qwen3:4b-instruct-2507-q4_K_M`) on a GPU we do not own, with a hard budget of $0–20 and no local GPU. We deploy the API + vLLM on **Modal Starter** (scale-to-zero, $30/month free credits ≈ 50 T4-hours, $0 platform fee) as primary, with a documented **RunPod Community RTX 3090** stopped-pod fallback (~$0.22/hr). We deliberately rejected Oracle's Always-Free ARM VM, Google Colab as a host, and Hugging Face ZeroGPU. No Kubernetes anywhere: one FastAPI container, docker-compose locally — orchestration is ceremony for a single service.

## Considered Options

- **Modal Starter (primary)** — the only $0 option with real engineering ergonomics: per-second billing, scale-to-zero web endpoints, containerized vLLM.
- **RunPod Community 3090 (fallback)** — cheapest reliable persistent GPU; "stop pod + network volume" pattern keeps a 7-day demo at ≈ $2–3.
- **Oracle A1 free tier — rejected**: post-June-2026 limits halved to 2 OCPU/12 GB, "out of capacity" near-universal for new accounts, CPU-only 4–8 tok/s misses the p95 ≤ 5 s budget, vLLM on ARM64 is a source-build dead end.
- **Colab free/Pro — rejected**: ToS bans serving web endpoints from notebooks (gray/red zone, termination without warning), no ingress without tunneling, ~12-h sessions, free TPU unusable by any serving stack.
- **HF ZeroGPU Spaces — rejected**: per-call GPU quotas (2/5/40 min), no persistent OpenAI-compatible server possible.
- **GCP $300/90-day trial — backup only**: GPUs unlock only after upgrading trial billing to paid.

## Consequences

- Two documented deployment paths (Modal app + docker-compose for RunPod/local); the LLM sits behind an OpenAI-compatible endpoint so the runtime is swappable without app changes.
- Cost stays $0 unless demo time exceeds ~50 T4-h/month, then ≈ $0.22/hr on RunPod.
- Deployment-time re-verification is mandatory: Modal credit terms, RunPod rates, and model cards move (all checked Sep 2026; flagged stale).

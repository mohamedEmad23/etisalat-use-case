# RunPod 3090 stop-pod fallback runbook

The documented fallback when Modal is unavailable or the demo needs a dedicated
per-hour GPU: a RunPod Community RTX 3090 instance running the same container,
using the **stop-pod storage-preserving pattern** so weights and artifacts survive
between demo sessions. Cost target ≈ $0.22/h while running; $0 when the pod is
stopped (the storage bill is the only residue).

## Why this fallback

- Same OpenAI-compatible runtime (`Ollama`, 4-bit Qwen3-4B-Instruct-2507) as the
  primary path — only `TELCO_LLM_BASE_URL` changes, no code change.
- Stop (not terminate) keeps the pod's disk: the pulled model and the churn
  artifacts persist, so restarts are minutes, not downloads.
- Community 3090 (24 GB VRAM) leaves ample headroom for the 4-bit 4B checkpoint
  (well above the ≥20% VRAM-headroom requirement).

## Start (preserves storage)

1. Console → **Pods** → *Start* on the existing stopped pod (create it once:
   **Deploy → Community Cloud → RTX 3090 → pod**, attach a Volume named
   `ollama-models` mounted at `/root/.ollama`, and a second volume
   `telco-churn-runs` mounted at `/srv/app/runs`).
2. Upload the repo (or `git clone`) to `/srv/app` on the pod.
3. Install runtime deps and pull the model (first start only)::

   curl -fsSL https://ollama.com/install.sh | sh
   cd /srv/app && python -m pip install uv && uv sync --frozen --no-dev
   ollama serve &          # daemon on 127.0.0.1:11434
   ollama pull qwen3:4b-instruct-2507-q4_K_M

4. Serve the API (the token MUST come from the environment, never the image)::

   export TELCO_API_BEARER_TOKEN='<token set by the operator>'
   export TELCO_LLM_BASE_URL='http://127.0.0.1:11434/v1'
   export TELCO_LLM_BACKEND=OLLAMA
   /srv/app/.venv/bin/python -m uvicorn telco_churn.api.app:create_app \
       --factory --host 0.0.0.0 --port 8000

5. Expose it: RunPod pod → **HTTP Service** on port 8000 (or an SSH tunnel for
   private demos). Verify with `curl <url>/health` then one authenticated
   `POST /chat` turn through `/demo` or curl.

## Stop (preserves storage)

1. **Stop** (do **not** terminate — terminate destroys the disk):
   Console → Pods → *Stop*.
2. Storage (both volumes + pod disk) persists; running cost drops to $0.

## Cost notes

| State | Cost |
| --- | --- |
| Pod running (3090 community) | ≈ $0.22/h |
| Pod stopped | $0 compute; persistent storage only |
| Weights re-download | avoided (volumes persist) |

## Verification record

- Runbook reviewed against the deployment spec ('Documented GPU fallback'):
  start/stop preserve storage — reviewed 2026-09-18.
- Live exercise on RunPod is deferred with the same user decision as the Modal
  staged deploy (no account created); the procedure above is the audit path.

# Operations log — telco-churn

Deployment-relevant operational events, in order. (The challenge suggested a wiki log; this
repo records the same content as a versioned docs file. No separate wiki was created.)

## Phase 5 — deployment (2026-09-18)

### Adopted deployment runtime

ADR-0001 amended: the served LLM runtime is **Ollama (llama.cpp) + Qwen3-4B-Instruct-2507 4-bit**
in the same scale-to-zero GPU container as the API (single-container pattern), weights cached in
a platform volume (`ollama-models`), artifacts volume (`telco-churn-runs`), bearer token from a
platform secret. vLLM-class runtime remains a documented variant for concurrent multi-user use
only. Deferral clause: with no platform account there is no cost; the local compose stack is a
legitimate PoC demonstration path.

### Artifacts and verification records

| Item | Status | Record |
| --- | --- | --- |
| `deploy/modal_app.py` (single GPU function, scaledown_window 1800 s, readiness wait, pull-if-missing) | written, `ast.parse` clean, ruff + mypy clean — **ready, not exercised** (needs a platform account with payment method) | staged commands in file docstring (`modal secret create telco-api-bearer`, `modal deploy deploy/modal_app.py`) |
| `deploy/Dockerfile` base pin | `python@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285` | digest recorded from `docker image inspect` 2026-09-18 |
| `deploy/Dockerfile` build | **built successfully** once (`telco-churn-api:phase5`, all layers green) then rebuild attempts hit two daemon-side failures (see incidents) | build logs in session record |
| `deploy/Dockerfile` CMD factory fix | verified by reasoning + image built; live `/health` smoke **pending daemon restart** (see incident 2) | — |
| `deploy/docker-compose.yml` | `docker compose config --quiet` → OK; resource limits added; artifact mount corrected to `/srv/app/runs` | — |
| `deploy/runpod-fallback.md` stop-pod runbook | reviewed 2026-09-18; live exercise deferred (same user decision as Modal) | runbook file |
| `docs/perf-report.md` | CPU baseline measured (p50 e2e ≈ 7.3 s); GPU budgets procedure + deferral recorded | perf report |
| `docs/architecture.md` + `README.md` | written (deliverable 3 + usage map) | — |

### Container defects found and fixed during build exercise

1. **CMD used the wrong import target** — image CMD pointed at a module-level `app` that does not
   exist (the API is a factory). Fixed: `telco_churn.api.app:create_app --factory`.
2. **Package not importable in image** — the venv contains dependencies only (`uv sync --no-dev`);
   `telco_churn` resolves via `PYTHONPATH=src` locally, but the container had no such path, so
   uvicorn raised `ModuleNotFoundError: No module named 'telco_churn'` (caught by the first live
   smoke). Fixed: `ENV PYTHONPATH=/srv/app/src` in the Dockerfile, same key in the Modal image env.
3. **Slow-link resilience** — PyPI fetches timed out inside the Docker VM during builds; added
   `UV_HTTP_TIMEOUT=120`, `UV_CONCURRENT_DOWNLOADS=8`, and a uv cache mount
   (`RUN --mount=type=cache,target=/root/.cache/uv`) so failed layers accumulate downloads
   instead of restarting from zero.

### Incident: Docker Desktop containerd content store corruption (2026-09-18)

`docker system df` and repeated builds began failing with
`open /var/lib/desktop-containerd/daemon/io.containerd.content.v1.content/blobs/sha256/df7bca…: input/output error`
(blob missing/corrupt) and `io.containerd.metadata.v1.bolt/meta.db: input/output error`.
Host disk pressure: `/` at 96% (21 GB free). Remediation on record: restart Docker Desktop;
if corruption persists, Troubleshoot → Clean/Purge data (images re-pull; no state of ours is
lost). The post-fix container smoke (build → `/health` with artifact mount) is re-run after the
daemon restart.

### Container hardening pass (2026-09-18, post-build research)

Cross-referenced the Dockerfile/compose against current official Docker, uv, and Ollama
guidance (research report P0–P11, citations on file); applied the full patch set:

- **Active ignore file** — `deploy/.dockerignore` was never read (Docker looks for
  `<Dockerfile-name>.dockerignore` next to the Dockerfile when the context root is the repo
  root): renamed to `deploy/Dockerfile.dockerignore` + added missing entries. The context no
  longer ships `.git`, `.venv`, `data/`, `runs/`, caches.
- **Pinned toolchain** — `COPY --from=ghcr.io/astral-sh/uv@sha256:440fd647…` (uv 0.11.16, the
  lockfile producer, digest-pinned; removes the PyPI round-trip for installing uv itself) and
  `ollama/ollama:0.34.2@sha256:da6e0dc5…` (version + digest pinned; `:latest` has a history of
  lagging releases).
- **uv sync flags** — `--locked --no-dev --no-install-project` (asserts lock freshness; the
  project has no build backend — `PYTHONPATH=/srv/app/src` stays the contract) + `UV_LINK_MODE=copy`,
  `UV_COMPILE_BYTECODE=1` per the official Docker guide.
- **Resource limits mirrored at service level** (`cpus`/`mem_limit` identical to the `deploy`
  blocks) per the compose-spec consistency requirement — enforced on `up` by Compose v2,
  version-proof for older tooling.
- **One-shot `model-init` service** — `ollama pull` after the Ollama service is healthy
  (`restart: "no"`); the API now `depends_on: service_completed_successfully`, so
  `up --build` alone produces a working `/chat` (previously the model store came up empty).
- **Lifecycle hardening** — `init: true` + `restart: unless-stopped` on both long-lived
  services; `stop_grace_period: 60s` on the API (10 s default would SIGKILL ~50 s cold LLM
  turns on `down`), 30 s on Ollama.
- **API healthcheck without curl** — stdlib `urllib` `/health` probe (`start_period: 60 s`
  for the baseline-profile build at startup, `start_interval: 5 s`).
- **`read_only: true` + `tmpfs: /tmp`** on the API (immutable root filesystem; rollback note
  in the compose comments).

Defects #4–#6 found and fixed during this pass:

4. **Missing `data/` mount** — the chat baseline profile reads the challenge CSV at startup
   (`settings.data_csv_path`), but compose mounted only `runs/` (and the Modal image excluded
   `data/` entirely). Fixed: `../data:/srv/app/data:ro` in compose; `!data/WA_Fn-UseC_…csv`
   re-inclusion in the Modal image ignore list.
5. **Uppercase enum value in the Modal image env** — `TELCO_LLM_BACKEND=OLLAMA` would fail
   pydantic validation at boot (enum values are lowercase `vllm`/`ollama`/`mock`; verified
   empirically). Fixed: `"ollama"`. The compose value was already lowercase and valid.
6. **Dead ignore file** (see above) — the image build context was shipping repo state
   needlessly; now corrected.

Verification of this pass: `docker compose config --quiet` → OK (client-side); `ast.parse` on
`modal_app.py` clean; pre-commit + pytest green (no `src/` changes; suite unchanged).
Live build + smoke re-run pending the user's Docker Desktop restart (see incident above).

## Open operational items

- GPU latency + VRAM-headroom measurement: deferred (explicit user decision) until a GPU runtime
  is exercised; harness and budget table ready in `docs/perf-report.md`.
- Modal account/secret creation and staged deploy: user's choice; `modal_app.py` is ready-not-exercised.
- RunPod 3090 live exercise: deferred (runbook reviewed).

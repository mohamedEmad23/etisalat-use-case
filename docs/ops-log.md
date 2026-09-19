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

7. **uv connect-timeout vs the CDN (root cause of the recurring PyPI flake)** — build logs showed
   "3 retries in 45.8s" (~15s/attempt) despite `UV_HTTP_TIMEOUT=120`; per uv's environment
   reference the failing knob was `UV_HTTP_CONNECT_TIMEOUT` (default **10 s**) — a probe to
   files.pythonhosted.org from this host measured 11.65 s, so default connects fail by a hair.
   Fixed in the Dockerfile ENV (`UV_HTTP_CONNECT_TIMEOUT=60`, `UV_HTTP_TIMEOUT=300`,
   `UV_HTTP_RETRIES=10`, `UV_CONCURRENT_DOWNLOADS` 8→4) plus a 5-attempt outer retry loop;
   the cache mount keeps every completed wheel so retries only cost missing wheels.

Verification of this pass: `docker compose config --quiet` → OK (client-side); `ast.parse` on
`modal_app.py` clean; pre-commit + pytest green (no `src/` changes; suite unchanged).
**Live build + smoke re-run: PASSED 2026-09-18** (after the user's Docker Desktop restart) —
`docker build` → `uv sync` completed on attempt 1/5 in 210.7 s; container smoke →
`GET /health` = `{"service":"ok","llm":"unreachable"}` (no Ollama in the smoke container —
correct); `docker compose config --quiet` → OK. Full compose stack (with model pull) is the
remaining user-run step.

8. **Host-port collision on 11434** — the compose stack published `11434:11434`, but a
   host-native Ollama already owns `127.0.0.1:11434`; `docker compose up` could never start
   the LLM service ("port is already allocated"). Fixed by never publishing the LLM port:
   the API reaches a bundled Ollama over the compose network and a host Ollama through
   `host.docker.internal:11434`; the port is documented with `expose` for inspection only.
9. **Compose interpolation cannot see the repo-root `.env`** — the compose project directory
   is `deploy/`, so `${TELCO_API_BEARER_TOKEN}` interpolated to a blank string and the API
   crashed at startup (min length 8). Fixed by reading repo-root `.env` via `env_file`
   (a token under `environment` would override `env_file` — it must not live there), plus a
   committed `.env.example`. Verified with `docker compose config -q` and a live auth probe
   (401 without a token, 200 with).
10. **The extraction grammar forbade the values it needed** — the LLM `filters` schema
    restricted every filter VALUE to the feature-name enum, so grammar-constrained decoding
    could only emit nonsense (`{"Internet_Service": "Dual"}`) and every chat turn ended in
    "could not read a valid value". Fixed in five layers: free-string filter values;
    whole-phrase folding in the canonicaliser (`Fiber-optic` → `Fiber optic`); per-feature
    value menus for categoricals (dataset literals only) plus a digits-only pattern for
    numeric slots; one bounded menu-echoing repair retry; and numeric grounding — numbers
    absent from the user's own message are dropped rather than silently fabricated.
    Regression evidence: `runs/extraction_suite_report.json` regenerated on the hardened
    pipeline — 23/23 cases, schema validity 1.0, slot accuracy 1.0.

**Incident (2026-09-19): wedged image store during the first bundled-LLM attempt.** Pulling
the ~6 GB `ollama/ollama` image while the host disk sat at 97% left layer extraction crawling
at byte level (zero `Pull complete`; `docker images` hanging). A `docker desktop restart`
cleared it; the user then wiped images/volumes/build-cache (~6 GB reclaimed) before the
successful run. Lesson: keep ≥15 GB host headroom before multi-GB pulls — the default
compose path no longer needs the ollama image at all.

**Verification (2026-09-19, `p5/deployment`):** image build ✓ (cold uv cache);
`docker compose -f deploy/docker-compose.yml up -d` → api healthy on `127.0.0.1:8000`;
`/health` = `{"service":"ok","llm":"reachable"}`; four authenticated `/chat` probes matched
the expected deterministic outputs (0.4094 / 0.1362 / 0.2877 / 0.4094 with the expected
`Profile used:` lines); `pytest tests/unit -q` → 93 passed, zero warnings; extraction suite
23/23; `pre-commit` all hooks green; `openspec validate --changes` green. The
`add-telco-churn-poc` change was synced to `openspec/specs/` (five capability specs) and
archived (see `openspec/changes/archive/2026-09-19-add-telco-churn-poc/`).

**Task 7.2 — scenario → evidence traceability:** `data-pipeline` →
`tests/unit/test_data_pipeline.py` + `docs/data-mismatches.md`; `churn-model` →
`tests/unit/test_churn_model.py` + `runs/train_report.json` + `docs/perf-report.md`;
`chat-pipeline` → `tests/unit/test_chat_pipeline.py` + `tests/unit/test_llm_client.py` +
`tests/eval/test_extraction_suite.py` + `runs/extraction_suite_report.json`; `chat-api` →
`tests/unit/test_chat_api.py` + `docs/security-checklist.md` + `runs/perf_report.json`;
`deployment` → `deploy/` + `README.md` + `docs/architecture.md` + `docs/runpod-fallback.md` +
`docs/adr/0001-modal-serverless-over-free-tier-vms.md` + this log.

## Open operational items

- GPU latency + VRAM-headroom measurement: deferred (explicit user decision) until a GPU runtime
  is exercised; harness and budget table ready in `docs/perf-report.md`.
- Modal account/secret creation and staged deploy: user's choice; `modal_app.py` is ready-not-exercised.
- RunPod 3090 live exercise: deferred (runbook reviewed).

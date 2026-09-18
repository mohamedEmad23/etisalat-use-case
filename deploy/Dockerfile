# CPU demo container for the telco-churn API.
# Non-root user, uv-managed deps, uvicorn serving :8000.
# Reproducibility: base image pinned by manifest digest (recorded 2026-09-18 —
#   docker image inspect python:3.13-slim --format '{{index .RepoDigests 0}}');
#   the uv binary is pinned by digest from the official distroless image
#   (ghcr.io/astral-sh/uv:0.11.16 — the same uv version that produced the
#   lockfile; digest resolved via the ghcr.io registry API 2026-09-18).
#   Re-resolve both when bumping.
FROM python@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

# Official uv-in-Docker pattern: copy the static binary from the distroless
# image instead of a PyPI round-trip — one fewer failure surface on slow links
# and a reproducible uv version.
COPY --from=ghcr.io/astral-sh/uv@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_CONCURRENT_DOWNLOADS=8 \
    PYTHONPATH=/srv/app/src

RUN useradd --create-home --shell /bin/bash telco
WORKDIR /srv/app

COPY --chown=telco:telco pyproject.toml uv.lock README.md ./
COPY --chown=telco:telco src/ ./src/
# uv cache mount makes dependency downloads survive failed layers — retries on a
# flaky PyPI link accumulate instead of restarting from zero.
# --locked asserts the lockfile is fresh for this pyproject; --no-install-project
# matches the no-build-backend src layout (the app runs via PYTHONPATH, not an
# installed package) and skips an unnecessary build step.
# UV_COMPILE_BYTECODE=1 precompiles the venv at build time (faster cold starts).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

USER telco
EXPOSE 8000

# Artifacts are mounted at /srv/app/runs and the training CSV at /srv/app/data
# (see compose). TELCO_API_BEARER_TOKEN must be set by the caller; the app fails
# fast at startup without it.
CMD [".venv/bin/python", "-m", "uvicorn", \
     "telco_churn.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

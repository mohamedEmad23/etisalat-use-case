# CPU demo container for the telco-churn API.
# Non-root user, uv-managed deps, uvicorn serving :8000.
# Reproducibility: base image pinned by manifest digest (recorded 2026-09-18 —
#   docker image inspect python:3.13-slim --format '{{index .RepoDigests 0}}');
# re-resolve when bumping the base tag.
FROM python@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_CONCURRENT_DOWNLOADS=8 \
    PYTHONPATH=/srv/app/src

RUN useradd --create-home --shell /bin/bash telco
WORKDIR /srv/app

COPY --chown=telco:telco pyproject.toml uv.lock README.md ./
COPY --chown=telco:telco src/ ./src/
# uv cache mount makes dependency downloads survive failed layers — retries on a
# flaky PyPI link accumulate instead of restarting from zero.
RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --no-cache-dir --upgrade pip uv \
    && uv sync --frozen --no-dev

USER telco
EXPOSE 8000

# Artifacts are mounted at /srv/app/runs (see compose). TELCO_API_BEARER_TOKEN
# must be set by the caller; the app fails fast at startup without it.
CMD [".venv/bin/python", "-m", "uvicorn", \
     "telco_churn.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

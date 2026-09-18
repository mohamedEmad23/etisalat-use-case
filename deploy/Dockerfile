# CPU demo container for the telco-churn API.
# Non-root user, uv-managed deps, uvicorn serving :8000.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash telco
WORKDIR /srv/app

COPY --chown=telco:telco pyproject.toml uv.lock README.md ./
COPY --chown=telco:telco src/ ./src/
RUN python -m pip install --no-cache-dir --upgrade pip uv \
    && uv sync --frozen --no-dev

USER telco
EXPOSE 8000

# Artifacts are mounted at /srv/app/runs (see compose). TELCO_API_BEARER_TOKEN
# must be set by the caller; the app fails fast at startup without it.
CMD [".venv/bin/python", "-m", "uvicorn", \
     "telco_churn.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

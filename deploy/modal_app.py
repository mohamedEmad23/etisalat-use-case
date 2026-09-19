"""Modal deployment: ONE GPU container running Ollama (llama.cpp) + the FastAPI API.

Adopted runtime (ADR-0001 amendment): a llama.cpp serving daemon (Ollama) with the
4-bit Qwen3-4B-Instruct-2507 checkpoint runs in the SAME scale-to-zero GPU container
as the API; model weights persist in a platform volume so cold starts never
re-download them. The vLLM runtime is the documented VARIANT for concurrent
multi-user serving (see docs/adr/0001-modal-serverless-over-free-tier-vms.md and
docs/runpod-fallback.md) — it stays a documented switch, not the default path.

READY, NOT EXERCISED: this module is reviewed and parsed but not deployed — a GPU
function requires a Modal account with a valid payment method on file, and the
account creation is deliberately deferred (no account = no card = no cost; the
local compose stack is the legitimate PoC demonstration).

Staged deploy (when the user opts in)::

    modal secret create telco-api-bearer TELCO_API_BEARER_TOKEN=<value>
    modal deploy deploy/modal_app.py

Local syntax check::

    .venv/bin/python -c "import ast; ast.parse(open('deploy/modal_app.py').read())"
"""

import os
import subprocess  # nosec B404 - launches the in-container LLM daemon and API server only (fixed argv, no user input)
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

MODEL = os.environ.get("TELCO_LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
GPU = os.environ.get("TELCO_MODAL_GPU", "T4")  # L4 via TELCO_MODAL_GPU=L4
OLLAMA_PORT = 11434
API_PORT = 8000
SCALEDOWN_WINDOW = 1800  # ~Modal's 20-minute ceiling; demo-window contract

ROOT = Path(__file__).resolve().parents[1]

app = modal.App("telco-churn")

# dockerignore-style exclusion list: keep the image to what the API needs.
_LOCAL_IGNORE = [
    ".git",
    ".venv",
    ".opencode",
    ".vscode",
    ".claude",
    "data",
    # '!' negation (dockerignore last-match-wins): re-include the single
    # challenge CSV the chat baseline profile reads at startup.
    "!data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "docs",
    "openspec",
    "tests",
    "mlflow.db",
    "mlruns",
    "*.joblib",
    "runs",  # artifacts ride the /srv/app/runs volume instead
]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("curl")
    .run_commands(
        # Official Ollama installer; lands `ollama` on PATH inside the container.
        "curl -fsSL https://ollama.com/install.sh | sh"
    )
    .pip_install("uv")
    .add_local_dir(
        str(ROOT),
        "/srv/app",
        copy=True,
        ignore=_LOCAL_IGNORE,
    )
    .run_commands("uv sync --locked --no-dev --no-install-project")
    .env(
        {
            "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}",
            "TELCO_LLM_BASE_URL": f"http://127.0.0.1:{OLLAMA_PORT}/v1",
            "TELCO_LLM_BACKEND": "ollama",
            "TELCO_API_HOST": "0.0.0.0",  # nosec B104 - bind all interfaces for serverless serving
            "PYTHONPATH": "/srv/app/src",
        }
    )
)

_weights = modal.Volume.from_name("ollama-models", create_if_missing=True)
_artifacts = modal.Volume.from_name("telco-churn-runs", create_if_missing=True)


def _wait_for_ollama(timeout_s: float = 120.0) -> None:
    """Poll the Ollama native API until the daemon answers."""
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/tags"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # nosec B310 - fixed localhost http endpoint, no user-supplied scheme
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1.0)
    raise RuntimeError(f"ollama serve did not become ready within {timeout_s:.0f}s")


def _pull_if_missing() -> None:
    """Pull the configured model only when it is absent from the weights volume."""
    listing = subprocess.run(  # nosec B603 B607 - fixed argv, no user input
        ["ollama", "ls"], capture_output=True, text=True, check=False
    ).stdout
    if MODEL in listing:
        return
    subprocess.run(["ollama", "pull", MODEL], check=True)  # nosec B603 B607 - fixed argv, no user input


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        "/root/.ollama": _weights,  # model weights persist across cold starts
        "/srv/app/runs": _artifacts,  # churn artifacts + reports
    },
    secrets=[modal.Secret.from_name("telco-api-bearer")],
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=900,  # bounded startup: first cold start pulls the model
)
@modal.web_server(port=API_PORT, startup_timeout=15 * 60)
def serve() -> None:
    # 1. LLM daemon inside the same container (no second container, no network hop).
    ollama_proc = subprocess.Popen(["ollama", "serve"])  # nosec B603 B607 - fixed argv, no user input
    # 2. Readiness wait, then weights are pulled only if the volume is empty.
    _wait_for_ollama()
    _pull_if_missing()
    # 3. API process bound to 0.0.0.0 — Modal proxies the web_server port.
    try:
        subprocess.run(  # nosec B603 B607 - fixed argv, no user input
            [
                "/srv/app/.venv/bin/python",
                "-m",
                "uvicorn",
                "telco_churn.api.app:create_app",
                "--factory",
                "--host",
                "0.0.0.0",  # nosec B104 - bind all interfaces for serverless serving
                "--port",
                str(API_PORT),
            ],
            check=True,
        )
    finally:
        ollama_proc.terminate()

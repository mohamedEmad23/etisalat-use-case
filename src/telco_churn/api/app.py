"""FastAPI wiring: POST /chat, GET /health, GET /demo, bearer auth, errors."""

from __future__ import annotations

import hmac
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from telco_churn.api.errors import install_handlers, unauthorized
from telco_churn.api.rate_limit import RateLimiter, RateLimitMiddleware
from telco_churn.chat.loop import ChurnChatPipeline, TurnReply, load_default_artifact
from telco_churn.chat.redact import redact
from telco_churn.chat.session import SessionStore
from telco_churn.config import get_settings
from telco_churn.model.artifact import ChurnArtifact
from telco_churn.serving.llm_client import LlmClient

logger = logging.getLogger(__name__)
_REDACTED_LOG = logging.getLogger("telco_churn.api.requests")
_DEMO_HTML = Path(__file__).resolve().parent / "static" / "demo.html"


class ChatIn(BaseModel):
    """Request body of POST /chat."""

    message: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class TurnOut(BaseModel):
    """Response of POST /chat (numerics come from the tool payload)."""

    session_id: str
    text: str
    payload: dict[str, object] | None


def require_auth(
    authorization: str | None = Header(default=None),
) -> None:
    """Bearer dependency; 401 fires before an LLM call is ever queued."""
    token = get_settings().api_bearer_token
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise unauthorized()
    supplied = authorization[len(prefix) :]
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise unauthorized()


def _log_turn(session_id: str, message: str) -> None:
    """Persist the request with PII scrubbed before it is ever stored."""
    _REDACTED_LOG.info("session=%s message=%s", session_id, redact(message))


def _to_out(session_id: str, reply: TurnReply) -> TurnOut:
    return TurnOut(session_id=session_id, text=reply.text, payload=reply.payload)


def create_app(
    *,
    artifact: ChurnArtifact | None = None,
    llm: LlmClient | None = None,
    store: SessionStore | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Build the API app; tests inject doubles for artifact and LLM."""
    settings = get_settings()
    resolved_artifact = artifact if artifact is not None else load_default_artifact()
    resolved_llm = llm or LlmClient()
    pipeline = ChurnChatPipeline(resolved_artifact, resolved_llm, store=store)
    rate_limiter = limiter or RateLimiter(
        max_calls=settings.rate_limit_calls,
        window_seconds=float(settings.rate_limit_window_seconds),
    )

    app = FastAPI(title="telco-churn", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.pipeline = pipeline
    app.state.limiter = rate_limiter
    install_handlers(app)
    app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)

    @app.post("/chat", response_model=TurnOut, dependencies=[Depends(require_auth)])
    async def chat(item: ChatIn) -> TurnOut:
        _log_turn(item.session_id, item.message)
        return _to_out(item.session_id, pipeline.handle(item.session_id, item.message))

    @app.get("/demo", include_in_schema=False)
    async def demo() -> FileResponse:
        """Self-contained demo page — static shell, no embedded data or secrets."""
        return FileResponse(
            _DEMO_HTML,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness + LLM reachability + configured model identity; no auth."""
        reachable = resolved_llm.reachable()
        return {
            "service": "ok",
            "llm": "reachable" if reachable else "unreachable",
            "llm_model": settings.llm_model,
            "llm_backend": settings.llm_backend.value,
        }

    return app

"""In-memory multi-turn session state (spec: sessions are in-process; a
restart clears them — documented single-instance behaviour)."""

from __future__ import annotations

from dataclasses import dataclass, field

from telco_churn.chat.redact import redact


@dataclass(slots=True)
class SessionState:
    """Accumulated feature profile for one chat session.

    ``features`` maps cleaned dataset column names to the user's own words
    for equality filters; refinement turns update entries in place so
    restatement is never required.
    """

    session_id: str
    features: dict[str, str] = field(default_factory=dict)


class SessionStore:
    """dict[session_id, SessionState] with session-lifetime scoping."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        existing = self._sessions.get(session_id)
        if existing is None:
            existing = SessionState(session_id=session_id)
            self._sessions[session_id] = existing
        return existing

    def clear(self) -> None:
        """Explicit clearing (used by tests and admin reset)."""
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)

    def snapshot(self, session_id: str) -> dict[str, str]:
        """Redacted copy of accumulated features — safe for logs."""
        state = self.get_or_create(session_id)
        return {
            redact(k) if isinstance(k, str) else k: redact(v)
            for k, v in state.features.items()
        }

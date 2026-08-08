"""
Append-only audit trail helper.

Provides convenience functions to create AuditLogEntry objects with
ISO-8601 timestamps for both LLM and code actions.
"""

from datetime import datetime, timezone
from typing import Any

from audita.core.schemas import AuditLogEntry


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def log_llm_action(
    stage: str,
    action: str,
    detail: dict[str, Any] | None = None,
) -> AuditLogEntry:
    """Create an audit entry for an LLM-driven action."""
    return AuditLogEntry(
        timestamp=_now_iso(),
        stage=stage,
        actor="llm",
        action=action,
        detail=detail or {},
    )


def log_code_action(
    stage: str,
    action: str,
    detail: dict[str, Any] | None = None,
) -> AuditLogEntry:
    """Create an audit entry for a deterministic code action."""
    return AuditLogEntry(
        timestamp=_now_iso(),
        stage=stage,
        actor="code",
        action=action,
        detail=detail or {},
    )

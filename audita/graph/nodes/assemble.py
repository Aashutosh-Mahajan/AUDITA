"""
Assemble node — pure code, no LLM.

Reads the final PipelineState and organises it into the structure that
``ui/sections.py`` needs for rendering: grouped charts, cleaning diff
table, and audit log.
"""

from typing import Any

from audita.core.audit_log import log_code_action
from audita.core.schemas import (
    AuditLogEntry,
    ChartResult,
    CleaningDiffEntry,
    VerificationStatus,
)


def _group_charts_by_category(
    charts: list[ChartResult],
) -> dict[str, list[dict[str, Any]]]:
    """Group charts by their ``category`` field for the UI."""
    groups: dict[str, list[dict[str, Any]]] = {}

    for chart in charts:
        category = chart.intent.category
        if category not in groups:
            groups[category] = []

        groups[category].append(
            {
                "chart_type": chart.intent.chart_type.value,
                "columns": chart.intent.columns,
                "rationale": chart.intent.rationale,
                "priority_score": chart.intent.priority_score,
                "figure_json": chart.figure_json,
                "verification_status": chart.verification_status.value,
                "verification_notes": chart.verification_notes,
                "error": chart.error,
            }
        )

    return groups


def _build_cleaning_diff_table(
    diffs: list[CleaningDiffEntry],
) -> list[dict[str, Any]]:
    """Build a flat table of cleaning diffs for the UI."""
    return [
        {
            "column": d.column,
            "action": d.action_type.value,
            "rows_affected": d.rows_affected,
            "before": d.before_stat,
            "after": d.after_stat,
        }
        for d in diffs
    ]


def _build_audit_log_table(
    entries: list[AuditLogEntry],
) -> list[dict[str, Any]]:
    """Build a flat table of audit log entries for the UI."""
    return [
        {
            "timestamp": e.timestamp,
            "stage": e.stage,
            "actor": e.actor,
            "action": e.action,
            "detail": e.detail,
        }
        for e in entries
    ]


def assemble(state: dict) -> dict:
    """LangGraph node: assemble the final dashboard data from pipeline state.

    Produces a ``dashboard`` dict with:
    - ``charts_by_category``: charts grouped by category
    - ``cleaning_diff_table``: flat list of cleaning diffs
    - ``audit_log_table``: flat list of audit entries
    - ``quality_summary``: before/after quality profiles
    - ``stats``: summary statistics
    """
    completed_charts: list[ChartResult] = state.get("completed_charts", [])
    cleaning_diff: list[CleaningDiffEntry] = state.get("cleaning_diff", [])
    audit_log: list[AuditLogEntry] = state.get("audit_log", [])
    raw_profile: dict = state.get("raw_profile", {})
    quality_audit: dict = state.get("quality_audit", {})
    clean_profile: dict = state.get("clean_profile", {})

    # Group charts
    charts_by_category = _group_charts_by_category(completed_charts)

    # Cleaning diff table
    cleaning_diff_table = _build_cleaning_diff_table(cleaning_diff)

    # Audit log table
    audit_log_table = _build_audit_log_table(audit_log)

    # Summary stats
    verified_count = sum(
        1
        for c in completed_charts
        if c.verification_status == VerificationStatus.VERIFIED
    )
    flagged_count = sum(
        1
        for c in completed_charts
        if c.verification_status == VerificationStatus.FLAGGED
    )
    failed_count = sum(
        1
        for c in completed_charts
        if c.verification_status == VerificationStatus.FAILED
    )

    dashboard = {
        "charts_by_category": charts_by_category,
        "cleaning_diff_table": cleaning_diff_table,
        "audit_log_table": audit_log_table,
        "quality_summary": {
            "before": quality_audit,
            "after": clean_profile,
        },
        "raw_profile": raw_profile,
        "stats": {
            "total_charts": len(completed_charts),
            "verified_charts": verified_count,
            "flagged_charts": flagged_count,
            "failed_charts": failed_count,
            "cleaning_actions_applied": len(cleaning_diff),
            "audit_log_entries": len(audit_log),
        },
    }

    audit_entry = log_code_action(
        stage="assemble",
        action="assembled_dashboard",
        detail=dashboard["stats"],
    )

    return {
        "dashboard": dashboard,
        "audit_log": [audit_entry],
    }

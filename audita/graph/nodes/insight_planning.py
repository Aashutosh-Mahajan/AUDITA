"""
Insight planning node — LLM call #2.

Asks the LLM for a ranked list of visualization intents, validates columns
and dtype compatibility against the clean profile and chart registry, then
applies deterministic filtering (priority threshold + deduplication).
"""

import json
from typing import Any

from pydantic import ValidationError

from audita.core.audit_log import log_llm_action, log_code_action
from audita.core.chart_registry import validate_chart_compatibility, DtypeCompatibilityError
from audita.core.llm_client import (
    request_viz_intents,
    _pydantic_list_to_tool_schema,
)
from audita.core.schemas import (
    VizIntent,
    AuditLogEntry,
)

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PRIORITY_SCORE = 2.5


# ---------------------------------------------------------------------------
# Validation & filtering
# ---------------------------------------------------------------------------

def _validate_intents(
    raw_intents: list[dict[str, Any]],
    clean_profile: dict[str, Any],
    cleaned_csv_path: str,
) -> tuple[list[VizIntent], list[AuditLogEntry]]:
    """Validate LLM-proposed intents against real columns and dtype rules."""
    valid_columns = list(clean_profile.keys())
    df = pd.read_csv(cleaned_csv_path)

    valid: list[VizIntent] = []
    rejections: list[AuditLogEntry] = []

    for raw in raw_intents:
        # Parse
        try:
            intent = VizIntent(**raw)
        except (ValidationError, TypeError) as e:
            rejections.append(
                log_code_action(
                    stage="insight_planning",
                    action="rejected_invalid_intent",
                    detail={"raw_intent": raw, "reason": str(e)},
                )
            )
            continue

        # Check columns exist
        missing_cols = [c for c in intent.columns if c not in valid_columns]
        if missing_cols:
            rejections.append(
                log_code_action(
                    stage="insight_planning",
                    action="rejected_nonexistent_columns",
                    detail={
                        "intent_columns": intent.columns,
                        "missing": missing_cols,
                    },
                )
            )
            continue

        # Dtype compatibility check (deterministic, not LLM)
        try:
            validate_chart_compatibility(df, intent)
        except DtypeCompatibilityError as e:
            rejections.append(
                log_code_action(
                    stage="insight_planning",
                    action="rejected_dtype_incompatible",
                    detail={
                        "chart_type": intent.chart_type.value,
                        "columns": intent.columns,
                        "reason": str(e),
                    },
                )
            )
            continue

        valid.append(intent)

    return valid, rejections


def _filter_and_deduplicate(intents: list[VizIntent]) -> list[VizIntent]:
    """Apply deterministic filtering:
    1. Drop intents with priority_score < MIN_PRIORITY_SCORE
    2. Deduplicate same column set + category (keep highest score)
    """
    # Filter by priority
    filtered = [i for i in intents if i.priority_score >= MIN_PRIORITY_SCORE]

    # Deduplicate: key = (frozenset(columns), category)
    best: dict[tuple[frozenset[str], str], VizIntent] = {}
    for intent in filtered:
        key = (frozenset(intent.columns), intent.category)
        if key not in best or intent.priority_score > best[key].priority_score:
            best[key] = intent

    # Sort by priority descending
    return sorted(best.values(), key=lambda i: i.priority_score, reverse=True)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def insight_planning(state: dict) -> dict:
    """LangGraph node: ask the LLM for visualization intents, validate and filter."""
    clean_profile: dict = state["clean_profile"]
    cleaned_csv_path: str = state["cleaned_csv_path"]
    valid_columns = list(clean_profile.keys())

    # Build the tool schema
    tool_schema = _pydantic_list_to_tool_schema(
        VizIntent,
        tool_name="propose_visualizations",
        description="Propose a ranked list of visualizations for the dataset",
    )

    # LLM call
    raw_intents = request_viz_intents(
        clean_profile=clean_profile,
        column_names=valid_columns,
        viz_intent_schema=tool_schema,
    )

    # Validate
    valid_intents, rejection_entries = _validate_intents(
        raw_intents, clean_profile, cleaned_csv_path
    )

    # Filter & deduplicate
    final_intents = _filter_and_deduplicate(valid_intents)

    # Audit
    audit_entries: list[AuditLogEntry] = [
        log_llm_action(
            stage="insight_planning",
            action="proposed_visualizations",
            detail={
                "total_proposed": len(raw_intents),
                "valid_after_check": len(valid_intents),
                "rejected": len(rejection_entries),
                "final_after_filter": len(final_intents),
            },
        ),
        *rejection_entries,
    ]

    return {
        "proposed_visualizations": final_intents,
        "audit_log": audit_entries,
    }

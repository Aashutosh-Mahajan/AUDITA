"""
Cleaning plan node — LLM call #1.

Asks the LLM to propose cleaning actions from the fixed CleaningActionType
enum, validated against real column names. Invalid proposals are rejected
and logged, not silently coerced.
"""

from typing import Any

from pydantic import ValidationError

from audita.core.audit_log import log_code_action, log_llm_action
from audita.core.llm_client import (
    _pydantic_list_to_tool_schema,
    request_cleaning_plan,
)
from audita.core.schemas import (
    AuditLogEntry,
    CleaningAction,
    CleaningActionType,
)


def _validate_actions(
    raw_actions: list[dict[str, Any]],
    valid_columns: list[str],
) -> tuple[list[CleaningAction], list[AuditLogEntry]]:
    """Validate raw LLM-proposed actions against real columns.

    Returns (valid_actions, rejection_audit_entries).
    """
    valid: list[CleaningAction] = []
    rejections: list[AuditLogEntry] = []

    for raw in raw_actions:
        # Try to parse as CleaningAction
        try:
            action = CleaningAction(**raw)
        except (ValidationError, TypeError) as e:
            rejections.append(
                log_code_action(
                    stage="cleaning_plan",
                    action="rejected_invalid_action",
                    detail={
                        "raw_action": raw,
                        "reason": f"Pydantic validation failed: {e}",
                    },
                )
            )
            continue

        # Validate column exists
        if action.column not in valid_columns:
            rejections.append(
                log_code_action(
                    stage="cleaning_plan",
                    action="rejected_nonexistent_column",
                    detail={
                        "column": action.column,
                        "action_type": action.action_type.value,
                        "valid_columns": valid_columns,
                    },
                )
            )
            continue

        valid.append(action)

    return valid, rejections


def cleaning_plan(state: dict) -> dict:
    """LangGraph node: ask the LLM for a cleaning plan, then validate it.

    Uses the quality audit (not the raw DataFrame) as LLM context.
    Validates every proposed column against ``raw_profile.columns``.
    """
    quality_audit: dict = state["quality_audit"]
    raw_profile: dict = state["raw_profile"]
    valid_columns = [col["name"] for col in raw_profile["columns"]]

    # Build the tool schema for a list of CleaningAction
    tool_schema = _pydantic_list_to_tool_schema(
        CleaningAction,
        tool_name="propose_cleaning_actions",
        description="Propose a list of cleaning actions for the dataset",
    )

    # LLM call
    raw_actions = request_cleaning_plan(
        quality_audit=quality_audit,
        column_names=valid_columns,
        cleaning_action_schema=tool_schema,
    )

    # Validate
    valid_actions, rejection_entries = _validate_actions(raw_actions, valid_columns)

    # If all actions were rejected, add NO_ACTION for each column as fallback
    if not valid_actions:
        valid_actions = [
            CleaningAction(
                column=col,
                action_type=CleaningActionType.NO_ACTION,
                rationale="Fallback — all LLM proposals were invalid",
            )
            for col in valid_columns
        ]

    # Audit entries
    audit_entries: list[AuditLogEntry] = [
        log_llm_action(
            stage="cleaning_plan",
            action="proposed_cleaning_actions",
            detail={
                "total_proposed": len(raw_actions),
                "valid_accepted": len(valid_actions),
                "rejected": len(rejection_entries),
            },
        ),
        *rejection_entries,
    ]

    return {
        "cleaning_plan": valid_actions,
        "human_approved_cleaning": False,  # gate flag — awaiting human approval
        "audit_log": audit_entries,
    }

"""
Cleaning execution node — dispatches approved cleaning actions through
the deterministic cleaning registry.

Snapshots before/after stats per column to build CleaningDiffEntry records.
"""

import os
import tempfile
from typing import Any

import pandas as pd

from audita.core.audit_log import log_code_action
from audita.core.cleaning_registry import execute_cleaning_action
from audita.core.schemas import (
    AuditLogEntry,
    CleaningAction,
    CleaningActionType,
    CleaningDiffEntry,
)


def _column_stats(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Compute a snapshot of key stats for a column (for diff tracking)."""
    if col not in df.columns:
        return {"dropped": True}

    series = df[col]
    stats: dict[str, Any] = {
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean()), 4),
        "n_unique": int(series.nunique(dropna=True)),
        "dtype": str(series.dtype),
    }

    if pd.api.types.is_numeric_dtype(series):
        stats.update(
            {
                "mean": round(float(series.mean()), 4)
                if not series.isna().all()
                else None,
                "std": round(float(series.std()), 4)
                if not series.isna().all()
                else None,
                "min": float(series.min()) if not series.isna().all() else None,
                "max": float(series.max()) if not series.isna().all() else None,
            }
        )

    return stats


def cleaning_exec(state: dict) -> dict:
    """LangGraph node: execute approved cleaning actions sequentially.

    Reads the DataFrame from ``state["csv_path"]``, applies each approved
    CleaningAction via the registry, tracks before/after diffs, and saves
    the cleaned DataFrame to a new temp path.
    """
    csv_path: str = state["csv_path"]
    cleaning_plan: list[CleaningAction] = state["cleaning_plan"]

    df = pd.read_csv(csv_path)

    diffs: list[CleaningDiffEntry] = []
    audit_entries: list[AuditLogEntry] = []

    for action in cleaning_plan:
        # Skip NO_ACTION
        if action.action_type == CleaningActionType.NO_ACTION:
            audit_entries.append(
                log_code_action(
                    stage="cleaning_exec",
                    action="skipped_no_action",
                    detail={"column": action.column},
                )
            )
            continue

        # Snapshot before
        before_stats = _column_stats(df, action.column)
        rows_before = len(df)

        # Execute
        df = execute_cleaning_action(df, action)

        # Snapshot after
        after_stats = _column_stats(df, action.column)
        rows_after = len(df)

        # Compute rows affected
        if action.action_type == CleaningActionType.DROP_ROWS:
            rows_affected = rows_before - rows_after
        elif action.action_type == CleaningActionType.DROP_COLUMN:
            rows_affected = rows_before  # entire column removed
        else:
            # For imputation/capping/parsing: count how many cells changed
            rows_affected = abs(
                before_stats.get("missing_count", 0)
                - after_stats.get("missing_count", 0)
            )

        diff = CleaningDiffEntry(
            column=action.column,
            action_type=action.action_type,
            rows_affected=rows_affected,
            before_stat=before_stats,
            after_stat=after_stats,
        )
        diffs.append(diff)

        audit_entries.append(
            log_code_action(
                stage="cleaning_exec",
                action=f"executed_{action.action_type.value}",
                detail={
                    "column": action.column,
                    "rows_affected": rows_affected,
                    "rationale": action.rationale,
                },
            )
        )

    # Save cleaned DataFrame
    temp_dir = tempfile.mkdtemp(prefix="audita_clean_")
    cleaned_csv_path = os.path.join(temp_dir, "cleaned_data.csv")
    df.to_csv(cleaned_csv_path, index=False)

    return {
        "cleaned_csv_path": cleaned_csv_path,
        "cleaning_diff": diffs,
        "audit_log": audit_entries,
    }

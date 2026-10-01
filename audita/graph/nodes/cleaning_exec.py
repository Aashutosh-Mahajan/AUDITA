"""
Cleaning execution node — dispatches approved cleaning actions through
the deterministic cleaning registry.

Snapshots before/after stats per column to build CleaningDiffEntry records.
"""

from typing import Any

import pandas as pd

from audita.core.audit_log import log_code_action
from audita.core.cleaning_registry import execute_cleaning_action
from audita.core.frame_io import read_frame, write_frame
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


def _count_changed_cells(before: pd.Series, after: pd.Series) -> int:
    """Count rows whose value in a column differs between two snapshots.

    A cell missing in both snapshots is never counted. When the column's dtype
    changed (``parse_dates`` turning strings into Timestamps) every other cell
    was rewritten, even though a midnight Timestamp prints as the very same
    "2024-01-01" text, so string comparison alone would report 0.
    """
    before = before.reset_index(drop=True)
    after = after.reset_index(drop=True)
    both_missing = before.isna() & after.isna()

    if before.dtype != after.dtype:
        return int((~both_missing).sum())

    differs = before.astype(str) != after.astype(str)
    return int((differs & ~both_missing).sum())


def cleaning_exec(state: dict) -> dict:
    """LangGraph node: execute approved cleaning actions sequentially.

    Reads the DataFrame from ``state["csv_path"]``, applies each approved
    CleaningAction via the registry, tracks before/after diffs, and saves
    the cleaned DataFrame to a new temp path.
    """
    csv_path: str = state["csv_path"]
    cleaning_plan: list[CleaningAction] = state["cleaning_plan"]

    df = read_frame(csv_path)

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
        before_values = (
            df[action.column].copy() if action.column in df.columns else None
        )

        # Execute — a single bad action must not abort the whole plan
        try:
            df = execute_cleaning_action(df, action)
        except Exception as exc:
            audit_entries.append(
                log_code_action(
                    stage="cleaning_exec",
                    action="skipped_failed_action",
                    detail={
                        "column": action.column,
                        "action_type": action.action_type.value,
                        "error": str(exc),
                    },
                )
            )
            continue

        # Snapshot after
        after_stats = _column_stats(df, action.column)
        rows_after = len(df)

        # Compute rows affected
        if action.action_type == CleaningActionType.DROP_ROWS:
            rows_affected = rows_before - rows_after
        elif action.action_type == CleaningActionType.DROP_COLUMN:
            rows_affected = rows_before  # entire column removed
        elif before_values is not None and action.column in df.columns:
            # Imputation, capping, parsing and standardising all rewrite cells
            # in place, so count the cells that actually changed. Deriving this
            # from the missing-count delta reported 0 for every action that
            # does not touch missingness.
            rows_affected = _count_changed_cells(before_values, df[action.column])
        else:
            rows_affected = 0

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

    # Save cleaned DataFrame — dtypes set by parse_dates must survive here
    cleaned_csv_path = write_frame(df, prefix="audita_clean_", stem="cleaned_data")

    return {
        "cleaned_csv_path": cleaned_csv_path,
        "cleaning_diff": diffs,
        "audit_log": audit_entries,
    }

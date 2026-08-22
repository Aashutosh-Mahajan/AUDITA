"""
LangGraph pipeline state definition.

``audit_log`` uses ``operator.add`` because it is written to concurrently by
parallel ``Send()``-dispatched nodes — this is required for LangGraph's fan-in
to work correctly.

``completed_charts`` needs the same concurrent fan-in but must NOT plainly
append: ``self_check`` rewrites the charts it has verified, and a plain
``operator.add`` would leave both the pre- and post-verification copies in the
list. ``merge_charts`` therefore appends new charts but replaces existing ones
in place, keyed by their intent.
"""

import operator
from typing import Annotated, TypedDict

from audita.core.schemas import (
    AuditLogEntry,
    ChartResult,
    CleaningAction,
    CleaningDiffEntry,
    VizIntent,
)


def _chart_key(chart: ChartResult) -> tuple:
    """Stable identity for a chart — its intent, which survives re-verification."""
    return (
        chart.intent.chart_type.value,
        tuple(chart.intent.columns),
        chart.intent.category,
    )


def merge_charts(
    existing: list[ChartResult] | None,
    incoming: list[ChartResult] | None,
) -> list[ChartResult]:
    """Reducer for ``completed_charts``: append new charts, replace known ones.

    Preserves the order in which charts first appeared so the dashboard stays
    stable across a verification pass or a retry.
    """
    merged: list[ChartResult] = list(existing or [])
    index = {_chart_key(c): i for i, c in enumerate(merged)}

    for chart in incoming or []:
        key = _chart_key(chart)
        if key in index:
            merged[index[key]] = chart
        else:
            index[key] = len(merged)
            merged.append(chart)

    return merged


class PipelineState(TypedDict):
    """Full state flowing through the AUDITA LangGraph pipeline."""

    # Ingest stage
    csv_path: str
    raw_profile: dict

    # Quality audit stage
    quality_audit: dict

    # Cleaning stages
    cleaning_plan: list[CleaningAction]
    cleaned_csv_path: str
    cleaning_diff: list[CleaningDiffEntry]

    # Post-cleaning profiling
    clean_profile: dict

    # Visualization stages
    proposed_visualizations: list[VizIntent]
    completed_charts: Annotated[list[ChartResult], merge_charts]  # fan-in, dedup by intent

    # Audit trail — concurrent writes from parallel nodes
    audit_log: Annotated[list[AuditLogEntry], operator.add]

    # Human-in-the-loop gate flag (Section 6)
    human_approved_cleaning: bool

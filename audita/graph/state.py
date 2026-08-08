"""
LangGraph pipeline state definition.

``completed_charts`` and ``audit_log`` use ``operator.add`` because they are
written to concurrently by parallel ``Send()``-dispatched nodes — this is
required for LangGraph's fan-in to work correctly.
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
    completed_charts: Annotated[list[ChartResult], operator.add]  # fan-in accumulator

    # Audit trail — concurrent writes from parallel nodes
    audit_log: Annotated[list[AuditLogEntry], operator.add]

    # Human-in-the-loop gate flag (Section 6)
    human_approved_cleaning: bool

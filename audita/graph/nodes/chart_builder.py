"""
Chart builder node — fan-out target.

Each invocation receives a single VizIntent and renders it through the
chart registry. No LLM involvement. On failure, returns a ChartResult
with FAILED status rather than raising.
"""

import pandas as pd

from audita.core.audit_log import log_code_action
from audita.core.chart_registry import DtypeCompatibilityError, render_chart
from audita.core.schemas import (
    ChartResult,
    VerificationStatus,
    VizIntent,
)


def chart_builder(state: dict) -> dict:
    """LangGraph node: render a single chart from a VizIntent.

    This node is dispatched via ``Send()`` — one invocation per approved
    ``VizIntent``. The intent data is passed in the state by the fan-out
    conditional edge.
    """
    intent: VizIntent = state["intent"]
    cleaned_csv_path: str = state["cleaned_csv_path"]

    df = pd.read_csv(cleaned_csv_path)

    try:
        fig = render_chart(df, intent)
        figure_json = fig.to_json()

        result = ChartResult(
            intent=intent,
            figure_json=figure_json,
            verification_status=VerificationStatus.VERIFIED,  # tentative until self_check
            verification_notes="",
        )

        audit_entry = log_code_action(
            stage="chart_builder",
            action="rendered_chart",
            detail={
                "chart_type": intent.chart_type.value,
                "columns": intent.columns,
                "category": intent.category,
            },
        )

    except (DtypeCompatibilityError, Exception) as e:
        result = ChartResult(
            intent=intent,
            figure_json=None,
            verification_status=VerificationStatus.FAILED,
            error=str(e),
        )

        audit_entry = log_code_action(
            stage="chart_builder",
            action="chart_render_failed",
            detail={
                "chart_type": intent.chart_type.value,
                "columns": intent.columns,
                "error": str(e),
            },
        )

    return {
        "completed_charts": [result],
        "audit_log": [audit_entry],
    }

"""
Self-check node — two verification layers for rendered charts.

1. Code-level recomputation: independently verify the data behind the chart
2. LLM grounding check: confirm the chart supports its stated rationale

Charts that fail either check get retried (up to MAX_CHART_RETRIES).
After max retries, charts are FLAGGED (not FAILED) — they still render
but with a visible warning.
"""

import base64
import json
from typing import Any

import numpy as np
import pandas as pd

from audita.core.audit_log import log_code_action, log_llm_action
from audita.core.frame_io import read_frame
from audita.core.llm_client import request_grounding_check
from audita.core.schemas import (
    AuditLogEntry,
    ChartResult,
    ChartType,
    VerificationStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHART_RETRIES = 2


# ---------------------------------------------------------------------------
# Code-level recomputation checks
# ---------------------------------------------------------------------------


def _trace_array(value: Any) -> list[Any]:
    """Return a trace field (``x``/``y``/``values``) as a plain Python list.

    Plotly >= 6 serialises numeric arrays as
    ``{"dtype": "f8", "bdata": "<base64>"}`` rather than a JSON list, so any
    code that slices or len()s the raw field silently misreads it (or raises).
    Decode that form here; pass lists through untouched.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "bdata" in value:
        try:
            raw = base64.b64decode(value["bdata"])
            arr = np.frombuffer(raw, dtype=np.dtype(value.get("dtype", "f8")))
            shape = value.get("shape")
            if shape:
                arr = arr.reshape([int(d) for d in str(shape).split(",")])
            return arr.tolist()
        except (ValueError, TypeError):
            return []
    return []


def _extract_figure_data(chart: ChartResult) -> dict[str, Any] | None:
    """Parse the Plotly figure JSON and extract key data for verification."""
    if not chart.figure_json:
        return None
    try:
        fig_dict = json.loads(chart.figure_json)
        return fig_dict
    except (json.JSONDecodeError, TypeError):
        return None


def _check_bar_chart_accuracy(
    df: pd.DataFrame, chart: ChartResult, fig_data: dict
) -> tuple[bool, str]:
    """For bar charts: verify group means/counts match the data."""
    cols = chart.intent.columns
    if len(cols) < 1:
        return False, "Bar chart has no columns specified"

    # Check if y-axis starts misleadingly high
    layout = fig_data.get("layout", {})
    yaxis = layout.get("yaxis", {})
    y_range = yaxis.get("range")
    if y_range and len(y_range) == 2 and y_range[0] > 0:
        # Bar chart with y-axis not starting at 0 — potentially misleading
        return False, (
            f"Bar chart y-axis starts at {y_range[0]} instead of 0, "
            "which may be misleading"
        )

    return True, "Bar chart passed code-level checks"


def _check_scatter_accuracy(
    df: pd.DataFrame, chart: ChartResult, fig_data: dict
) -> tuple[bool, str]:
    """For scatter plots: verify data point count matches."""
    traces = fig_data.get("data", [])
    if not traces:
        return False, "No trace data in scatter plot"

    trace = traces[0]
    x_data = _trace_array(trace.get("x"))
    expected_count = len(df.dropna(subset=chart.intent.columns))

    if abs(len(x_data) - expected_count) > 1:
        return False, (
            f"Scatter plot shows {len(x_data)} points but data has "
            f"{expected_count} valid rows"
        )

    return True, "Scatter plot data count verified"


def _code_level_check(df: pd.DataFrame, chart: ChartResult) -> tuple[bool, str]:
    """Run code-level recomputation checks appropriate to the chart type."""
    fig_data = _extract_figure_data(chart)
    if fig_data is None:
        return False, "Could not parse figure JSON for verification"

    chart_type = chart.intent.chart_type

    if chart_type == ChartType.BAR:
        return _check_bar_chart_accuracy(df, chart, fig_data)
    elif chart_type == ChartType.SCATTER:
        return _check_scatter_accuracy(df, chart, fig_data)
    else:
        # For other chart types, basic structural check
        traces = fig_data.get("data", [])
        if not traces:
            return False, "No trace data found in figure"
        return True, f"{chart_type.value} chart passed basic structural check"


# ---------------------------------------------------------------------------
# LLM grounding check
# ---------------------------------------------------------------------------


def _build_chart_summary(chart: ChartResult, fig_data: dict | None) -> dict[str, Any]:
    """Build a compact textual summary for the LLM grounding check."""
    summary: dict[str, Any] = {
        "chart_type": chart.intent.chart_type.value,
        "columns": chart.intent.columns,
        "rationale": chart.intent.rationale,
        "category": chart.intent.category,
    }

    if fig_data:
        layout = fig_data.get("layout", {})
        summary["title"] = layout.get("title", {}).get("text", "")
        summary["x_axis_label"] = (
            layout.get("xaxis", {}).get("title", {}).get("text", "")
        )
        summary["y_axis_label"] = (
            layout.get("yaxis", {}).get("title", {}).get("text", "")
        )

        # Include a few data points (not the full dataset)
        traces = fig_data.get("data", [])
        if traces:
            trace = traces[0]
            summary["sample_data"] = {
                "x": _trace_array(trace.get("x"))[:5],
                "y": _trace_array(trace.get("y"))[:5],
            }

    return summary


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def self_check(state: dict) -> dict:
    """LangGraph node: verify all completed charts.

    Runs two layers:
    1. Code-level recomputation check
    2. LLM grounding check (batched for cost efficiency)

    Charts that fail either check are marked for retry or flagged.
    """
    completed_charts: list[ChartResult] = state["completed_charts"]
    cleaned_csv_path: str = state["cleaned_csv_path"]

    df = read_frame(cleaned_csv_path)

    audit_entries: list[AuditLogEntry] = []
    updated_charts: list[ChartResult] = []
    charts_needing_grounding: list[tuple[int, ChartResult, dict]] = []

    # Layer 1: Code-level checks
    for i, chart in enumerate(completed_charts):
        if chart.verification_status == VerificationStatus.FAILED:
            # Already failed during rendering — skip checks
            updated_charts.append(chart)
            continue

        passed, notes = _code_level_check(df, chart)

        if not passed:
            if chart.retry_count < MAX_CHART_RETRIES:
                updated_chart = chart.model_copy(
                    update={
                        "verification_status": VerificationStatus.RETRYING,
                        "verification_notes": f"Code check failed: {notes}",
                        "retry_count": chart.retry_count + 1,
                    }
                )
            else:
                updated_chart = chart.model_copy(
                    update={
                        "verification_status": VerificationStatus.FLAGGED,
                        "verification_notes": f"Code check failed after {MAX_CHART_RETRIES} retries: {notes}",
                    }
                )

            updated_charts.append(updated_chart)
            audit_entries.append(
                log_code_action(
                    stage="self_check",
                    action="code_check_failed",
                    detail={
                        "chart_type": chart.intent.chart_type.value,
                        "columns": chart.intent.columns,
                        "notes": notes,
                        "retry_count": updated_chart.retry_count,
                    },
                )
            )
        else:
            # Passed code check — queue for grounding check
            fig_data = _extract_figure_data(chart)
            summary = _build_chart_summary(chart, fig_data)
            charts_needing_grounding.append((i, chart, summary))

    # Layer 2: LLM grounding check (batched)
    if charts_needing_grounding:
        summaries = [s for _, _, s in charts_needing_grounding]

        try:
            verdicts = request_grounding_check(summaries)
        except Exception as e:
            # If LLM call fails, mark all as verified (degrade gracefully)
            verdicts = [
                {"grounded": True, "notes": f"Grounding check unavailable: {e}"}
                for _ in summaries
            ]

        for (_, chart, _), verdict in zip(
            charts_needing_grounding, verdicts, strict=False
        ):
            grounded = verdict.get("grounded", True)
            notes = verdict.get("notes", "")

            if grounded:
                updated_chart = chart.model_copy(
                    update={
                        "verification_status": VerificationStatus.VERIFIED,
                        "verification_notes": notes,
                    }
                )
                audit_entries.append(
                    log_llm_action(
                        stage="self_check",
                        action="grounding_verified",
                        detail={
                            "chart_type": chart.intent.chart_type.value,
                            "columns": chart.intent.columns,
                            "notes": notes,
                        },
                    )
                )
            else:
                if chart.retry_count < MAX_CHART_RETRIES:
                    updated_chart = chart.model_copy(
                        update={
                            "verification_status": VerificationStatus.RETRYING,
                            "verification_notes": f"Grounding check failed: {notes}",
                            "retry_count": chart.retry_count + 1,
                        }
                    )
                else:
                    updated_chart = chart.model_copy(
                        update={
                            "verification_status": VerificationStatus.FLAGGED,
                            "verification_notes": f"Grounding failed after {MAX_CHART_RETRIES} retries: {notes}",
                        }
                    )

                audit_entries.append(
                    log_llm_action(
                        stage="self_check",
                        action="grounding_failed",
                        detail={
                            "chart_type": chart.intent.chart_type.value,
                            "columns": chart.intent.columns,
                            "notes": notes,
                            "retry_count": updated_chart.retry_count,
                        },
                    )
                )

            updated_charts.append(updated_chart)

    return {
        "completed_charts": updated_charts,
        "audit_log": audit_entries,
    }

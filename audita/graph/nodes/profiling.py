"""
Profiling node — re-runs the exact same audit function from quality_audit.py
on the cleaned DataFrame to produce ``clean_profile``.

Reusing the same function (not a near-duplicate) ensures before/after are
directly comparable in the UI.
"""

from audita.core.audit_log import log_code_action
from audita.core.frame_io import read_frame
from audita.graph.nodes.quality_audit import compute_column_audit


def profiling(state: dict) -> dict:
    """LangGraph node: profile the cleaned DataFrame.

    Reads from ``state["cleaned_csv_path"]`` and runs ``compute_column_audit``
    (the exact same function used in the quality_audit node).
    """
    cleaned_csv_path: str = state["cleaned_csv_path"]
    df = read_frame(cleaned_csv_path)

    clean_profile = compute_column_audit(df)

    audit_entry = log_code_action(
        stage="profiling",
        action="computed_clean_profile",
        detail={
            "columns_profiled": len(clean_profile),
            "rows": len(df),
        },
    )

    return {
        "clean_profile": clean_profile,
        "audit_log": [audit_entry],
    }

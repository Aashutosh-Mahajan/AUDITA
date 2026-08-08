"""
Streamlit UI component helpers — reusable rendering primitives for the
AUDITA dashboard: status cards, verification badges, data tables, metrics.
"""

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Verification status badges
# ---------------------------------------------------------------------------

_STATUS_CONFIG = {
    "verified": {"color": "#22c55e", "icon": "✅", "label": "Verified"},
    "flagged": {"color": "#eab308", "icon": "⚠️", "label": "Flagged"},
    "failed": {"color": "#ef4444", "icon": "❌", "label": "Failed"},
    "retrying": {"color": "#3b82f6", "icon": "🔄", "label": "Retrying"},
}


def verification_badge(status: str, notes: str = "") -> None:
    """Render a colored verification status badge."""
    config = _STATUS_CONFIG.get(status, _STATUS_CONFIG["failed"])
    badge_html = (
        f'<span style="'
        f"background-color: {config['color']}20; "
        f"color: {config['color']}; "
        f"padding: 4px 12px; "
        f"border-radius: 12px; "
        f"font-size: 0.85em; "
        f"font-weight: 600; "
        f"border: 1px solid {config['color']}40; "
        f'">'
        f"{config['icon']} {config['label']}"
        f"</span>"
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    if notes and status == "flagged":
        st.caption(f"📝 {notes}")


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------


def metric_card(
    label: str, value: Any, delta: Any = None, delta_color: str = "normal"
) -> None:
    """Render a Streamlit metric with consistent styling."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def stats_row(stats: dict[str, Any]) -> None:
    """Render a row of metric cards from a stats dict."""
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items(), strict=False):
        with col:
            metric_card(label=label.replace("_", " ").title(), value=value)


# ---------------------------------------------------------------------------
# Data quality comparison cards
# ---------------------------------------------------------------------------


def quality_comparison_card(
    col_name: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    """Render a before/after comparison card for a column's quality metrics."""
    with st.expander(f"📊 {col_name}", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Before Cleaning**")
            for key, val in before.items():
                if key in ("top_values", "near_duplicate_labels"):
                    continue
                st.text(f"  {key}: {val}")

        with c2:
            st.markdown("**After Cleaning**")
            for key, val in after.items():
                if key in ("top_values", "near_duplicate_labels"):
                    continue
                st.text(f"  {key}: {val}")

        # Highlight improvements
        before_missing = before.get("missing_pct", 0)
        after_missing = after.get("missing_pct", 0)
        if before_missing > 0 and after_missing < before_missing:
            improvement = round((before_missing - after_missing) * 100, 1)
            st.success(f"Missing data reduced by {improvement} percentage points")


# ---------------------------------------------------------------------------
# Cleaning diff table
# ---------------------------------------------------------------------------


def cleaning_diff_table(diffs: list[dict[str, Any]]) -> None:
    """Render the cleaning diff table with action rationale."""
    if not diffs:
        st.info("No cleaning actions were applied.")
        return

    for diff in diffs:
        with st.expander(
            f"🔧 {diff['column']} — `{diff['action']}` ({diff['rows_affected']} rows affected)",
            expanded=False,
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before**")
                st.json(diff["before"])
            with c2:
                st.markdown("**After**")
                st.json(diff["after"])


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def audit_log_section(entries: list[dict[str, Any]]) -> None:
    """Render a collapsible audit log table."""
    if not entries:
        st.info("No audit log entries.")
        return

    with st.expander("📋 Full Audit Log", expanded=False):
        for entry in entries:
            actor_icon = "🤖" if entry["actor"] == "llm" else "⚙️"
            st.markdown(
                f"**{actor_icon} [{entry['stage']}]** {entry['action']} "
                f"— _{entry['timestamp']}_"
            )
            if entry.get("detail"):
                st.json(entry["detail"])
            st.divider()


# ---------------------------------------------------------------------------
# Progress indicator
# ---------------------------------------------------------------------------

_STAGE_LABELS = {
    "ingest": ("📁", "Ingesting data..."),
    "quality_audit": ("🔍", "Auditing data quality..."),
    "cleaning_plan": ("🧹", "Proposing cleaning plan..."),
    "cleaning_exec": ("⚡", "Executing cleaning actions..."),
    "profiling": ("📊", "Profiling cleaned data..."),
    "insight_planning": ("💡", "Planning visualizations..."),
    "chart_builder": ("📈", "Building charts..."),
    "self_check": ("✔️", "Verifying charts..."),
    "assemble": ("🏗️", "Assembling dashboard..."),
}


def stage_progress(stage: str) -> None:
    """Render a progress indicator for the current pipeline stage."""
    icon, label = _STAGE_LABELS.get(stage, ("⏳", f"Running {stage}..."))
    st.markdown(f"### {icon} {label}")


# ---------------------------------------------------------------------------
# Cleaning plan approval widget
# ---------------------------------------------------------------------------


def cleaning_plan_approval(
    cleaning_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render checkboxes for each proposed cleaning action.

    Returns the list of approved actions (those the user checked).
    """
    st.markdown("### 🧹 Proposed Cleaning Plan")
    st.markdown("Review and approve/reject individual cleaning actions:")

    approved = []

    for i, action in enumerate(cleaning_plan):
        col_name = action.get("column", "unknown")
        action_type = action.get("action_type", "unknown")
        rationale = action.get("rationale", "")

        checked = st.checkbox(
            f"**{col_name}** — `{action_type}`",
            value=True,
            key=f"clean_action_{i}",
            help=rationale,
        )

        if checked:
            approved.append(action)

    st.markdown(f"**{len(approved)}/{len(cleaning_plan)}** actions approved")

    return approved

"""
Dashboard section renderers — each function renders one section of the
AUDITA Streamlit dashboard from cached pipeline state.

Sections:
1. Data quality summary (before/after)
2. Cleaning diff table with rationale
3. Charts grouped by category with verification badges
4. Collapsible full audit log
"""

import json
from typing import Any

import plotly.io as pio
import streamlit as st

from audita.ui.components import (
    stats_row,
    quality_comparison_card,
    cleaning_diff_table,
    verification_badge,
    audit_log_section,
)


# ---------------------------------------------------------------------------
# Section 1: Data Quality Summary
# ---------------------------------------------------------------------------

def render_quality_summary(dashboard: dict[str, Any]) -> None:
    """Render before/after data quality comparison."""
    st.markdown("## 📊 Data Quality Summary")

    quality = dashboard.get("quality_summary", {})
    before = quality.get("before", {})
    after = quality.get("after", {})

    if not before and not after:
        st.info("No quality data available.")
        return

    # High-level stats
    raw_profile = dashboard.get("raw_profile", {})
    stats = {
        "Total Rows": raw_profile.get("n_rows", "—"),
        "Total Columns": raw_profile.get("n_cols", "—"),
        "Cleaning Actions": dashboard.get("stats", {}).get("cleaning_actions_applied", 0),
        "Charts Generated": dashboard.get("stats", {}).get("total_charts", 0),
    }
    stats_row(stats)

    st.markdown("---")
    st.markdown("### Per-Column Quality (Before → After)")

    # Render per-column comparisons
    all_columns = set(list(before.keys()) + list(after.keys()))
    for col_name in sorted(all_columns):
        col_before = before.get(col_name, {})
        col_after = after.get(col_name, {})
        quality_comparison_card(col_name, col_before, col_after)


# ---------------------------------------------------------------------------
# Section 2: Cleaning Diff
# ---------------------------------------------------------------------------

def render_cleaning_diff(dashboard: dict[str, Any]) -> None:
    """Render the cleaning diff table with action rationale."""
    st.markdown("## 🔧 Cleaning Actions Applied")

    diffs = dashboard.get("cleaning_diff_table", [])
    cleaning_diff_table(diffs)


# ---------------------------------------------------------------------------
# Section 3: Visualizations by Category
# ---------------------------------------------------------------------------

def render_charts(dashboard: dict[str, Any]) -> None:
    """Render charts grouped by category with verification badges."""
    st.markdown("## 📈 Visualizations")

    charts_by_category = dashboard.get("charts_by_category", {})

    if not charts_by_category:
        st.info("No charts were generated.")
        return

    # Category tabs
    categories = list(charts_by_category.keys())
    tabs = st.tabs([f"📌 {cat.title()}" for cat in categories])

    for tab, category in zip(tabs, categories):
        with tab:
            charts = charts_by_category[category]

            for i, chart in enumerate(charts):
                with st.container():
                    # Header with verification badge
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f"**{chart['chart_type'].title()}** — "
                            f"{', '.join(chart['columns'])}"
                        )
                        st.caption(chart.get("rationale", ""))
                    with col2:
                        verification_badge(
                            chart.get("verification_status", "failed"),
                            chart.get("verification_notes", ""),
                        )

                    # Render the chart
                    figure_json = chart.get("figure_json")
                    if figure_json:
                        try:
                            fig = pio.from_json(figure_json)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render chart: {e}")
                    elif chart.get("error"):
                        st.error(f"Chart failed: {chart['error']}")
                    else:
                        st.warning("No figure data available.")

                    st.divider()


# ---------------------------------------------------------------------------
# Section 4: Audit Log
# ---------------------------------------------------------------------------

def render_audit_log(dashboard: dict[str, Any]) -> None:
    """Render the collapsible full audit log."""
    st.markdown("## 📋 Audit Trail")

    entries = dashboard.get("audit_log_table", [])
    audit_log_section(entries)


# ---------------------------------------------------------------------------
# Full dashboard renderer
# ---------------------------------------------------------------------------

def render_dashboard(dashboard: dict[str, Any]) -> None:
    """Render the complete AUDITA dashboard from assembled state."""
    render_quality_summary(dashboard)
    st.markdown("---")
    render_cleaning_diff(dashboard)
    st.markdown("---")
    render_charts(dashboard)
    st.markdown("---")
    render_audit_log(dashboard)

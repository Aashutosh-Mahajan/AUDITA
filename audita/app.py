"""
AUDITA — Streamlit entrypoint.

File upload → graph.stream() for live progress → cached dashboard.
Human-in-the-loop cleaning approval gate.
Normal reruns render from cache only — no re-invocation.
"""

import hashlib
import os
import tempfile

import streamlit as st

from audita.graph.build_graph import build_graph
from audita.graph.nodes.assemble import assemble
from audita.ui.components import (
    cleaning_plan_approval,
    stage_progress,
)
from audita.ui.sections import render_dashboard

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AUDITA — Data Cleaning & Visualization Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 AUDITA")
st.caption("Auditable, Self-Verifying Data Cleaning & Visualization Agent")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file to analyze, clean, and visualize.",
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "AUDITA uses AI to propose data cleaning actions and "
        "insightful visualizations — but **code executes and verifies** "
        "every step. Every action is logged in an immutable audit trail."
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_hash(uploaded_file) -> str:
    """Compute a hash of the uploaded file for caching."""
    content = uploaded_file.getvalue()
    return hashlib.sha256(content).hexdigest()[:16]


def _save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file to a temp path and return the path."""
    temp_dir = tempfile.mkdtemp(prefix="audita_upload_")
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path


# ---------------------------------------------------------------------------
# Pipeline execution states
# ---------------------------------------------------------------------------

# Initialise session state keys
if "pipeline_stage" not in st.session_state:
    st.session_state["pipeline_stage"] = None

if "graph_instance" not in st.session_state:
    st.session_state["graph_instance"] = None

if "thread_config" not in st.session_state:
    st.session_state["thread_config"] = None


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    file_hash = _file_hash(uploaded_file)
    result_key = f"result_{file_hash}"
    state_key = f"state_{file_hash}"
    stage_key = f"stage_{file_hash}"

    # Check for cached result
    if result_key in st.session_state:
        # Render from cache — no re-invocation
        st.success(
            "✅ Results loaded from cache. Upload a different file or click re-run below."
        )
        render_dashboard(st.session_state[result_key])

        # Re-run buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-run Cleaning Plan", key="rerun_clean"):
                del st.session_state[result_key]
                if state_key in st.session_state:
                    del st.session_state[state_key]
                st.rerun()
        with col2:
            if st.button("🔄 Re-run Insight Planning", key="rerun_insight"):
                del st.session_state[result_key]
                if state_key in st.session_state:
                    del st.session_state[state_key]
                st.rerun()

    else:
        # No cached result — run the pipeline
        csv_path = _save_uploaded_file(uploaded_file)

        # Build graph
        if st.session_state.get("graph_instance") is None:
            st.session_state["graph_instance"] = build_graph(with_checkpointer=True)

        graph = st.session_state["graph_instance"]
        thread_config = {"configurable": {"thread_id": file_hash}}

        # Check if we're resuming after human approval
        current_stage = st.session_state.get(stage_key, "start")

        if current_stage == "awaiting_approval":
            # Show the cleaning plan for approval
            pipeline_state = st.session_state.get(state_key, {})
            cleaning_plan = pipeline_state.get("cleaning_plan", [])

            st.markdown("---")
            st.markdown("## 🛑 Human Review Required")

            # Convert cleaning plan to dicts for the approval widget
            plan_dicts = [
                {
                    "column": a.column,
                    "action_type": a.action_type.value,
                    "rationale": a.rationale,
                    "params": a.params,
                }
                if hasattr(a, "column")
                else a
                for a in cleaning_plan
            ]

            approved = cleaning_plan_approval(plan_dicts)

            if st.button("✅ Approve and Continue", type="primary"):
                # Filter the cleaning plan to only approved actions
                from audita.core.schemas import CleaningAction

                approved_actions = [CleaningAction(**a) for a in approved]

                # Resume the graph with the approved plan
                st.session_state[stage_key] = "running_post_approval"

                # Write the approved plan into the interrupted checkpoint.
                # Resuming REQUIRES streaming with input=None — passing a state
                # dict here would start a fresh run from START instead of
                # continuing from the human gate.
                graph.update_state(
                    thread_config,
                    {
                        "cleaning_plan": approved_actions,
                        "human_approved_cleaning": True,
                    },
                )

                progress = st.empty()
                with progress.container():
                    stage_progress("cleaning_exec")

                try:
                    for event in graph.stream(
                        None,
                        config=thread_config,
                    ):
                        # Update progress display
                        for node_name in event:
                            with progress.container():
                                stage_progress(node_name)

                    # Get the full final state
                    full_state = graph.get_state(thread_config).values

                    # Assemble dashboard
                    dashboard_result = assemble(full_state)
                    dashboard = dashboard_result.get("dashboard", {})

                    # Cache the result
                    st.session_state[result_key] = dashboard
                    st.session_state[stage_key] = "complete"

                    st.rerun()

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)

        elif current_stage == "start":
            # Fresh run — start the pipeline
            progress = st.empty()

            with progress.container():
                stage_progress("ingest")

            try:
                initial_state = {"csv_path": csv_path}

                for event in graph.stream(
                    initial_state,
                    config=thread_config,
                ):
                    for node_name in event:
                        with progress.container():
                            stage_progress(node_name)

                # Graph interrupted at cleaning_exec (human gate)
                interrupted_state = graph.get_state(thread_config).values

                # Store state and switch to approval mode
                st.session_state[state_key] = interrupted_state
                st.session_state[stage_key] = "awaiting_approval"
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline error during initial run: {e}")
                st.exception(e)

else:
    # No file uploaded — show welcome state
    st.markdown("---")
    st.markdown(
        """
        ### 👋 Welcome to AUDITA

        Upload a CSV file in the sidebar to get started. AUDITA will:

        1. **Ingest** your data with automatic encoding/delimiter detection
        2. **Audit** data quality (missing values, outliers, duplicates)
        3. **Propose** a cleaning plan (AI-powered, human-approved)
        4. **Clean** your data with full before/after tracking
        5. **Visualize** key insights with verified, trustworthy charts
        6. **Log** every action in an immutable audit trail

        ---

        > *"The LLM proposes. Code executes. Code verifies."*
        """
    )

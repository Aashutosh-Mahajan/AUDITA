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
from audita.graph.nodes.assemble import assemble  # fallback if the graph errored
from audita.graph.nodes.ingest import is_excel_path, list_sheet_names
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
        "Choose a CSV or Excel file",
        type=["csv", "tsv", "xlsx", "xlsm", "xls"],
        help="Upload a CSV or Excel workbook to analyze, clean, and visualize.",
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


def _file_hash(uploaded_file, sheet_name: str | None = None) -> str:
    """Compute a cache key from the uploaded file's bytes and chosen sheet."""
    digest = hashlib.sha256(uploaded_file.getvalue())
    if sheet_name:
        digest.update(sheet_name.encode("utf-8"))
    return digest.hexdigest()[:16]


@st.cache_data(show_spinner=False)
def _save_bytes(name: str, content: bytes) -> str:
    """Persist uploaded bytes once per file so reruns reuse the same path."""
    temp_dir = tempfile.mkdtemp(prefix="audita_upload_")
    file_path = os.path.join(temp_dir, name)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path


def _save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file to a temp path and return the path."""
    return _save_bytes(uploaded_file.name, uploaded_file.getvalue())


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
    source_path = _save_uploaded_file(uploaded_file)

    # Excel workbooks may hold several sheets — let the user pick which one
    # to analyse, and key the cache on it so switching sheets re-runs.
    selected_sheet: str | None = None
    if is_excel_path(source_path):
        try:
            sheet_names = list_sheet_names(source_path)
        except Exception as exc:  # unreadable / corrupt workbook
            st.error(f"Could not read that Excel workbook: {exc}")
            st.stop()

        with st.sidebar:
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Sheet",
                    sheet_names,
                    help="Which worksheet to analyze.",
                )
            else:
                selected_sheet = sheet_names[0]
                st.caption(f"Sheet: **{selected_sheet}**")

    file_hash = _file_hash(uploaded_file, selected_sheet)
    run_key = f"run_{file_hash}"
    run_index = st.session_state.setdefault(run_key, 0)
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

        # Re-run: clear every cached key for this file AND advance the run
        # counter. The counter feeds the thread_id, so the pipeline starts on
        # a fresh checkpoint instead of resuming one that already finished.
        if st.button("🔄 Re-run Analysis", key=f"rerun_{file_hash}"):
            for key in (result_key, state_key, stage_key):
                st.session_state.pop(key, None)
            st.session_state[run_key] = run_index + 1
            st.rerun()

    else:
        # No cached result — run the pipeline

        # Build graph
        if st.session_state.get("graph_instance") is None:
            st.session_state["graph_instance"] = build_graph(with_checkpointer=True)

        graph = st.session_state["graph_instance"]
        thread_config = {
            "configurable": {"thread_id": f"{file_hash}-{run_index}"}
        }

        # Check if we're resuming after human approval. Anything other than an
        # explicit "awaiting_approval" means we have no live checkpoint to
        # resume, so start the pipeline from the top.
        current_stage = st.session_state.get(stage_key, "start")
        if current_stage not in ("start", "awaiting_approval"):
            current_stage = "start"

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

                    # Get the full final state — the assemble node writes
                    # "dashboard" into it, so no second assembly is needed.
                    full_state = graph.get_state(thread_config).values
                    dashboard = full_state.get("dashboard") or assemble(
                        full_state
                    ).get("dashboard", {})

                    # Cache the result
                    st.session_state[result_key] = dashboard
                    st.session_state[stage_key] = "complete"

                    st.rerun()

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)

        else:
            # Fresh run — start the pipeline
            progress = st.empty()

            with progress.container():
                stage_progress("ingest")

            try:
                initial_state: dict = {"source_path": source_path}
                if selected_sheet:
                    initial_state["sheet_name"] = selected_sheet

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

        Upload a CSV or Excel file in the sidebar to get started. AUDITA will:

        1. **Ingest** your data — CSV with automatic encoding/delimiter
           detection, or any sheet of an Excel workbook
        2. **Audit** data quality (missing values, outliers, duplicates)
        3. **Propose** a cleaning plan (AI-powered, human-approved)
        4. **Clean** your data with full before/after tracking
        5. **Visualize** key insights with verified, trustworthy charts
        6. **Log** every action in an immutable audit trail

        ---

        > *"The LLM proposes. Code executes. Code verifies."*
        """
    )

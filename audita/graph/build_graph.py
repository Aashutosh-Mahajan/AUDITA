"""
Graph wiring — constructs the AUDITA LangGraph pipeline.

Linear edges:
  ingest → quality_audit → cleaning_plan → [human gate] →
  cleaning_exec → profiling → insight_planning →
  [fan-out to chart_builder] → self_check → [conditional: retry or continue] →
  assemble

Key mechanics:
- Fan-out: Send() from conditional edge after insight_planning
- Fan-in: completed_charts uses operator.add (see state.py)
- Retry loop: self_check routes failed charts back to chart_builder
- Human gate: interrupt_before=["cleaning_exec"]
"""

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from audita.core.schemas import VerificationStatus
from audita.graph.nodes.assemble import assemble
from audita.graph.nodes.chart_builder import chart_builder
from audita.graph.nodes.cleaning_exec import cleaning_exec
from audita.graph.nodes.cleaning_plan import cleaning_plan
from audita.graph.nodes.ingest import ingest
from audita.graph.nodes.insight_planning import insight_planning
from audita.graph.nodes.profiling import profiling
from audita.graph.nodes.quality_audit import quality_audit
from audita.graph.nodes.self_check import self_check
from audita.graph.state import PipelineState

# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _fan_out_charts(state: dict) -> list[Send]:
    """Fan-out: dispatch one chart_builder invocation per approved VizIntent."""
    intents = state.get("proposed_visualizations", [])
    cleaned_csv_path = state.get("cleaned_csv_path", "")

    sends = []
    for intent in intents:
        sends.append(
            Send(
                "chart_builder",
                {
                    "intent": intent,
                    "cleaned_csv_path": cleaned_csv_path,
                },
            )
        )

    return sends


def _check_retry_or_continue(
    state: dict,
) -> Literal["chart_builder_retry", "assemble"]:
    """After self_check: route back to chart_builder if any charts need retry."""
    completed_charts = state.get("completed_charts", [])

    retrying = [
        c
        for c in completed_charts
        if c.verification_status == VerificationStatus.RETRYING
    ]

    if retrying:
        return "chart_builder_retry"
    return "assemble"


def _fan_out_retries(state: dict) -> list[Send]:
    """Fan-out retries: re-dispatch only charts marked as RETRYING."""
    completed_charts = state.get("completed_charts", [])
    cleaned_csv_path = state.get("cleaned_csv_path", "")

    sends = []
    non_retrying = []

    for chart in completed_charts:
        if chart.verification_status == VerificationStatus.RETRYING:
            sends.append(
                Send(
                    "chart_builder",
                    {
                        "intent": chart.intent,
                        "cleaned_csv_path": cleaned_csv_path,
                    },
                )
            )
        else:
            non_retrying.append(chart)

    return sends


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(with_checkpointer: bool = True) -> Any:
    """Build and compile the AUDITA LangGraph pipeline.

    Args:
        with_checkpointer: If True, compile with MemorySaver for
            mid-pipeline resumability (needed for human-in-the-loop gate).

    Returns:
        Compiled LangGraph.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("ingest", ingest)
    graph.add_node("quality_audit", quality_audit)
    graph.add_node("cleaning_plan", cleaning_plan)
    graph.add_node("cleaning_exec", cleaning_exec)
    graph.add_node("profiling", profiling)
    graph.add_node("insight_planning", insight_planning)
    graph.add_node("chart_builder", chart_builder)
    graph.add_node("self_check", self_check)
    graph.add_node("assemble", assemble)

    # Linear edges (before fan-out)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "quality_audit")
    graph.add_edge("quality_audit", "cleaning_plan")
    # Human gate: graph pauses before cleaning_exec
    graph.add_edge("cleaning_plan", "cleaning_exec")
    graph.add_edge("cleaning_exec", "profiling")
    graph.add_edge("profiling", "insight_planning")

    # Fan-out: insight_planning → multiple chart_builder invocations
    graph.add_conditional_edges(
        "insight_planning",
        _fan_out_charts,
        ["chart_builder"],
    )

    # Fan-in: chart_builder → self_check
    graph.add_edge("chart_builder", "self_check")

    # Conditional: self_check → retry or assemble
    graph.add_conditional_edges(
        "self_check",
        _check_retry_or_continue,
        {
            "chart_builder_retry": "chart_builder",
            "assemble": "assemble",
        },
    )

    # Terminal edge
    graph.add_edge("assemble", END)

    # Compile
    compile_kwargs: dict[str, Any] = {}

    if with_checkpointer:
        checkpointer = MemorySaver()
        compile_kwargs["checkpointer"] = checkpointer

    # Human-in-the-loop gate: interrupt before cleaning_exec
    compile_kwargs["interrupt_before"] = ["cleaning_exec"]

    compiled = graph.compile(**compile_kwargs)
    return compiled

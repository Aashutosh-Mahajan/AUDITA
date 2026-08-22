"""
End-to-end graph tests.

Every bug these cover was invisible to the registry/schema unit tests: they
all lived in the wiring between nodes, so they only surface when the compiled
graph is actually driven through the human gate to a dashboard.
"""

import pandas as pd
import pytest

from audita.core.frame_io import read_frame
from audita.core.schemas import VerificationStatus
from audita.graph.build_graph import build_graph


def run_pipeline(source_path: str, sheet_name: str | None = None) -> dict:
    """Drive the graph the way app.py does: run to the gate, approve, resume."""
    graph = build_graph(with_checkpointer=True)
    config = {"configurable": {"thread_id": "test"}}

    initial: dict = {"source_path": source_path}
    if sheet_name:
        initial["sheet_name"] = sheet_name

    for _ in graph.stream(initial, config=config):
        pass

    snapshot = graph.get_state(config)
    assert snapshot.next == ("cleaning_exec",), "expected a pause at the human gate"

    # Approve the plan, then resume with input=None
    graph.update_state(
        config,
        {
            "cleaning_plan": snapshot.values["cleaning_plan"],
            "human_approved_cleaning": True,
        },
    )
    for _ in graph.stream(None, config=config):
        pass

    return graph.get_state(config).values


class TestHumanGate:
    def test_resumes_past_the_gate_instead_of_restarting(self, csv_file, stub_llm):
        """Regression: resuming with a state dict restarted the run from START,
        so the pipeline looped back to the gate and never produced anything."""
        state = run_pipeline(csv_file)

        assert state["completed_charts"], "pipeline produced no charts"
        assert state.get("cleaned_csv_path"), "cleaning_exec never ran"

    def test_pauses_before_cleaning_exec(self, csv_file, stub_llm):
        graph = build_graph(with_checkpointer=True)
        config = {"configurable": {"thread_id": "gate"}}
        for _ in graph.stream({"source_path": csv_file}, config=config):
            pass

        snapshot = graph.get_state(config)
        assert snapshot.next == ("cleaning_exec",)
        assert snapshot.values["human_approved_cleaning"] is False
        assert not snapshot.values.get("completed_charts")


class TestDashboardOutput:
    def test_dashboard_survives_in_graph_state(self, csv_file, stub_llm):
        """Regression: "dashboard" was not declared on PipelineState, so
        LangGraph silently dropped assemble's only output."""
        state = run_pipeline(csv_file)

        dashboard = state.get("dashboard")
        assert dashboard, "assemble's output was dropped from state"
        assert dashboard["charts_by_category"]
        assert dashboard["cleaning_diff_table"]
        assert dashboard["audit_log_table"]

    def test_every_chart_renders_a_figure(self, csv_file, stub_llm):
        state = run_pipeline(csv_file)

        for chart in state["completed_charts"]:
            assert chart.verification_status == VerificationStatus.VERIFIED, (
                f"{chart.intent.chart_type} was not verified: "
                f"{chart.verification_notes or chart.error}"
            )
            assert chart.figure_json, f"{chart.intent.chart_type} has no figure"

    def test_charts_are_not_duplicated(self, csv_file, stub_llm):
        """Regression: operator.add on completed_charts meant self_check
        appended a second copy of every chart it verified."""
        state = run_pipeline(csv_file)

        keys = [
            (c.intent.chart_type.value, tuple(c.intent.columns))
            for c in state["completed_charts"]
        ]
        assert len(keys) == len(set(keys)), f"duplicate charts: {keys}"
        assert len(keys) == len(stub_llm.viz_intents)
        assert state["dashboard"]["stats"]["total_charts"] == len(keys)

    def test_distinct_chart_types_on_one_column_both_survive(self, csv_file, stub_llm):
        """Regression: the dedupe key omitted chart_type, so the histogram and
        the box plot of "sales" collapsed into one."""
        state = run_pipeline(csv_file)

        sales_charts = {
            c.intent.chart_type.value
            for c in state["completed_charts"]
            if c.intent.columns == ["sales"]
        }
        assert sales_charts == {"histogram", "box"}


class TestRetryLoop:
    def test_ungrounded_chart_retries_then_flags(self, csv_file, stub_llm):
        """Regression: the retry branch used a plain edge into chart_builder,
        a Send() target, so every retry died with KeyError on 'intent'."""
        stub_llm.ungrounded = {"scatter"}
        state = run_pipeline(csv_file)

        by_type = {c.intent.chart_type.value: c for c in state["completed_charts"]}
        scatter = by_type["scatter"]

        assert scatter.verification_status == VerificationStatus.FLAGGED
        assert scatter.retry_count == 2, "retry_count did not accumulate across retries"
        # The rest of the run is unaffected and still completes
        assert by_type["histogram"].verification_status == VerificationStatus.VERIFIED
        assert state.get("dashboard"), "a flagged chart must not block assemble"


class TestExcelIngest:
    def test_reads_the_requested_sheet(self, xlsx_file, stub_llm):
        state = run_pipeline(xlsx_file, sheet_name="Data")

        columns = [c["name"] for c in state["raw_profile"]["columns"]]
        assert columns == ["region", "sales", "units", "date"]
        assert state["completed_charts"], "no charts from the Excel workbook"

    def test_defaults_to_the_first_sheet(self, xlsx_file):
        from audita.graph.nodes.ingest import ingest

        result = ingest({"source_path": xlsx_file})
        assert result["audit_log"][0].detail["sheet"] == "Cover"

    def test_unknown_sheet_falls_back_rather_than_failing(self, xlsx_file):
        from audita.graph.nodes.ingest import ingest

        result = ingest({"source_path": xlsx_file, "sheet_name": "Nope"})
        detail = result["audit_log"][0].detail
        assert detail["sheet"] == "Cover"
        assert detail["sheet_fallback"] is True

    def test_legacy_csv_path_key_still_accepted(self, csv_file):
        from audita.graph.nodes.ingest import ingest

        result = ingest({"csv_path": csv_file})
        assert result["raw_profile"]["n_cols"] == 4


class TestDtypePreservation:
    def test_parsed_dates_survive_the_handoff_to_charting(self, csv_file, stub_llm):
        """Regression: intermediates went through to_csv/read_csv, so the
        datetime column parse_dates produced reverted to object."""
        state = run_pipeline(csv_file)

        cleaned = read_frame(state["cleaned_csv_path"])
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


class TestCleaningRobustness:
    def test_action_on_a_dropped_column_does_not_abort_the_plan(self, csv_file):
        """Regression: the follow-up action raised KeyError inside pandas and
        took the whole node down, losing every other action too."""
        from audita.core.schemas import CleaningAction, CleaningActionType
        from audita.graph.nodes.cleaning_exec import cleaning_exec
        from audita.graph.nodes.ingest import ingest

        ingested = ingest({"source_path": csv_file})
        plan = [
            CleaningAction(
                column="units",
                action_type=CleaningActionType.DROP_COLUMN,
                rationale="x",
            ),
            CleaningAction(
                column="units",
                action_type=CleaningActionType.IMPUTE_MEAN,
                rationale="x",
            ),
            CleaningAction(
                column="sales",
                action_type=CleaningActionType.IMPUTE_MEDIAN,
                rationale="x",
            ),
        ]

        result = cleaning_exec(
            {"csv_path": ingested["csv_path"], "cleaning_plan": plan}
        )

        cleaned = read_frame(result["cleaned_csv_path"])
        assert "units" not in cleaned.columns
        assert cleaned["sales"].isna().sum() == 0, "later actions were lost"
        assert any(
            e.action == "skipped_failed_action" for e in result["audit_log"]
        ), "the skipped action was not recorded in the audit trail"


@pytest.mark.parametrize("model_env", ["", None])
def test_default_model_is_a_valid_id(monkeypatch, model_env):
    """Regression: the default was 'GPT-5.4-mini'; ids are case-sensitive."""
    from audita.core import llm_client

    if model_env is None:
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_MODEL", model_env)

    assert llm_client._get_model_name() == llm_client.DEFAULT_MODEL
    assert llm_client.DEFAULT_MODEL.lower() == llm_client.DEFAULT_MODEL

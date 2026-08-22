"""Shared fixtures for the graph-level tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A small dataset with every quirk the pipeline is meant to handle:
    missing numerics, near-duplicate category labels, and string dates.
    """
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "region": ["North", "South", "north ", "East"] * (n // 4),
            "sales": rng.normal(100, 20, n).round(2),
            "units": rng.integers(1, 50, n),
            "date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        }
    )
    df.loc[3:6, "sales"] = None
    return df


@pytest.fixture
def csv_file(tmp_path, sample_frame) -> str:
    path = tmp_path / "sample.csv"
    sample_frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def xlsx_file(tmp_path, sample_frame) -> str:
    """A two-sheet workbook; the target data is on the *second* sheet so a
    test that ignores sheet_name cannot accidentally pass.
    """
    pytest.importorskip("openpyxl")
    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"placeholder": [1, 2, 3]}).to_excel(
            writer, index=False, sheet_name="Cover"
        )
        sample_frame.to_excel(writer, index=False, sheet_name="Data")
    return str(path)


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace all three LLM calls with deterministic responses.

    Returns a control object so a test can override the grounding verdicts.
    """

    class Stub:
        cleaning_plan = [
            {"column": "sales", "action_type": "impute_median", "rationale": "gaps"},
            {
                "column": "region",
                "action_type": "standardize_categories",
                "rationale": "near-duplicate labels",
            },
            {"column": "date", "action_type": "parse_dates", "rationale": "strings"},
        ]
        viz_intents = [
            {
                "chart_type": "histogram",
                "columns": ["sales"],
                "rationale": "spread of sales",
                "priority_score": 4.0,
                "category": "distribution",
            },
            {
                "chart_type": "box",
                "columns": ["sales"],
                "rationale": "outliers in sales",
                "priority_score": 3.5,
                "category": "distribution",
            },
            {
                "chart_type": "bar",
                "columns": ["region", "sales"],
                "rationale": "sales by region",
                "priority_score": 4.5,
                "category": "categorical",
            },
            {
                "chart_type": "scatter",
                "columns": ["units", "sales"],
                "rationale": "units against sales",
                "priority_score": 3.0,
                "category": "relationship",
            },
        ]
        # Chart types whose grounding check should be forced to fail
        ungrounded: set[str] = set()

    stub = Stub()

    import audita.graph.nodes.cleaning_plan as cleaning_plan_node
    import audita.graph.nodes.insight_planning as insight_node
    import audita.graph.nodes.self_check as self_check_node

    monkeypatch.setattr(
        cleaning_plan_node,
        "request_cleaning_plan",
        lambda **_: list(stub.cleaning_plan),
    )
    monkeypatch.setattr(
        insight_node, "request_viz_intents", lambda **_: list(stub.viz_intents)
    )
    monkeypatch.setattr(
        self_check_node,
        "request_grounding_check",
        lambda summaries: [
            {
                "grounded": s["chart_type"] not in stub.ungrounded,
                "notes": "forced failure"
                if s["chart_type"] in stub.ungrounded
                else "ok",
            }
            for s in summaries
        ],
    )
    return stub

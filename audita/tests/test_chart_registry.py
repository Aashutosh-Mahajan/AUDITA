"""
Tests for the chart registry — each ChartType render function tested for
(a) successful figure generation on compatible dtypes, and
(b) raising DtypeCompatibilityError on incompatible dtypes.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from audita.core.chart_registry import (
    CHART_REGISTRY,
    DtypeCompatibilityError,
    render_chart,
    validate_chart_compatibility,
)
from audita.core.schemas import ChartType, VizIntent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixed_df() -> pd.DataFrame:
    """DataFrame with numeric, categorical, and date columns."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 28, 32, 45],
            "salary": [50000, 60000, 70000, 80000, 55000, 65000, 75000],
            "department": ["Eng", "HR", "Eng", "Sales", "HR", "Eng", "Sales"],
            "status": [
                "active",
                "active",
                "inactive",
                "active",
                "inactive",
                "active",
                "active",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                    "2024-07-01",
                ]
            ),
        }
    )


def _make_string_only_df() -> pd.DataFrame:
    """DataFrame with only string columns — no numeric at all."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "city": ["NYC", "LA", "Chicago", "NYC"],
        }
    )


def _intent(
    chart_type: ChartType, columns: list[str], category: str = "test"
) -> VizIntent:
    return VizIntent(
        chart_type=chart_type,
        columns=columns,
        rationale="Test intent",
        priority_score=3.0,
        category=category,
    )


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    def test_all_chart_types_registered(self):
        for chart_type in ChartType:
            assert chart_type in CHART_REGISTRY, f"Missing renderer for {chart_type}"


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_success_on_numeric(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.HISTOGRAM, ["age"], "distribution")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_fails_on_string_column(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.HISTOGRAM, ["department"], "distribution")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)


# ---------------------------------------------------------------------------
# Box plot tests
# ---------------------------------------------------------------------------


class TestBox:
    def test_success_single_numeric(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.BOX, ["salary"], "distribution")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_success_numeric_with_grouping(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.BOX, ["salary", "department"], "distribution")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_fails_on_string_primary(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.BOX, ["department"], "distribution")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)


# ---------------------------------------------------------------------------
# Bar chart tests
# ---------------------------------------------------------------------------


class TestBar:
    def test_success_categorical_counts(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.BAR, ["department"], "categorical")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_success_categorical_with_numeric(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.BAR, ["department", "salary"], "categorical")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_fails_on_numeric_first_column(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.BAR, ["age"], "categorical")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)


# ---------------------------------------------------------------------------
# Line chart tests
# ---------------------------------------------------------------------------


class TestLine:
    def test_success_date_x_numeric_y(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.LINE, ["date", "salary"], "trend")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_success_numeric_x_numeric_y(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.LINE, ["age", "salary"], "trend")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_fails_with_string_y(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.LINE, ["date", "department"], "trend")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)


# ---------------------------------------------------------------------------
# Scatter plot tests
# ---------------------------------------------------------------------------


class TestScatter:
    def test_success_two_numeric(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.SCATTER, ["age", "salary"], "relationship")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_fails_on_two_string_columns(self):
        df = _make_string_only_df()
        intent = _intent(ChartType.SCATTER, ["name", "city"], "relationship")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)

    def test_fails_with_one_column(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.SCATTER, ["age"], "relationship")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)


# ---------------------------------------------------------------------------
# Heatmap tests
# ---------------------------------------------------------------------------


class TestHeatmap:
    def test_success_multiple_numeric(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.HEATMAP, ["age", "salary"], "relationship")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_fails_with_string_column(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.HEATMAP, ["age", "department"], "relationship")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)

    def test_fails_with_single_column(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.HEATMAP, ["age"], "relationship")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)


# ---------------------------------------------------------------------------
# Pie chart tests
# ---------------------------------------------------------------------------


class TestPie:
    def test_success_categorical(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.PIE, ["department"], "categorical")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_success_categorical_with_numeric_values(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.PIE, ["department", "salary"], "categorical")
        fig = render_chart(df, intent)
        assert isinstance(fig, go.Figure)

    def test_fails_on_numeric_column(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.PIE, ["age"], "categorical")
        with pytest.raises(DtypeCompatibilityError):
            render_chart(df, intent)


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_column_raises_error(self):
        df = _make_mixed_df()
        intent = _intent(ChartType.HISTOGRAM, ["nonexistent"], "distribution")
        with pytest.raises(DtypeCompatibilityError, match="Columns not found"):
            validate_chart_compatibility(df, intent)

"""
Chart type registry — deterministic dispatch from ChartType enum to
hand-written Plotly render functions.

Also contains the dtype compatibility table used by insight_planning to
validate LLM-proposed visualization intents before they reach this registry.
"""

from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from audita.core.schemas import ChartType, VizIntent


# ---------------------------------------------------------------------------
# Dtype compatibility table (deterministic, not LLM)
# ---------------------------------------------------------------------------

def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


class DtypeCompatibilityError(Exception):
    """Raised when a chart type is incompatible with the given column dtypes."""
    pass


def validate_chart_compatibility(
    df: pd.DataFrame, intent: VizIntent
) -> None:
    """Validate that the intent's chart_type is compatible with the column dtypes.

    Raises ``DtypeCompatibilityError`` with a descriptive message on failure.
    """
    cols = intent.columns
    chart = intent.chart_type

    # Check all columns exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DtypeCompatibilityError(f"Columns not found in DataFrame: {missing}")

    if chart == ChartType.HISTOGRAM:
        if len(cols) < 1:
            raise DtypeCompatibilityError("Histogram requires at least 1 column")
        if not _is_numeric(df[cols[0]]):
            raise DtypeCompatibilityError(
                f"Histogram requires a numeric column, got dtype={df[cols[0]].dtype} for '{cols[0]}'"
            )

    elif chart == ChartType.BOX:
        if len(cols) < 1:
            raise DtypeCompatibilityError("Box plot requires at least 1 column")
        if not _is_numeric(df[cols[0]]):
            raise DtypeCompatibilityError(
                f"Box plot requires a numeric column, got dtype={df[cols[0]].dtype} for '{cols[0]}'"
            )

    elif chart == ChartType.SCATTER:
        if len(cols) < 2:
            raise DtypeCompatibilityError("Scatter plot requires 2 columns")
        if not _is_numeric(df[cols[0]]) or not _is_numeric(df[cols[1]]):
            raise DtypeCompatibilityError(
                f"Scatter plot requires 2 numeric columns, got dtypes="
                f"{df[cols[0]].dtype}, {df[cols[1]].dtype}"
            )

    elif chart == ChartType.HEATMAP:
        if len(cols) < 2:
            raise DtypeCompatibilityError("Heatmap requires at least 2 columns")
        non_numeric = [c for c in cols if not _is_numeric(df[c])]
        if non_numeric:
            raise DtypeCompatibilityError(
                f"Heatmap requires all numeric columns, non-numeric: {non_numeric}"
            )

    elif chart == ChartType.BAR:
        if len(cols) < 1:
            raise DtypeCompatibilityError("Bar chart requires at least 1 column")
        if not _is_categorical(df[cols[0]]):
            raise DtypeCompatibilityError(
                f"Bar chart requires a categorical first column, got dtype={df[cols[0]].dtype}"
            )

    elif chart == ChartType.PIE:
        if len(cols) < 1:
            raise DtypeCompatibilityError("Pie chart requires at least 1 column")
        if not _is_categorical(df[cols[0]]):
            raise DtypeCompatibilityError(
                f"Pie chart requires a categorical column, got dtype={df[cols[0]].dtype}"
            )

    elif chart == ChartType.LINE:
        if len(cols) < 2:
            raise DtypeCompatibilityError("Line chart requires at least 2 columns (x, y)")
        # x can be date, numeric, or ordinal; y must be numeric
        x_ok = _is_numeric(df[cols[0]]) or _is_datetime(df[cols[0]]) or _is_categorical(df[cols[0]])
        if not x_ok:
            raise DtypeCompatibilityError(
                f"Line chart x-axis must be numeric, datetime, or ordinal, got dtype={df[cols[0]].dtype}"
            )
        if not _is_numeric(df[cols[1]]):
            raise DtypeCompatibilityError(
                f"Line chart y-axis must be numeric, got dtype={df[cols[1]].dtype}"
            )


# ---------------------------------------------------------------------------
# Render functions — one per ChartType
# ---------------------------------------------------------------------------

def render_histogram(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Render a histogram for a single numeric column."""
    validate_chart_compatibility(df, intent)
    col = intent.columns[0]
    fig = px.histogram(
        df, x=col,
        title=f"Distribution of {col}",
        labels={col: col},
    )
    fig.update_layout(bargap=0.05)
    return fig


def render_box(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Render a box plot. If 2 columns provided, uses first as grouping."""
    validate_chart_compatibility(df, intent)
    cols = intent.columns
    if len(cols) >= 2 and _is_categorical(df[cols[1]]):
        fig = px.box(
            df, x=cols[1], y=cols[0],
            title=f"Box Plot of {cols[0]} by {cols[1]}",
        )
    else:
        fig = px.box(
            df, y=cols[0],
            title=f"Box Plot of {cols[0]}",
        )
    return fig


def render_bar(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Render a bar chart — categorical x, optional numeric y (defaults to count)."""
    validate_chart_compatibility(df, intent)
    cols = intent.columns
    cat_col = cols[0]

    if len(cols) >= 2 and _is_numeric(df[cols[1]]):
        # Aggregate: mean of numeric grouped by categorical
        agg = df.groupby(cat_col, observed=True)[cols[1]].mean().reset_index()
        fig = px.bar(
            agg, x=cat_col, y=cols[1],
            title=f"Mean {cols[1]} by {cat_col}",
        )
    else:
        # Value counts
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        fig = px.bar(
            counts, x=cat_col, y="count",
            title=f"Count by {cat_col}",
        )
    return fig


def render_line(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Render a line chart — x (ordinal/date/numeric), y (numeric)."""
    validate_chart_compatibility(df, intent)
    cols = intent.columns
    sorted_df = df.sort_values(cols[0])
    fig = px.line(
        sorted_df, x=cols[0], y=cols[1],
        title=f"{cols[1]} over {cols[0]}",
    )
    return fig


def render_scatter(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Render a scatter plot — 2 numeric columns."""
    validate_chart_compatibility(df, intent)
    cols = intent.columns
    fig = px.scatter(
        df, x=cols[0], y=cols[1],
        title=f"{cols[1]} vs {cols[0]}",
    )
    return fig


def render_heatmap(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Render a correlation heatmap of the specified numeric columns."""
    validate_chart_compatibility(df, intent)
    corr = df[intent.columns].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    return fig


def render_pie(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Render a pie chart — categorical column (value counts or with numeric values)."""
    validate_chart_compatibility(df, intent)
    cols = intent.columns
    cat_col = cols[0]

    if len(cols) >= 2 and _is_numeric(df[cols[1]]):
        agg = df.groupby(cat_col, observed=True)[cols[1]].sum().reset_index()
        fig = px.pie(
            agg, names=cat_col, values=cols[1],
            title=f"{cols[1]} by {cat_col}",
        )
    else:
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        fig = px.pie(
            counts, names=cat_col, values="count",
            title=f"Distribution of {cat_col}",
        )
    return fig


# ---------------------------------------------------------------------------
# Registry dispatch dict
# ---------------------------------------------------------------------------

CHART_REGISTRY: dict[ChartType, Callable[[pd.DataFrame, VizIntent], go.Figure]] = {
    ChartType.HISTOGRAM: render_histogram,
    ChartType.BOX: render_box,
    ChartType.BAR: render_bar,
    ChartType.LINE: render_line,
    ChartType.SCATTER: render_scatter,
    ChartType.HEATMAP: render_heatmap,
    ChartType.PIE: render_pie,
}


def render_chart(df: pd.DataFrame, intent: VizIntent) -> go.Figure:
    """Look up and execute a chart render function from the registry.

    Raises ``KeyError`` if the chart type is not registered.
    Raises ``DtypeCompatibilityError`` if dtypes are incompatible.
    """
    handler = CHART_REGISTRY[intent.chart_type]
    return handler(df, intent)

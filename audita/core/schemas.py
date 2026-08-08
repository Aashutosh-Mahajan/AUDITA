"""
Pydantic schemas for all AUDITA pipeline data models.

Every LLM structured-output call must return objects validated against
CleaningAction / VizIntent. All execution is deterministic, hand-written
functions dispatched from a fixed action/chart-type registry.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Cleaning schemas
# ---------------------------------------------------------------------------


class CleaningActionType(str, Enum):
    """Fixed set of cleaning actions the LLM may propose."""

    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    DROP_ROWS = "drop_rows"
    DROP_COLUMN = "drop_column"
    STANDARDIZE_CATEGORIES = "standardize_categories"
    PARSE_DATES = "parse_dates"
    CAP_OUTLIERS = "cap_outliers"
    NO_ACTION = "no_action"


class CleaningAction(BaseModel):
    """A single cleaning action proposed by the LLM for a specific column."""

    column: str
    action_type: CleaningActionType
    rationale: str
    params: dict[str, Any] = Field(
        default_factory=dict
    )  # e.g. {"cap_percentile": 0.99}


class CleaningDiffEntry(BaseModel):
    """Before/after snapshot for one cleaning action's effect on a column."""

    column: str
    action_type: CleaningActionType
    rows_affected: int
    before_stat: dict[str, Any]
    after_stat: dict[str, Any]


# ---------------------------------------------------------------------------
# Visualization schemas
# ---------------------------------------------------------------------------


class ChartType(str, Enum):
    """Fixed set of chart types the LLM may propose."""

    HISTOGRAM = "histogram"
    BOX = "box"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"


class VizIntent(BaseModel):
    """A single visualization intent proposed by the LLM."""

    chart_type: ChartType
    columns: list[str]
    rationale: str
    priority_score: float = Field(ge=0, le=5)
    category: str  # "distribution" | "relationship" | "trend" | "categorical"


# ---------------------------------------------------------------------------
# Verification schemas
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """Outcome of the self-check stage for a rendered chart."""

    VERIFIED = "verified"
    FLAGGED = "flagged"
    RETRYING = "retrying"
    FAILED = "failed"


class ChartResult(BaseModel):
    """Result of rendering + verifying a single chart."""

    intent: VizIntent
    figure_json: str | None = None  # Plotly figure.to_json()
    verification_status: VerificationStatus
    verification_notes: str = ""
    retry_count: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Audit log schema
# ---------------------------------------------------------------------------


class AuditLogEntry(BaseModel):
    """One entry in the append-only audit trail."""

    timestamp: str
    stage: str
    actor: str  # "llm" | "code"
    action: str
    detail: dict[str, Any]

"""
Tests for Pydantic schema validation — ensures enum membership,
range constraints, and required-field enforcement.
"""

import pytest
from pydantic import ValidationError

from audita.core.schemas import (
    CleaningActionType,
    CleaningAction,
    CleaningDiffEntry,
    ChartType,
    VizIntent,
    VerificationStatus,
    ChartResult,
    AuditLogEntry,
)


# ---------------------------------------------------------------------------
# Enum membership tests
# ---------------------------------------------------------------------------

class TestCleaningActionTypeEnum:
    def test_all_members_present(self):
        expected = {
            "impute_mean", "impute_median", "impute_mode",
            "drop_rows", "drop_column", "standardize_categories",
            "parse_dates", "cap_outliers", "no_action",
        }
        actual = {e.value for e in CleaningActionType}
        assert actual == expected

    def test_invalid_value_rejected(self):
        with pytest.raises(ValueError):
            CleaningActionType("delete_everything")


class TestChartTypeEnum:
    def test_all_members_present(self):
        expected = {"histogram", "box", "bar", "line", "scatter", "heatmap", "pie"}
        actual = {e.value for e in ChartType}
        assert actual == expected

    def test_invalid_value_rejected(self):
        with pytest.raises(ValueError):
            ChartType("3d_globe")


class TestVerificationStatusEnum:
    def test_all_members_present(self):
        expected = {"verified", "flagged", "retrying", "failed"}
        actual = {e.value for e in VerificationStatus}
        assert actual == expected

    def test_invalid_value_rejected(self):
        with pytest.raises(ValueError):
            VerificationStatus("unknown")


# ---------------------------------------------------------------------------
# CleaningAction model tests
# ---------------------------------------------------------------------------

class TestCleaningAction:
    def test_valid_construction(self):
        action = CleaningAction(
            column="age",
            action_type=CleaningActionType.IMPUTE_MEAN,
            rationale="Fill missing age values with mean",
        )
        assert action.column == "age"
        assert action.action_type == CleaningActionType.IMPUTE_MEAN
        assert action.params == {}

    def test_with_params(self):
        action = CleaningAction(
            column="salary",
            action_type=CleaningActionType.CAP_OUTLIERS,
            rationale="Cap extreme salary values",
            params={"cap_percentile": 0.99},
        )
        assert action.params["cap_percentile"] == 0.99

    def test_missing_required_field_column(self):
        with pytest.raises(ValidationError):
            CleaningAction(
                action_type=CleaningActionType.DROP_ROWS,
                rationale="Remove incomplete rows",
            )

    def test_missing_required_field_rationale(self):
        with pytest.raises(ValidationError):
            CleaningAction(
                column="age",
                action_type=CleaningActionType.DROP_ROWS,
            )

    def test_invalid_action_type(self):
        with pytest.raises(ValidationError):
            CleaningAction(
                column="age",
                action_type="magic_fix",
                rationale="Does not exist",
            )


# ---------------------------------------------------------------------------
# CleaningDiffEntry model tests
# ---------------------------------------------------------------------------

class TestCleaningDiffEntry:
    def test_valid_construction(self):
        entry = CleaningDiffEntry(
            column="age",
            action_type=CleaningActionType.IMPUTE_MEAN,
            rows_affected=15,
            before_stat={"missing_pct": 0.10, "mean": 34.2},
            after_stat={"missing_pct": 0.0, "mean": 34.5},
        )
        assert entry.rows_affected == 15
        assert entry.before_stat["missing_pct"] == 0.10

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            CleaningDiffEntry(
                column="age",
                action_type=CleaningActionType.IMPUTE_MEAN,
                # rows_affected missing
                before_stat={},
                after_stat={},
            )


# ---------------------------------------------------------------------------
# VizIntent model tests
# ---------------------------------------------------------------------------

class TestVizIntent:
    def test_valid_construction(self):
        intent = VizIntent(
            chart_type=ChartType.HISTOGRAM,
            columns=["age"],
            rationale="Show age distribution",
            priority_score=4.0,
            category="distribution",
        )
        assert intent.chart_type == ChartType.HISTOGRAM
        assert intent.priority_score == 4.0

    def test_priority_score_lower_bound(self):
        """priority_score must be >= 0."""
        with pytest.raises(ValidationError):
            VizIntent(
                chart_type=ChartType.BAR,
                columns=["category"],
                rationale="Show counts",
                priority_score=-0.1,
                category="categorical",
            )

    def test_priority_score_upper_bound(self):
        """priority_score must be <= 5."""
        with pytest.raises(ValidationError):
            VizIntent(
                chart_type=ChartType.BAR,
                columns=["category"],
                rationale="Show counts",
                priority_score=5.1,
                category="categorical",
            )

    def test_priority_score_zero_is_valid(self):
        intent = VizIntent(
            chart_type=ChartType.PIE,
            columns=["status"],
            rationale="Low priority",
            priority_score=0.0,
            category="categorical",
        )
        assert intent.priority_score == 0.0

    def test_priority_score_five_is_valid(self):
        intent = VizIntent(
            chart_type=ChartType.SCATTER,
            columns=["x", "y"],
            rationale="High priority",
            priority_score=5.0,
            category="relationship",
        )
        assert intent.priority_score == 5.0

    def test_invalid_chart_type(self):
        with pytest.raises(ValidationError):
            VizIntent(
                chart_type="treemap",
                columns=["a"],
                rationale="Nope",
                priority_score=3.0,
                category="distribution",
            )

    def test_missing_columns(self):
        with pytest.raises(ValidationError):
            VizIntent(
                chart_type=ChartType.BAR,
                rationale="No columns",
                priority_score=3.0,
                category="categorical",
            )


# ---------------------------------------------------------------------------
# ChartResult model tests
# ---------------------------------------------------------------------------

class TestChartResult:
    def test_valid_verified_chart(self):
        intent = VizIntent(
            chart_type=ChartType.HISTOGRAM,
            columns=["age"],
            rationale="Distribution of age",
            priority_score=4.5,
            category="distribution",
        )
        result = ChartResult(
            intent=intent,
            figure_json='{"data": []}',
            verification_status=VerificationStatus.VERIFIED,
        )
        assert result.verification_status == VerificationStatus.VERIFIED
        assert result.retry_count == 0
        assert result.error is None

    def test_failed_chart_with_error(self):
        intent = VizIntent(
            chart_type=ChartType.SCATTER,
            columns=["a", "b"],
            rationale="Relationship",
            priority_score=3.0,
            category="relationship",
        )
        result = ChartResult(
            intent=intent,
            verification_status=VerificationStatus.FAILED,
            error="Incompatible dtypes",
            retry_count=2,
        )
        assert result.verification_status == VerificationStatus.FAILED
        assert result.error == "Incompatible dtypes"
        assert result.figure_json is None

    def test_missing_intent(self):
        with pytest.raises(ValidationError):
            ChartResult(
                verification_status=VerificationStatus.VERIFIED,
            )


# ---------------------------------------------------------------------------
# AuditLogEntry model tests
# ---------------------------------------------------------------------------

class TestAuditLogEntry:
    def test_valid_construction(self):
        entry = AuditLogEntry(
            timestamp="2026-01-01T00:00:00Z",
            stage="ingest",
            actor="code",
            action="loaded_csv",
            detail={"rows": 1000, "cols": 10},
        )
        assert entry.stage == "ingest"
        assert entry.actor == "code"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AuditLogEntry(
                timestamp="2026-01-01T00:00:00Z",
                stage="ingest",
                # actor missing
                action="loaded_csv",
                detail={},
            )

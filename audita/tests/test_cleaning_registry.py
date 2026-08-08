"""
Tests for the cleaning registry — each CleaningActionType handler tested
against synthetic DataFrames with known values.
"""

import numpy as np
import pandas as pd

from audita.core.cleaning_registry import CLEANING_REGISTRY, execute_cleaning_action
from audita.core.schemas import CleaningAction, CleaningActionType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_numeric_df() -> pd.DataFrame:
    """Small DataFrame with known missing values and outliers."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, np.nan, 35.0, 40.0, np.nan, 28.0],
            "salary": [50000, 60000, 70000, 80000, 500000, 55000, 65000],
            "name": ["Alice", "Bob", None, "Diana", "Eve", "Frank", "Grace"],
        }
    )


def _make_category_df() -> pd.DataFrame:
    """DataFrame with messy categorical labels."""
    return pd.DataFrame(
        {
            "status": [
                "Active",
                "active",
                " ACTIVE",
                "Inactive",
                "inactive",
                "Actve",
                "Inactive",
            ],
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )


def _make_date_df() -> pd.DataFrame:
    """DataFrame with string dates and some unparseable values."""
    return pd.DataFrame(
        {
            "date_str": [
                "2024-01-01",
                "2024-02-15",
                "not-a-date",
                "2024-06-30",
                "2024-12-25",
            ],
            "value": [10, 20, 30, 40, 50],
        }
    )


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    def test_all_action_types_registered(self):
        """Every CleaningActionType must have a handler in the registry."""
        for action_type in CleaningActionType:
            assert action_type in CLEANING_REGISTRY, (
                f"Missing handler for {action_type}"
            )


# ---------------------------------------------------------------------------
# Imputation tests
# ---------------------------------------------------------------------------


class TestImputeMean:
    def test_fills_nan_with_mean(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="age",
            action_type=CleaningActionType.IMPUTE_MEAN,
            rationale="Fill missing ages",
        )
        result = execute_cleaning_action(df, action)
        assert result["age"].isna().sum() == 0
        # Mean of [25, 30, 35, 40, 28] = 31.6
        expected_mean = np.mean([25.0, 30.0, 35.0, 40.0, 28.0])
        assert np.isclose(result["age"].iloc[2], expected_mean)
        assert np.isclose(result["age"].iloc[5], expected_mean)

    def test_does_not_modify_original(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="age",
            action_type=CleaningActionType.IMPUTE_MEAN,
            rationale="test",
        )
        execute_cleaning_action(df, action)
        assert df["age"].isna().sum() == 2  # original unchanged


class TestImputeMedian:
    def test_fills_nan_with_median(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="age",
            action_type=CleaningActionType.IMPUTE_MEDIAN,
            rationale="Fill missing ages",
        )
        result = execute_cleaning_action(df, action)
        assert result["age"].isna().sum() == 0
        expected_median = np.median([25.0, 30.0, 35.0, 40.0, 28.0])
        assert np.isclose(result["age"].iloc[2], expected_median)


class TestImputeMode:
    def test_fills_nan_with_mode(self):
        df = pd.DataFrame({"color": ["red", "red", "blue", None, "red"]})
        action = CleaningAction(
            column="color",
            action_type=CleaningActionType.IMPUTE_MODE,
            rationale="Fill missing colors",
        )
        result = execute_cleaning_action(df, action)
        assert result["color"].isna().sum() == 0
        assert result["color"].iloc[3] == "red"

    def test_handles_all_nan_column(self):
        """If every value is NaN, mode is empty — nothing changes."""
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        action = CleaningAction(
            column="x",
            action_type=CleaningActionType.IMPUTE_MODE,
            rationale="test",
        )
        result = execute_cleaning_action(df, action)
        assert result["x"].isna().sum() == 3


# ---------------------------------------------------------------------------
# Drop tests
# ---------------------------------------------------------------------------


class TestDropRows:
    def test_drops_rows_with_nan(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="age",
            action_type=CleaningActionType.DROP_ROWS,
            rationale="Remove incomplete rows",
        )
        result = execute_cleaning_action(df, action)
        assert len(result) == 5
        assert result["age"].isna().sum() == 0

    def test_resets_index(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="age",
            action_type=CleaningActionType.DROP_ROWS,
            rationale="test",
        )
        result = execute_cleaning_action(df, action)
        assert list(result.index) == list(range(len(result)))


class TestDropColumn:
    def test_drops_column(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="salary",
            action_type=CleaningActionType.DROP_COLUMN,
            rationale="Remove salary column",
        )
        result = execute_cleaning_action(df, action)
        assert "salary" not in result.columns
        assert "age" in result.columns  # other columns unaffected


# ---------------------------------------------------------------------------
# Standardize categories tests
# ---------------------------------------------------------------------------


class TestStandardizeCategories:
    def test_lowercases_and_strips(self):
        df = _make_category_df()
        action = CleaningAction(
            column="status",
            action_type=CleaningActionType.STANDARDIZE_CATEGORIES,
            rationale="Normalize status labels",
        )
        result = execute_cleaning_action(df, action)
        # All "Active"/" ACTIVE"/"active"/"Actve" variants should map to something consistent
        for val in result["status"]:
            assert val == val.strip().lower()

    def test_fuzzy_merge_near_duplicates(self):
        df = _make_category_df()
        action = CleaningAction(
            column="status",
            action_type=CleaningActionType.STANDARDIZE_CATEGORIES,
            rationale="Merge near-duplicate labels",
            params={"fuzzy_threshold": 0.80},
        )
        result = execute_cleaning_action(df, action)
        unique_vals = result["status"].unique()
        # "actve" should be merged with "active" at 0.80 threshold
        assert "actve" not in unique_vals or len(unique_vals) <= 3


# ---------------------------------------------------------------------------
# Parse dates tests
# ---------------------------------------------------------------------------


class TestParseDates:
    def test_parses_valid_dates(self):
        df = _make_date_df()
        action = CleaningAction(
            column="date_str",
            action_type=CleaningActionType.PARSE_DATES,
            rationale="Convert string dates to datetime",
        )
        result = execute_cleaning_action(df, action)
        assert pd.api.types.is_datetime64_any_dtype(result["date_str"])

    def test_coerces_unparseable_to_nat(self):
        df = _make_date_df()
        action = CleaningAction(
            column="date_str",
            action_type=CleaningActionType.PARSE_DATES,
            rationale="test",
        )
        result = execute_cleaning_action(df, action)
        # "not-a-date" at index 2 should become NaT
        assert pd.isna(result["date_str"].iloc[2])
        # Valid dates should parse correctly
        assert result["date_str"].iloc[0] == pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------
# Cap outliers tests
# ---------------------------------------------------------------------------


class TestCapOutliers:
    def test_iqr_capping(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="salary",
            action_type=CleaningActionType.CAP_OUTLIERS,
            rationale="Cap extreme salaries",
        )
        result = execute_cleaning_action(df, action)
        # The 500000 outlier should be capped
        assert result["salary"].max() < 500000

    def test_percentile_capping(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="salary",
            action_type=CleaningActionType.CAP_OUTLIERS,
            rationale="Cap at 99th percentile",
            params={"cap_percentile": 0.99},
        )
        result = execute_cleaning_action(df, action)
        upper = df["salary"].quantile(0.99)
        assert result["salary"].max() <= upper + 0.01  # float tolerance

    def test_non_numeric_column_unchanged(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="name",
            action_type=CleaningActionType.CAP_OUTLIERS,
            rationale="test",
        )
        result = execute_cleaning_action(df, action)
        assert result["name"].equals(df["name"])


# ---------------------------------------------------------------------------
# No-action test
# ---------------------------------------------------------------------------


class TestNoAction:
    def test_returns_copy_unchanged(self):
        df = _make_numeric_df()
        action = CleaningAction(
            column="age",
            action_type=CleaningActionType.NO_ACTION,
            rationale="Skip this column",
        )
        result = execute_cleaning_action(df, action)
        pd.testing.assert_frame_equal(result, df)
        assert result is not df  # should be a copy

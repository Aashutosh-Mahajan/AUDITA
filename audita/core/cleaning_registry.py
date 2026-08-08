"""
Cleaning action registry — deterministic dispatch from CleaningActionType enum
to hand-written pandas functions.

The LLM proposes actions; this registry executes them. No LLM-generated code
is ever run. Each handler takes (df, action) and returns a (possibly modified)
copy of the DataFrame.
"""

from typing import Callable

import numpy as np
import pandas as pd

from audita.core.schemas import CleaningAction, CleaningActionType


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------

def _impute_mean(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Replace NaN values in a numeric column with the column mean."""
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[action.column]):
        df[action.column] = df[action.column].fillna(df[action.column].mean())
    return df


def _impute_median(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Replace NaN values in a numeric column with the column median."""
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[action.column]):
        df[action.column] = df[action.column].fillna(df[action.column].median())
    return df


def _impute_mode(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Replace NaN values with the column mode (works for any dtype)."""
    df = df.copy()
    mode_values = df[action.column].mode()
    if len(mode_values) > 0:
        df[action.column] = df[action.column].fillna(mode_values.iloc[0])
    return df


def _drop_rows(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Drop rows where the target column has NaN values."""
    df = df.copy()
    df = df.dropna(subset=[action.column]).reset_index(drop=True)
    return df


def _drop_column(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Drop the target column entirely."""
    df = df.copy()
    df = df.drop(columns=[action.column])
    return df


def _standardize_categories(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Lowercase, strip whitespace, and optionally merge near-duplicate labels.

    Uses rapidfuzz for fuzzy matching when available; falls back to exact
    lowercased matching only.
    """
    df = df.copy()
    col = action.column

    # Step 1: lowercase + strip
    df[col] = df[col].astype(str).str.strip().str.lower()

    # Step 2: fuzzy merge near-duplicates
    threshold = action.params.get("fuzzy_threshold", 0.85)
    try:
        from rapidfuzz import fuzz
        unique_vals = df[col].unique().tolist()
        # Build a mapping from near-duplicate → canonical (first occurrence wins)
        mapping: dict[str, str] = {}
        for val in unique_vals:
            if val in mapping:
                continue
            for other in unique_vals:
                if other == val or other in mapping:
                    continue
                similarity = fuzz.ratio(val, other) / 100.0
                if similarity >= threshold:
                    mapping[other] = val
        if mapping:
            df[col] = df[col].replace(mapping)
    except ImportError:
        pass  # rapidfuzz not available — skip fuzzy merge

    return df


def _parse_dates(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Parse a string column into datetime, coercing unparseable values to NaT."""
    df = df.copy()
    date_format = action.params.get("format", None)
    df[action.column] = pd.to_datetime(
        df[action.column], format=date_format, errors="coerce"
    )
    return df


def _cap_outliers(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Cap outliers using IQR-based or percentile-based capping.

    If ``params["cap_percentile"]`` is provided, uses that percentile for
    both lower (1 - p) and upper (p) bounds. Otherwise defaults to 1.5×IQR.
    """
    df = df.copy()
    col = action.column

    if not pd.api.types.is_numeric_dtype(df[col]):
        return df

    cap_percentile = action.params.get("cap_percentile", None)

    if cap_percentile is not None:
        lower = df[col].quantile(1 - cap_percentile)
        upper = df[col].quantile(cap_percentile)
    else:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _no_action(df: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    """Identity pass-through — no changes made."""
    return df.copy()


# ---------------------------------------------------------------------------
# Registry dispatch dict
# ---------------------------------------------------------------------------

CLEANING_REGISTRY: dict[
    CleaningActionType, Callable[[pd.DataFrame, CleaningAction], pd.DataFrame]
] = {
    CleaningActionType.IMPUTE_MEAN: _impute_mean,
    CleaningActionType.IMPUTE_MEDIAN: _impute_median,
    CleaningActionType.IMPUTE_MODE: _impute_mode,
    CleaningActionType.DROP_ROWS: _drop_rows,
    CleaningActionType.DROP_COLUMN: _drop_column,
    CleaningActionType.STANDARDIZE_CATEGORIES: _standardize_categories,
    CleaningActionType.PARSE_DATES: _parse_dates,
    CleaningActionType.CAP_OUTLIERS: _cap_outliers,
    CleaningActionType.NO_ACTION: _no_action,
}


def execute_cleaning_action(
    df: pd.DataFrame, action: CleaningAction
) -> pd.DataFrame:
    """Look up and execute a cleaning action from the registry.

    Raises ``KeyError`` if the action type is not registered.
    """
    handler = CLEANING_REGISTRY[action.action_type]
    return handler(df, action)

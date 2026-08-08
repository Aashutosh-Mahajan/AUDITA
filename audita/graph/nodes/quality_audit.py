"""
Quality audit node — pure code, no LLM.

Computes per-column quality metrics that get fed to the LLM in the
cleaning-plan stage.  This same function is reused by ``profiling.py``
to produce the post-cleaning profile so before/after are directly comparable.
"""

from typing import Any

import pandas as pd

from audita.core.audit_log import log_code_action


def compute_column_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-column quality metrics.

    Returns a JSON-serializable dict keyed by column name.
    """
    audit: dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_pct": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            # Numeric column stats
            desc = series.describe()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())

            col_info.update(
                {
                    "mean": round(float(desc.get("mean", 0)), 4),
                    "std": round(float(desc.get("std", 0)), 4),
                    "min": float(desc.get("min", 0)),
                    "max": float(desc.get("max", 0)),
                    "iqr_outlier_count": outlier_count,
                }
            )

        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        ):
            # String/categorical column stats
            non_null = series.dropna()

            # Top value counts (up to 10)
            top_counts = non_null.value_counts().head(10)
            col_info["top_values"] = {str(k): int(v) for k, v in top_counts.items()}

            # Fuzzy near-duplicate label detection
            unique_vals = [str(v) for v in non_null.unique()]
            near_duplicates = _find_near_duplicates(unique_vals, threshold=0.85)
            if near_duplicates:
                col_info["near_duplicate_labels"] = near_duplicates

            # Date-parse success rate for string columns
            date_success_rate = _date_parse_success_rate(non_null)
            if date_success_rate > 0.5:
                col_info["date_parse_success_rate"] = round(date_success_rate, 4)
                col_info["looks_like_date"] = True

        audit[col] = col_info

    return audit


def _find_near_duplicates(
    values: list[str], threshold: float = 0.85
) -> list[dict[str, Any]]:
    """Find fuzzy near-duplicate pairs among unique values.

    Uses rapidfuzz if available; falls back to difflib.SequenceMatcher.
    """
    pairs: list[dict[str, Any]] = []

    try:
        from rapidfuzz import fuzz

        for i, a in enumerate(values):
            for b in values[i + 1 :]:
                similarity = fuzz.ratio(a.lower(), b.lower()) / 100.0
                if similarity >= threshold and a.lower() != b.lower():
                    pairs.append(
                        {
                            "value_a": a,
                            "value_b": b,
                            "similarity": round(similarity, 3),
                        }
                    )
    except ImportError:
        from difflib import SequenceMatcher

        for i, a in enumerate(values):
            for b in values[i + 1 :]:
                similarity = SequenceMatcher(None, a.lower(), b.lower()).ratio()
                if similarity >= threshold and a.lower() != b.lower():
                    pairs.append(
                        {
                            "value_a": a,
                            "value_b": b,
                            "similarity": round(similarity, 3),
                        }
                    )

    return pairs


def _date_parse_success_rate(series: pd.Series) -> float:
    """Attempt to parse a string Series as dates; return success fraction."""
    if len(series) == 0:
        return 0.0
    parsed = pd.to_datetime(series, errors="coerce")
    success_count = parsed.notna().sum()
    return float(success_count / len(series))


def quality_audit(state: dict) -> dict:
    """LangGraph node: run quality audit on the raw DataFrame.

    Reads the CSV from ``state["csv_path"]``, computes per-column quality
    metrics, and returns the ``quality_audit`` dict.
    """
    csv_path: str = state["csv_path"]
    df = pd.read_csv(csv_path)

    audit_result = compute_column_audit(df)

    audit_entry = log_code_action(
        stage="quality_audit",
        action="computed_quality_metrics",
        detail={
            "columns_audited": len(audit_result),
            "columns_with_issues": sum(
                1
                for col_info in audit_result.values()
                if col_info.get("missing_pct", 0) > 0
                or col_info.get("iqr_outlier_count", 0) > 0
                or col_info.get("near_duplicate_labels")
                or col_info.get("looks_like_date")
            ),
        },
    )

    return {
        "quality_audit": audit_result,
        "audit_log": [audit_entry],
    }

"""
Ingest node — reads a CSV file with auto-delimiter and encoding detection,
produces a raw profile dict.
"""

import csv
import os
import tempfile
from io import StringIO
from typing import Any

import chardet
import pandas as pd

from audita.core.audit_log import log_code_action
from audita.core.schemas import AuditLogEntry


def _detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet; fall back to utf-8."""
    with open(file_path, "rb") as f:
        raw = f.read(10_000)  # sample first 10KB
    result = chardet.detect(raw)
    encoding = result.get("encoding", "utf-8") or "utf-8"
    return encoding


def _detect_delimiter(file_path: str, encoding: str) -> str:
    """Use csv.Sniffer to detect the delimiter; fall back to comma."""
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(8_192)
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except csv.Error:
        return ","


def _build_raw_profile(df: pd.DataFrame, file_path: str) -> dict[str, Any]:
    """Build the raw_profile dict from the ingested DataFrame."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "file_size_bytes": os.path.getsize(file_path),
        "columns": [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in df.columns
        ],
    }


def ingest(state: dict) -> dict:
    """LangGraph node: ingest a CSV file and produce raw_profile.

    Reads ``state["csv_path"]``, returns updates to state including the
    raw profile and an audit log entry.
    """
    csv_path: str = state["csv_path"]

    # Detect encoding (fall back to utf-8)
    try:
        encoding = _detect_encoding(csv_path)
        # Verify the encoding works
        pd.read_csv(csv_path, encoding=encoding, nrows=0)
    except (UnicodeDecodeError, LookupError):
        encoding = "utf-8"

    # Detect delimiter
    delimiter = _detect_delimiter(csv_path, encoding)

    # Read the CSV
    df = pd.read_csv(csv_path, encoding=encoding, delimiter=delimiter)

    # Save to a temp path for downstream nodes (clean copy)
    temp_dir = tempfile.mkdtemp(prefix="audita_")
    raw_csv_path = os.path.join(temp_dir, "raw_data.csv")
    df.to_csv(raw_csv_path, index=False)

    # Build raw profile
    raw_profile = _build_raw_profile(df, csv_path)

    # Audit log entry
    audit_entry = log_code_action(
        stage="ingest",
        action="loaded_csv",
        detail={
            "file": os.path.basename(csv_path),
            "encoding": encoding,
            "delimiter": repr(delimiter),
            "rows": raw_profile["n_rows"],
            "cols": raw_profile["n_cols"],
        },
    )

    return {
        "csv_path": raw_csv_path,
        "raw_profile": raw_profile,
        "audit_log": [audit_entry],
    }

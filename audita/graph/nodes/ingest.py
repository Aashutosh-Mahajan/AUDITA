"""
Ingest node — reads a tabular source file and produces a raw profile dict.

Supports CSV/TSV (auto encoding + delimiter detection) and Excel workbooks
(.xlsx/.xlsm/.xls), optionally targeting a named sheet. Whatever the input
format, the node normalises it to a single CSV on disk so every downstream
node has one dtype-stable contract to read.
"""

import csv
import os
from typing import Any

import chardet
import pandas as pd

from audita.core.audit_log import log_code_action
from audita.core.frame_io import write_frame

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}
CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}


def is_excel_path(file_path: str) -> bool:
    """Return True when the path looks like an Excel workbook."""
    return os.path.splitext(file_path)[1].lower() in EXCEL_EXTENSIONS


def list_sheet_names(file_path: str) -> list[str]:
    """Return the sheet names in an Excel workbook (empty list for CSVs)."""
    if not is_excel_path(file_path):
        return []
    with pd.ExcelFile(file_path) as workbook:
        return list(workbook.sheet_names)


def _detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet; fall back to utf-8."""
    with open(file_path, "rb") as f:
        raw = f.read(10_000)  # sample first 10KB
    result = chardet.detect(raw)
    encoding = result.get("encoding", "utf-8") or "utf-8"
    return encoding


def _detect_delimiter(file_path: str, encoding: str) -> str:
    """Use csv.Sniffer to detect the delimiter; fall back to comma."""
    with open(file_path, encoding=encoding, errors="replace") as f:
        sample = f.read(8_192)
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except csv.Error:
        return ","


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_csv(file_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a delimited text file, detecting encoding and delimiter."""
    try:
        encoding = _detect_encoding(file_path)
        pd.read_csv(file_path, encoding=encoding, nrows=0)  # verify it works
    except (UnicodeDecodeError, LookupError):
        encoding = "utf-8"

    delimiter = _detect_delimiter(file_path, encoding)
    df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)

    return df, {"format": "csv", "encoding": encoding, "delimiter": repr(delimiter)}


def _read_excel(
    file_path: str, sheet_name: str | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one sheet of an Excel workbook.

    Defaults to the first sheet when ``sheet_name`` is not given or does not
    exist in the workbook — an unknown name is a caller mistake, not a reason
    to fail the whole run.
    """
    sheet_names = list_sheet_names(file_path)
    if not sheet_names:
        raise ValueError(f"Excel workbook has no sheets: {file_path}")

    resolved = sheet_name if sheet_name in sheet_names else sheet_names[0]
    df = pd.read_excel(file_path, sheet_name=resolved)

    return df, {
        "format": "excel",
        "sheet": resolved,
        "available_sheets": sheet_names,
        "sheet_fallback": sheet_name is not None and sheet_name != resolved,
    }


def _build_raw_profile(df: pd.DataFrame, file_path: str) -> dict[str, Any]:
    """Build the raw_profile dict from the ingested DataFrame."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "file_size_bytes": os.path.getsize(file_path),
        "columns": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def ingest(state: dict) -> dict:
    """LangGraph node: ingest a CSV or Excel file and produce raw_profile.

    Reads ``state["source_path"]`` (``state["csv_path"]`` is still accepted),
    optionally honouring ``state["sheet_name"]`` for Excel workbooks, and
    returns state updates including the raw profile and an audit log entry.
    """
    source_path: str = state.get("source_path") or state["csv_path"]
    sheet_name: str | None = state.get("sheet_name")

    if is_excel_path(source_path):
        df, read_detail = _read_excel(source_path, sheet_name)
    else:
        df, read_detail = _read_csv(source_path)

    # Normalise to one intermediate file so downstream nodes have a single
    # input contract, written in a format that preserves dtypes.
    raw_csv_path = write_frame(df, prefix="audita_", stem="raw_data")

    raw_profile = _build_raw_profile(df, source_path)

    audit_entry = log_code_action(
        stage="ingest",
        action="loaded_dataset",
        detail={
            "file": os.path.basename(source_path),
            **read_detail,
            "rows": raw_profile["n_rows"],
            "cols": raw_profile["n_cols"],
        },
    )

    return {
        "source_path": source_path,
        "csv_path": raw_csv_path,
        "raw_profile": raw_profile,
        "audit_log": [audit_entry],
    }

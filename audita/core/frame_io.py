"""
Intermediate DataFrame persistence.

Nodes hand DataFrames to each other through the filesystem. CSV cannot carry
dtypes, so a round-trip silently undid work: ``parse_dates`` produced a real
datetime column, ``to_csv`` wrote it back as text, and the next node read it
as ``object`` again — making date-axis line charts impossible and the
before/after profile misleading.

Parquet preserves dtypes exactly, so it is preferred. When no Parquet engine
is installed we fall back to CSV rather than failing the run; the dtype loss
returns, but the pipeline still completes.
"""

import os
import tempfile

import pandas as pd

PARQUET_SUFFIX = ".parquet"


def parquet_available() -> bool:
    """Return True when a Parquet engine is importable."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def write_frame(df: pd.DataFrame, prefix: str, stem: str) -> str:
    """Write ``df`` to a fresh temp directory and return the path.

    Uses Parquet when possible so dtypes survive; falls back to CSV.
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)

    if parquet_available():
        path = os.path.join(temp_dir, stem + PARQUET_SUFFIX)
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):
            # e.g. a column type no engine can serialise — fall through to CSV
            pass

    path = os.path.join(temp_dir, stem + ".csv")
    df.to_csv(path, index=False)
    return path


def read_frame(path: str) -> pd.DataFrame:
    """Read a DataFrame written by :func:`write_frame` (Parquet or CSV)."""
    if path.endswith(PARQUET_SUFFIX):
        return pd.read_parquet(path)
    return pd.read_csv(path)

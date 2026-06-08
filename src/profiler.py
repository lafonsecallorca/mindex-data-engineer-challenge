"""
profiler.py — Reusable DataFrame profiling for data quality assessment.

Produces a structured quality summary for any pandas DataFrame, covering:
- Shape metrics (rows, columns, duplicates)
- Per-column null analysis
- Numeric column statistics (min, max, mean, zeros, negatives)
- Date column detection and statistics (min, max, future dates)

Design decisions:
- Returns plain dicts/lists so output is JSON-serializable without custom encoders.
- Date detection is heuristic: we attempt pd.to_datetime on object/string columns
  and classify a column as date-like if ≥50% of non-null values parse successfully.
- "Today" is injected as a parameter for testability — no hidden dependency on wall clock.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Threshold: fraction of non-null values that must parse as dates
# for us to classify a column as date-like.
_DATE_DETECTION_THRESHOLD = 0.50


def profile(df: pd.DataFrame, name: str, today: datetime | None = None) -> dict[str, Any]:
    """Return a quality summary dict for *df* identified by *name*.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to profile.
    name : str
        A human-readable label (e.g. the source file name).
    today : datetime, optional
        Reference date for "future date" detection.  Defaults to now.

    Returns
    -------
    dict  — JSON-serializable summary.
    """
    if today is None:
        today = datetime.now()

    today_ts = pd.Timestamp(today)

    result: dict[str, Any] = {
        "name": name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = _base_column_stats(series)

        # Numeric analysis
        if pd.api.types.is_numeric_dtype(series):
            col_info.update(_numeric_stats(series))

        # Date analysis — try on object/string columns AND columns already typed as datetime
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            date_stats = _try_date_stats(series, today_ts)
            if date_stats is not None:
                col_info.update(date_stats)
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info.update(_datetime_stats(series, today_ts))

        result["columns"][col] = col_info

    return result


def _base_column_stats(series: pd.Series) -> dict[str, Any]:
    """Null counts and percentages — applies to every column."""
    null_count = int(series.isna().sum())
    total = len(series)
    return {
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_pct": round(null_count / total * 100, 2) if total > 0 else 0.0,
    }


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    """Min, max, mean, zero count, negative count for numeric columns."""
    non_null = series.dropna()
    if non_null.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "zero_count": 0,
            "negative_count": 0,
        }
    return {
        "min": _safe_scalar(non_null.min()),
        "max": _safe_scalar(non_null.max()),
        "mean": round(float(non_null.mean()), 4),
        "zero_count": int((non_null == 0).sum()),
        "negative_count": int((non_null < 0).sum()),
    }


def _try_date_stats(series: pd.Series, today: pd.Timestamp) -> dict[str, Any] | None:
    """Attempt to parse an object column as dates.

    Returns date stats if ≥50 % of non-null values parse, else None.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None

    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    success_rate = parsed.notna().sum() / len(non_null)

    if success_rate < _DATE_DETECTION_THRESHOLD:
        return None

    return _datetime_stats(parsed.dropna(), today, detected=True)


def _datetime_stats(
    series: pd.Series, today: pd.Timestamp, detected: bool = False
) -> dict[str, Any]:
    """Stats for a series already in datetime64 form."""
    non_null = series.dropna()
    stats: dict[str, Any] = {"is_date_column": True}
    if detected:
        stats["date_detected_from_strings"] = True

    if non_null.empty:
        stats.update({"min_date": None, "max_date": None, "future_date_count": 0})
    else:
        stats.update({
            "min_date": str(non_null.min().date()),
            "max_date": str(non_null.max().date()),
            "future_date_count": int((non_null > today).sum()),
        })
    return stats


def _safe_scalar(value: Any) -> int | float | str | None:
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def profile_all(
    file_paths: dict[str, str],
    today: datetime | None = None,
) -> list[dict[str, Any]]:
    """Profile multiple CSV files and return a list of summaries.

    Parameters
    ----------
    file_paths : dict[str, str]
        Mapping of logical name → file path.
    today : datetime, optional
        Reference date passed through to ``profile()``.
    """
    reports = []
    for name, path in file_paths.items():
        df = pd.read_csv(path)
        reports.append(profile(df, name, today=today))
    return reports

"""
cleaner.py — Data cleaning pipeline for raw source files.

Each clean_* function takes a raw DataFrame and returns a cleaned DataFrame.
Cleaning decisions are documented inline and summarized in the README.
"""

from datetime import datetime

import numpy as np
import pandas as pd

_DEFAULT_TODAY = datetime(2026, 6, 2)


def clean_stores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Deduplicate on store_id — keep first occurrence when a store_id
    # appears with variant names (data entry inconsistency).
    df = df.drop_duplicates(subset=["store_id"], keep="first")

    # Pad zip codes to 5 digits — CSV readers strip leading zeros from numeric-looking values.
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    # Impute NULL regions using state-to-region mapping derived from existing data.
    # Static fallback for states where all rows have NULL region.
    _STATE_REGION_FALLBACK = {
        "OR": "West", "WA": "West", "CA": "West",
        "TX": "South", "FL": "South",
        "NY": "Northeast", "PA": "Northeast",
        "MN": "Midwest", "IL": "Midwest",
    }
    state_region = df.dropna(subset=["region"]).drop_duplicates(subset=["state"]).set_index("state")["region"]
    df["region"] = df.apply(
        lambda r: (
            r["region"] if pd.notna(r["region"])
            else state_region.get(r["state"], _STATE_REGION_FALLBACK.get(r["state"], "Unknown"))
        ),
        axis=1,
    )

    # Parse opened_date to proper datetime
    df["opened_date"] = pd.to_datetime(df["opened_date"], format="%Y-%m-%d")

    return df.reset_index(drop=True)


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop exact duplicate rows.
    df = df.drop_duplicates()

    # Products with multiple prices (undocumented price changes).
    # Keep the highest (assumed most recent) price for the dimension table.
    # The fact table will use the unit_price recorded on the transaction itself.
    df = df.sort_values("unit_price", ascending=False).drop_duplicates(
        subset=["product_id"], keep="first"
    )

    # Fill NULL categories with "Unknown".
    df["category"] = df["category"].fillna("Unknown")

    # Flag zero-price products.
    # Keep the record (valid catalog entry) but flag for downstream awareness.
    df["is_zero_price"] = df["unit_price"] == 0.0

    return df.sort_values("product_id").reset_index(drop=True)


def _parse_transaction_dates(series: pd.Series) -> pd.Series:
    """Parse a transaction_date series with mixed formats into datetime.

    Handles three formats present in the raw data:
      - YYYY-MM-DD (ISO, majority of rows)
      - MM/DD/YYYY (US format, 10 rows)
      - DD-MM-YYYY (EU format, 10 rows)

    Strategy: try ISO first (most rows), then US, then EU.
    This ordering resolves ambiguity — a value like "05/06/2026" is treated as
    May 6 (US convention) rather than June 5 (EU), which matches the seed
    generator's behavior.
    """
    result = pd.Series(pd.NaT, index=series.index)
    remaining = series.index.tolist()

    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
    for fmt in formats:
        if not remaining:
            break
        subset = series.loc[remaining]
        parsed = pd.to_datetime(subset, format=fmt, errors="coerce")
        matched = parsed.dropna().index.tolist()
        result.loc[matched] = parsed.loc[matched]
        remaining = [i for i in remaining if i not in matched]

    return result


def clean_transactions(
    df: pd.DataFrame,
    valid_store_ids: set[str],
    valid_product_ids: set[str],
    today: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean transactions and return (clean_df, quarantine_df).

    Quarantined records are excluded from the warehouse but preserved
    for auditability with a reason column.
    """
    today = today or _DEFAULT_TODAY
    df = df.copy()

    # --- Type fixes ---

    # Strip non-numeric characters (e.g. "$") from total_amount, then coerce.
    # Invalid values become NaN and are quarantined rather than crashing the pipeline.
    df["total_amount"] = pd.to_numeric(
        df["total_amount"]
        .astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )

    # Parse mixed-format dates.
    df["transaction_date"] = _parse_transaction_dates(df["transaction_date"])

    # --- Deduplication ---

    # Deduplicate on transaction_id. Keep first occurrence.
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")

    # --- Build quarantine flags (multi-reason: a row can fail multiple checks) ---

    quarantine_reasons: dict[int, list[str]] = {}

    def _flag(mask: pd.Series, reason: str) -> None:
        for idx in df.loc[mask].index:
            quarantine_reasons.setdefault(idx, []).append(reason)

    # Invalid total_amount — could not parse to a number.
    _flag(df["total_amount"].isna(), "invalid_total_amount")

    # Future-dated transactions — can't record sales that haven't occurred.
    _flag(df["transaction_date"] > pd.Timestamp(today), "future_date")

    # Zero-quantity transactions — no economic event occurred.
    _flag(df["quantity"] == 0, "zero_quantity")

    # Orphaned store_ids — no matching row in dim_store.
    _flag(~df["store_id"].isin(valid_store_ids), "orphaned_store_id")

    # Orphaned product_ids — no matching row in dim_product.
    _flag(~df["product_id"].isin(valid_product_ids), "orphaned_product_id")

    # Unparseable dates.
    _flag(df["transaction_date"].isna(), "unparseable_date")

    # --- Split clean vs quarantine ---

    quarantine_idx = list(quarantine_reasons.keys())
    quarantine_df = df.loc[quarantine_idx].copy()
    quarantine_df["quarantine_reason"] = quarantine_df.index.map(
        lambda i: "; ".join(quarantine_reasons[i])
    )

    clean_df = df.drop(index=quarantine_idx).copy()

    # --- Enrichment flags on clean data ---

    # Flag price mismatches (silent discounts) — total_amount != qty * unit_price.
    expected = (clean_df["quantity"] * clean_df["unit_price"]).round(2)
    clean_df["has_price_mismatch"] = ~np.isclose(
        clean_df["total_amount"], expected, atol=0.01
    )

    # Flag return transactions (negative quantity).
    clean_df["is_return"] = clean_df["quantity"] < 0

    # Flag guest transactions (NULL customer_id).
    clean_df["is_guest"] = clean_df["customer_id"].isna()

    return clean_df.reset_index(drop=True), quarantine_df.reset_index(drop=True)

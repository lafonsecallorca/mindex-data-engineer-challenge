"""
pipeline.py — Entry point for the data engineering pipeline.

Runs all phases in order:
  1. Profile raw data  → output/profiling_report.json
  2. Clean data        → validated DataFrames
  3. Load warehouse    → output/warehouse.db
  4. Run analytics     → output/analytics.json

Usage:
    python src/pipeline.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when run from the project root.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from profiler import profile_all
from cleaner import clean_stores, clean_products, clean_transactions
from loader import load_warehouse
from analytics import run_all_analytics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference date aligned with the seed_data.py generator.
TODAY = datetime(2026, 6, 2)


def run_profiling() -> None:
    """Phase 1: Profile all raw source files."""

    source_files = {
        "stores": str(DATA_DIR / "stores.csv"),
        "products": str(DATA_DIR / "products.csv"),
        "transactions": str(DATA_DIR / "transactions.csv"),
    }

    reports = profile_all(source_files, today=TODAY)

    output_path = OUTPUT_DIR / "profiling_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=str)

    print(f"[profiling] Report written to {output_path}")
    for r in reports:
        print(f"  {r['name']}: {r['row_count']} rows, {r['column_count']} cols, "
              f"{r['duplicate_row_count']} duplicate rows")


def run_cleaning() -> tuple:
    """Phase 2: Clean all raw source files."""

    stores_raw = pd.read_csv(DATA_DIR / "stores.csv")
    products_raw = pd.read_csv(DATA_DIR / "products.csv")
    transactions_raw = pd.read_csv(DATA_DIR / "transactions.csv")

    stores = clean_stores(stores_raw)
    products = clean_products(products_raw)

    valid_store_ids = set(stores["store_id"])
    valid_product_ids = set(products["product_id"])

    transactions, quarantine = clean_transactions(
        transactions_raw, valid_store_ids, valid_product_ids, today=TODAY
    )

    # Persist quarantine for auditability.
    quarantine_path = OUTPUT_DIR / "quarantine.csv"
    quarantine.to_csv(quarantine_path, index=False)

    print(f"[cleaning] stores:       {len(stores_raw)} → {len(stores)} rows")
    print(f"[cleaning] products:     {len(products_raw)} → {len(products)} rows")
    print(f"[cleaning] transactions: {len(transactions_raw)} → {len(transactions)} rows "
          f"({len(quarantine)} quarantined)")
    print(f"[cleaning] Quarantine written to {quarantine_path}")
    print(f"  Quarantine breakdown (per rule):")
    if not quarantine.empty:
        all_reasons = quarantine["quarantine_reason"].str.split("; ").explode()
        for reason, count in all_reasons.value_counts().items():
            print(f"    {reason}: {count}")
    print(f"  Flags on clean transactions:")
    print(f"    returns:          {transactions['is_return'].sum()}")
    print(f"    price mismatches: {transactions['has_price_mismatch'].sum()}")
    print(f"    guest (no cust):  {transactions['is_guest'].sum()}")

    return stores, products, transactions, quarantine


def run_loading(stores, products, transactions) -> None:
    """Phase 3: Build star schema in SQLite."""

    db_path = OUTPUT_DIR / "warehouse.db"
    counts = load_warehouse(db_path, stores, products, transactions)

    print(f"[loading] Database written to {db_path}")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")


def main() -> None:
    print("=" * 60)
    print("Phase 1: Data Profiling")
    print("=" * 60)
    run_profiling()

    print()
    print("=" * 60)
    print("Phase 2: Data Cleaning")
    print("=" * 60)
    stores, products, transactions, quarantine = run_cleaning()

    print()
    print("=" * 60)
    print("Phase 3: Data Loading")
    print("=" * 60)
    run_loading(stores, products, transactions)

    print()
    print("=" * 60)
    print("Phase 4: Analytics")
    print("=" * 60)
    run_analytics()


def run_analytics() -> None:
    """Phase 4: Run analytical queries and save results."""

    db_path = OUTPUT_DIR / "warehouse.db"
    results = run_all_analytics(db_path)

    output_path = OUTPUT_DIR / "analytics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[analytics] Results written to {output_path}")
    for key, rows in results.items():
        print(f"  {key}: {len(rows)} rows")


if __name__ == "__main__":
    main()

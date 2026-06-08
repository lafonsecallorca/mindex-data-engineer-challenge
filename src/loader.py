"""
loader.py — Builds a star schema in SQLite from cleaned DataFrames.

Tables:
  - dim_date:    calendar attributes for every date in the transaction window
  - dim_store:   one row per store location
  - dim_product: one row per product
  - fact_sales:  one row per transaction, foreign keys to all dimensions
"""

import sqlite3
from pathlib import Path

import pandas as pd


def build_dim_date(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    """Generate a date dimension covering the full transaction window."""
    dates = pd.date_range(start=min_date.normalize(), end=max_date.normalize(), freq="D")
    df = pd.DataFrame({"full_date": dates})

    df["date_key"] = df["full_date"].dt.strftime("%Y%m%d").astype(int)
    df["year"] = df["full_date"].dt.year
    df["quarter"] = df["full_date"].dt.quarter
    df["month"] = df["full_date"].dt.month
    df["month_name"] = df["full_date"].dt.month_name()
    df["day"] = df["full_date"].dt.day
    df["day_of_week"] = df["full_date"].dt.day_name()
    df["is_weekend"] = df["full_date"].dt.dayofweek >= 5

    df["full_date"] = df["full_date"].dt.strftime("%Y-%m-%d")

    return df


def build_dim_store(stores: pd.DataFrame) -> pd.DataFrame:
    """Shape cleaned stores into the store dimension."""
    df = stores[["store_id", "store_name", "city", "state", "zip_code", "region", "opened_date"]].copy()
    df["opened_date"] = df["opened_date"].dt.strftime("%Y-%m-%d")
    return df


def build_dim_product(products: pd.DataFrame) -> pd.DataFrame:
    """Shape cleaned products into the product dimension."""
    return products[["product_id", "product_name", "category", "unit_price", "supplier_id", "is_zero_price"]].copy()


def build_fact_sales(transactions: pd.DataFrame) -> pd.DataFrame:
    """Shape cleaned transactions into the fact table with dimension keys."""
    df = transactions.copy()

    df["date_key"] = pd.to_datetime(df["transaction_date"]).dt.strftime("%Y%m%d").astype(int)

    return df[[
        "transaction_id",
        "date_key",
        "store_id",
        "product_id",
        "customer_id",
        "quantity",
        "unit_price",
        "total_amount",
        "is_return",
        "is_guest",
        "has_price_mismatch",
    ]]


_SCHEMA_DDL = """
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_store;
DROP TABLE IF EXISTS dim_product;

CREATE TABLE dim_date (
    date_key      INTEGER PRIMARY KEY,
    full_date     TEXT NOT NULL,
    year          INTEGER NOT NULL,
    quarter       INTEGER NOT NULL,
    month         INTEGER NOT NULL,
    month_name    TEXT NOT NULL,
    day           INTEGER NOT NULL,
    day_of_week   TEXT NOT NULL,
    is_weekend    INTEGER NOT NULL
);

CREATE TABLE dim_store (
    store_id      TEXT PRIMARY KEY,
    store_name    TEXT NOT NULL,
    city          TEXT NOT NULL,
    state         TEXT NOT NULL,
    zip_code      TEXT NOT NULL,
    region        TEXT NOT NULL,
    opened_date   TEXT NOT NULL
);

CREATE TABLE dim_product (
    product_id    TEXT PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    unit_price    REAL NOT NULL,
    supplier_id   TEXT NOT NULL,
    is_zero_price INTEGER NOT NULL
);

CREATE TABLE fact_sales (
    transaction_id     TEXT PRIMARY KEY,
    date_key           INTEGER NOT NULL REFERENCES dim_date(date_key),
    store_id           TEXT NOT NULL REFERENCES dim_store(store_id),
    product_id         TEXT NOT NULL REFERENCES dim_product(product_id),
    customer_id        TEXT,
    quantity           INTEGER NOT NULL,
    unit_price         REAL NOT NULL,
    total_amount       REAL NOT NULL,
    is_return          INTEGER NOT NULL,
    is_guest           INTEGER NOT NULL,
    has_price_mismatch INTEGER NOT NULL
);
"""


def load_warehouse(
    db_path: str | Path,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame,
) -> dict[str, int]:
    """Build and populate the star schema in SQLite.

    Idempotent — drops and recreates all tables on every run.
    """
    dim_date = build_dim_date(
        transactions["transaction_date"].min(),
        transactions["transaction_date"].max(),
    )
    dim_store = build_dim_store(stores)
    dim_product = build_dim_product(products)
    fact_sales = build_fact_sales(transactions)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA_DDL)
        conn.execute("PRAGMA foreign_keys = ON;")

        dim_date.to_sql("dim_date", conn, if_exists="append", index=False)
        dim_store.to_sql("dim_store", conn, if_exists="append", index=False)
        dim_product.to_sql("dim_product", conn, if_exists="append", index=False)
        fact_sales.to_sql("fact_sales", conn, if_exists="append", index=False)

        conn.commit()

        # Verify row counts
        _TABLES = ["dim_date", "dim_store", "dim_product", "fact_sales"]
        counts = {}
        for table in _TABLES:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()

    return counts

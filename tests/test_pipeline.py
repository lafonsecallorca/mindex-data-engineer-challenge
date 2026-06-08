"""
tests for the data pipeline — profiling, cleaning, and analytics.

Fixtures use small, controlled DataFrames with known values so assertions
are deterministic and independent of the actual source data.
"""

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from profiler import profile
from cleaner import clean_stores, clean_transactions, _parse_transaction_dates
from analytics import (
    top_stores_by_net_revenue,
    avg_transaction_value_by_region,
    return_rate_by_store,
    mom_revenue_change_by_category,
    top_customers_by_spend,
)


# ── Profiler Tests ───────────────────────────────────────────────────────────


class TestProfiler:
    def test_basic_shape_and_nulls(self):
        """Profiler reports correct row/column counts and null statistics."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", None, "c"],
            "value": [10.0, 20.0, 30.0],
        })
        result = profile(df, "test_basic")

        assert result["name"] == "test_basic"
        assert result["row_count"] == 3
        assert result["column_count"] == 3
        assert result["duplicate_row_count"] == 0

        assert result["columns"]["name"]["null_count"] == 1
        assert result["columns"]["name"]["null_pct"] == pytest.approx(33.33, abs=0.01)
        assert result["columns"]["id"]["null_count"] == 0

    def test_numeric_stats(self):
        """Profiler computes correct min, max, mean, zero/negative counts."""
        df = pd.DataFrame({"val": [-5, 0, 0, 10, 20]})
        result = profile(df, "test_numeric")
        col = result["columns"]["val"]

        assert col["min"] == -5
        assert col["max"] == 20
        assert col["mean"] == pytest.approx(5.0)
        assert col["zero_count"] == 2
        assert col["negative_count"] == 1

    def test_empty_dataframe(self):
        """Profiler handles an empty DataFrame without errors."""
        df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="str")})
        result = profile(df, "empty")

        assert result["row_count"] == 0
        assert result["column_count"] == 2
        assert result["duplicate_row_count"] == 0
        assert result["columns"]["a"]["null_count"] == 0

    def test_all_null_column(self):
        """Profiler correctly reports a column that is entirely null."""
        df = pd.DataFrame({"x": [None, None, None]})
        result = profile(df, "all_null")

        assert result["columns"]["x"]["null_count"] == 3
        assert result["columns"]["x"]["null_pct"] == 100.0

    def test_date_detection(self):
        """Profiler detects date-like string columns and counts future dates."""
        today = datetime(2026, 6, 2)
        df = pd.DataFrame({
            "dt": ["2026-05-01", "2026-06-01", "2026-07-15", "2026-08-01"],
        })
        result = profile(df, "dates", today=today)
        col = result["columns"]["dt"]

        assert col["is_date_column"] is True
        assert col["min_date"] == "2026-05-01"
        assert col["max_date"] == "2026-08-01"
        assert col["future_date_count"] == 2  # July 15 and Aug 1 are after June 2

    def test_duplicate_detection(self):
        """Profiler counts duplicate rows correctly."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = profile(df, "dupes")

        assert result["duplicate_row_count"] == 1  # second (1, "x") is the dupe


# ── Cleaner Tests ────────────────────────────────────────────────────────────


class TestCleaner:
    def test_zip_code_padding(self):
        """Zip codes with stripped leading zeros are padded to 5 digits."""
        df = pd.DataFrame({
            "store_id": ["S001", "S002"],
            "store_name": ["Store A", "Store B"],
            "city": ["City", "Town"],
            "state": ["NY", "NY"],
            "zip_code": [938, 14623],
            "region": ["Northeast", "Northeast"],
            "opened_date": ["2020-01-01", "2020-01-01"],
        })
        result = clean_stores(df)

        assert result.loc[result["store_id"] == "S001", "zip_code"].iloc[0] == "00938"
        assert result.loc[result["store_id"] == "S002", "zip_code"].iloc[0] == "14623"

    def test_store_deduplication(self):
        """Near-duplicate stores (same store_id) are deduplicated, keeping first."""
        df = pd.DataFrame({
            "store_id": ["S007", "S007"],
            "store_name": ["Downtown Rochester", "Rochester Downtown"],
            "city": ["Rochester", "Rochester"],
            "state": ["NY", "NY"],
            "zip_code": ["14604", "14604"],
            "region": ["Northeast", "Northeast"],
            "opened_date": ["2006-01-22", "2006-01-22"],
        })
        result = clean_stores(df)

        assert len(result) == 1
        assert result.iloc[0]["store_name"] == "Downtown Rochester"

    def test_date_parsing_mixed_formats(self):
        """Mixed date formats (ISO, US, EU) are parsed correctly."""
        series = pd.Series([
            "2026-05-15",    # ISO
            "05/20/2026",    # US (MM/DD/YYYY)
            "25-04-2026",    # EU (DD-MM-YYYY)
        ])
        result = _parse_transaction_dates(series)

        assert result.iloc[0] == pd.Timestamp("2026-05-15")
        assert result.iloc[1] == pd.Timestamp("2026-05-20")
        assert result.iloc[2] == pd.Timestamp("2026-04-25")

    def test_dollar_sign_stripping(self):
        """Total amounts with '$' prefix are correctly parsed to float."""
        df = pd.DataFrame({
            "transaction_id": ["T1", "T2"],
            "transaction_date": ["2026-05-01", "2026-05-01"],
            "store_id": ["S001", "S001"],
            "product_id": ["P001", "P001"],
            "customer_id": ["C001", "C001"],
            "quantity": [1, 2],
            "unit_price": [10.0, 20.0],
            "total_amount": ["$10.00", "40.00"],
        })
        clean, _ = clean_transactions(df, {"S001"}, {"P001"})

        assert clean["total_amount"].dtype == np.float64
        assert clean.iloc[0]["total_amount"] == 10.00
        assert clean.iloc[1]["total_amount"] == 40.00

    def test_quarantine_orphaned_store(self):
        """Transactions with store_ids not in the dimension are quarantined."""
        df = pd.DataFrame({
            "transaction_id": ["T1", "T2"],
            "transaction_date": ["2026-05-01", "2026-05-01"],
            "store_id": ["S001", "S999"],
            "product_id": ["P001", "P001"],
            "customer_id": ["C001", "C001"],
            "quantity": [1, 1],
            "unit_price": [10.0, 10.0],
            "total_amount": [10.0, 10.0],
        })
        clean, quarantine = clean_transactions(df, {"S001"}, {"P001"})

        assert len(clean) == 1
        assert len(quarantine) == 1
        assert quarantine.iloc[0]["quarantine_reason"] == "orphaned_store_id"

    def test_quarantine_future_date(self):
        """Transactions with future dates are quarantined."""
        df = pd.DataFrame({
            "transaction_id": ["T1", "T2"],
            "transaction_date": ["2026-05-01", "2026-12-25"],
            "store_id": ["S001", "S001"],
            "product_id": ["P001", "P001"],
            "customer_id": ["C001", "C001"],
            "quantity": [1, 1],
            "unit_price": [10.0, 10.0],
            "total_amount": [10.0, 10.0],
        })
        clean, quarantine = clean_transactions(df, {"S001"}, {"P001"})

        assert len(clean) == 1
        assert "future_date" in quarantine.iloc[0]["quarantine_reason"]

    def test_quarantine_invalid_amount(self):
        """Transactions with unparseable total_amount are quarantined, not crashed."""
        df = pd.DataFrame({
            "transaction_id": ["T1", "T2", "T3"],
            "transaction_date": ["2026-05-01", "2026-05-01", "2026-05-01"],
            "store_id": ["S001", "S001", "S001"],
            "product_id": ["P001", "P001", "P001"],
            "customer_id": ["C001", "C001", "C001"],
            "quantity": [1, 1, 1],
            "unit_price": [10.0, 10.0, 10.0],
            "total_amount": ["10.00", "N/A", ""],
        })
        clean, quarantine = clean_transactions(df, {"S001"}, {"P001"})

        assert len(clean) == 1
        assert len(quarantine) == 2
        for _, row in quarantine.iterrows():
            assert "invalid_total_amount" in row["quarantine_reason"]

    def test_multi_reason_quarantine(self):
        """A transaction violating multiple rules gets all reasons recorded."""
        df = pd.DataFrame({
            "transaction_id": ["T1"],
            "transaction_date": ["2026-12-25"],  # future
            "store_id": ["S999"],                # orphaned
            "product_id": ["P001"],
            "customer_id": ["C001"],
            "quantity": [0],                     # zero qty
            "unit_price": [10.0],
            "total_amount": [0],
        })
        _, quarantine = clean_transactions(df, {"S001"}, {"P001"})

        assert len(quarantine) == 1
        reasons = quarantine.iloc[0]["quarantine_reason"]
        assert "future_date" in reasons
        assert "zero_quantity" in reasons
        assert "orphaned_store_id" in reasons


# ── Analytics Tests ──────────────────────────────────────────────────────────


class TestAnalytics:
    @pytest.fixture
    def warehouse_conn(self):
        """Create an in-memory SQLite warehouse with a small controlled fixture."""
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE dim_date (
                date_key INTEGER PRIMARY KEY, full_date TEXT, year INTEGER,
                quarter INTEGER, month INTEGER, month_name TEXT,
                day INTEGER, day_of_week TEXT, is_weekend INTEGER
            );
            CREATE TABLE dim_store (
                store_id TEXT PRIMARY KEY, store_name TEXT, city TEXT,
                state TEXT, zip_code TEXT, region TEXT, opened_date TEXT
            );
            CREATE TABLE dim_product (
                product_id TEXT PRIMARY KEY, product_name TEXT, category TEXT,
                unit_price REAL, supplier_id TEXT, is_zero_price INTEGER
            );
            CREATE TABLE fact_sales (
                transaction_id TEXT PRIMARY KEY, date_key INTEGER,
                store_id TEXT, product_id TEXT, customer_id TEXT,
                quantity INTEGER, unit_price REAL, total_amount REAL,
                is_return INTEGER, is_guest INTEGER, has_price_mismatch INTEGER
            );

            INSERT INTO dim_date VALUES
                (20260425, '2026-04-25', 2026, 2, 4, 'April', 25, 'Saturday', 1),
                (20260525, '2026-05-25', 2026, 2, 5, 'May', 25, 'Monday', 0),
                (20260526, '2026-05-26', 2026, 2, 5, 'May', 26, 'Tuesday', 0),
                (20260601, '2026-06-01', 2026, 2, 6, 'June', 1, 'Monday', 0);

            INSERT INTO dim_store VALUES
                ('S001', 'Store Alpha', 'Victor', 'NY', '14564', 'Northeast', '2020-01-01'),
                ('S002', 'Store Beta', 'Austin', 'TX', '78748', 'South', '2020-01-01');

            INSERT INTO dim_product VALUES
                ('P001', 'Widget', 'Electronics', 50.00, 'SUP001', 0),
                ('P002', 'Gadget', 'Apparel', 30.00, 'SUP002', 0);

            -- S001: 2 sales ($100 + $150) + 1 return (-$50) = $200 net
            -- S002: 1 sale ($200) = $200 net
            -- Guest transaction T5 (is_guest=1) for filtering tests
            -- C004 is a heavy-returner with net negative spend
            INSERT INTO fact_sales VALUES
                ('T1', 20260525, 'S001', 'P001', 'C001', 2, 50.0, 100.0, 0, 0, 0),
                ('T2', 20260526, 'S001', 'P001', 'C002', 3, 50.0, 150.0, 0, 0, 0),
                ('T3', 20260526, 'S001', 'P001', 'C001', -1, 50.0, -50.0, 1, 0, 0),
                ('T4', 20260601, 'S002', 'P001', 'C003', 4, 50.0, 200.0, 0, 0, 0),
                ('T5', 20260525, 'S002', 'P002', NULL, 1, 30.0, 30.0, 0, 1, 0),
                ('T6', 20260425, 'S001', 'P002', 'C004', 1, 30.0, 30.0, 0, 0, 0),
                ('T7', 20260525, 'S001', 'P002', 'C004', -2, 30.0, -60.0, 1, 0, 0);
        """)
        yield conn
        conn.close()

    def test_top_stores_net_revenue(self, warehouse_conn):
        """Net revenue correctly includes returns as negative amounts."""
        results = top_stores_by_net_revenue(warehouse_conn)

        revenues = {r["store_id"]: r["net_revenue"] for r in results}
        # 30-day window: 2026-05-03 to 2026-06-01
        # S001 in window: T1=100, T2=150, T3=-50, T7=-60 = 140 (T6 is Apr 25, outside)
        # S002 in window: T4=200, T5=30 = 230
        assert revenues["S002"] == 230.0
        assert revenues["S001"] == 140.0

    def test_avg_transaction_value_excludes_returns(self, warehouse_conn):
        """Average transaction value excludes return transactions."""
        results = avg_transaction_value_by_region(warehouse_conn)

        region_avg = {r["region"]: r["avg_transaction_value"] for r in results}
        # Northeast non-returns: T1=100, T2=150, T6=30 → avg = 93.33
        # South non-returns: T4=200, T5=30 → avg = 115.0
        assert region_avg["Northeast"] == pytest.approx(93.33, abs=0.01)
        assert region_avg["South"] == 115.0

    def test_return_rate_by_store(self, warehouse_conn):
        """Return rate is correctly calculated and high-return flag works."""
        results = return_rate_by_store(warehouse_conn)

        rates = {r["store_id"]: r for r in results}
        # S001: 5 total txns (T1,T2,T3,T6,T7), 2 returns (T3,T7) = 40%
        # S002: 2 total txns (T4,T5), 0 returns = 0%
        assert rates["S001"]["return_rate_pct"] == 40.0
        assert rates["S001"]["high_return_flag"] == 1
        assert rates["S002"]["return_rate_pct"] == 0.0
        assert rates["S002"]["high_return_flag"] == 0

    def test_mom_revenue_change(self, warehouse_conn):
        """Month-over-month change uses LAG correctly across months."""
        results = mom_revenue_change_by_category(warehouse_conn)

        # Filter to Electronics: Apr=30 (T6 is Apparel, not Electronics)
        electronics = [r for r in results if r["category"] == "Electronics"]
        elec_by_month = {r["month"]: r for r in electronics}

        # Electronics: April has no sales, May=200 (T1+T2-T3+... only P001 txns)
        # May electronics: T1=100, T2=150, T3=-50 = 200
        assert elec_by_month[5]["revenue"] == 200.0
        # June electronics: T4=200
        assert elec_by_month[6]["revenue"] == 200.0
        assert elec_by_month[6]["pct_change"] == 0.0  # 200→200 = 0% change

    def test_top_customers_excludes_guests_and_negative_spenders(self, warehouse_conn):
        """Top customers excludes guests and customers with net negative spend."""
        results = top_customers_by_spend(warehouse_conn)

        customer_ids = [r["customer_id"] for r in results]
        # Guest (NULL customer_id from T5) should be excluded
        assert None not in customer_ids
        # C004 has net spend of 30 - 60 = -30, should be excluded by HAVING > 0
        assert "C004" not in customer_ids
        # C003 should be top spender (200)
        assert results[0]["customer_id"] == "C003"
        assert results[0]["lifetime_spend"] == 200.0

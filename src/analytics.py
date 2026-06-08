"""
analytics.py — Business analytics queries against the star schema.

All queries use SQL via sqlite3. SQL is the natural choice for a star schema:
- The warehouse is already modeled for dimensional queries
- SQL is declarative, auditable, and understood by analysts and engineers alike
- Avoids pulling large result sets into Python memory unnecessarily

Each function runs a single query and returns a structured result.
"""

import sqlite3
from pathlib import Path
from typing import Any


def _query(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Execute a query and return results as a list of dicts."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def top_stores_by_net_revenue(conn: sqlite3.Connection) -> list[dict]:
    """Q1: Top 5 stores by net revenue in the most recent 30-day window.

    Net revenue = SUM(total_amount) including returns (negative amounts).
    The 30-day window is relative to the max date in the data, not today,
    so results are stable regardless of when the pipeline runs.
    """
    return _query(conn, """
        WITH date_bounds AS (
            SELECT MAX(full_date) AS max_date,
                   DATE(MAX(full_date), '-29 days') AS window_start
            FROM dim_date d
            INNER JOIN fact_sales f ON d.date_key = f.date_key
        )
        SELECT s.store_id,
               s.store_name,
               s.region,
               ROUND(SUM(f.total_amount), 2) AS net_revenue,
               COUNT(*) AS transaction_count
        FROM fact_sales f
        INNER JOIN dim_store s ON f.store_id = s.store_id
        INNER JOIN dim_date d ON f.date_key = d.date_key
        CROSS JOIN date_bounds b
        WHERE d.full_date BETWEEN b.window_start AND b.max_date
        GROUP BY s.store_id, s.store_name, s.region
        ORDER BY net_revenue DESC
        LIMIT 5
    """)


def mom_revenue_change_by_category(conn: sqlite3.Connection) -> list[dict]:
    """Q2: Month-over-month revenue change (%) by product category.

    Uses LAG to compute change from previous month. First month shows NULL
    for pct_change since there's no prior period.
    """
    return _query(conn, """
        WITH monthly AS (
            SELECT p.category,
                   d.year,
                   d.month,
                   d.month_name,
                   ROUND(SUM(f.total_amount), 2) AS revenue
            FROM fact_sales f
            INNER JOIN dim_product p ON f.product_id = p.product_id
            INNER JOIN dim_date d ON f.date_key = d.date_key
            GROUP BY p.category, d.year, d.month, d.month_name
        ),
        -- Build a spine of all category-month combinations so missing months
        -- appear as explicit NULLs rather than being silently skipped by LAG.
        months AS (
            SELECT DISTINCT year, month, month_name FROM dim_date
        ),
        categories AS (
            SELECT DISTINCT category FROM dim_product
        ),
        spine AS (
            SELECT c.category, m.year, m.month, m.month_name
            FROM categories c CROSS JOIN months m
        ),
        filled AS (
            SELECT s.category, s.year, s.month, s.month_name,
                   monthly.revenue
            FROM spine s
            LEFT JOIN monthly
                ON s.category = monthly.category
               AND s.year = monthly.year
               AND s.month = monthly.month
        ),
        with_lag AS (
            SELECT category,
                   year,
                   month,
                   month_name,
                   revenue,
                   LAG(revenue) OVER (
                       PARTITION BY category ORDER BY year, month
                   ) AS prev_revenue
            FROM filled
        )
        SELECT category,
               year,
               month,
               month_name,
               revenue,
               prev_revenue,
               CASE
                   WHEN prev_revenue IS NOT NULL AND prev_revenue != 0 AND revenue IS NOT NULL
                   THEN ROUND((revenue - prev_revenue) * 100.0 / prev_revenue, 2)
                   ELSE NULL
               END AS pct_change
        FROM with_lag
        ORDER BY category, year, month
    """)


def return_rate_by_store(conn: sqlite3.Connection) -> list[dict]:
    """Q3: Return rate by store (return txns / total txns).

    Flags stores where return rate exceeds 10%.
    """
    return _query(conn, """
        SELECT s.store_id,
               s.store_name,
               COUNT(*) AS total_transactions,
               SUM(CASE WHEN f.is_return = 1 THEN 1 ELSE 0 END) AS return_transactions,
               ROUND(
                   SUM(CASE WHEN f.is_return = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
                   2
               ) AS return_rate_pct,
               CASE
                   WHEN SUM(CASE WHEN f.is_return = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) > 10.0
                   THEN 1 ELSE 0
               END AS high_return_flag
        FROM fact_sales f
        INNER JOIN dim_store s ON f.store_id = s.store_id
        GROUP BY s.store_id, s.store_name
        ORDER BY return_rate_pct DESC
    """)


def avg_transaction_value_by_region(conn: sqlite3.Connection) -> list[dict]:
    """Q4: Average transaction value by region, excluding returns."""
    return _query(conn, """
        SELECT s.region,
               COUNT(*) AS transaction_count,
               ROUND(AVG(f.total_amount), 2) AS avg_transaction_value,
               ROUND(SUM(f.total_amount), 2) AS total_revenue
        FROM fact_sales f
        INNER JOIN dim_store s ON f.store_id = s.store_id
        WHERE f.is_return = 0
        GROUP BY s.region
        ORDER BY avg_transaction_value DESC
    """)


def top_customers_by_spend(conn: sqlite3.Connection) -> list[dict]:
    """Q5: Top 10 customers by lifetime spend.

    Excludes guest/anonymous transactions (NULL customer_id).
    Includes transaction count and average order value.
    """
    return _query(conn, """
        SELECT f.customer_id,
               ROUND(SUM(f.total_amount), 2) AS lifetime_spend,
               COUNT(*) AS transaction_count,
               ROUND(AVG(f.total_amount), 2) AS avg_order_value
        FROM fact_sales f
        WHERE f.is_guest = 0
        GROUP BY f.customer_id
        HAVING SUM(f.total_amount) > 0
        ORDER BY lifetime_spend DESC
        LIMIT 10
    """)


def run_all_analytics(db_path: str | Path) -> dict[str, Any]:
    """Run all analytics queries and return results as a dict."""
    conn = sqlite3.connect(str(db_path))
    try:
        results = {
            "q1_top_stores_by_net_revenue": top_stores_by_net_revenue(conn),
            "q2_mom_revenue_change_by_category": mom_revenue_change_by_category(conn),
            "q3_return_rate_by_store": return_rate_by_store(conn),
            "q4_avg_transaction_value_by_region": avg_transaction_value_by_region(conn),
            "q5_top_customers_by_spend": top_customers_by_spend(conn),
        }
    finally:
        conn.close()
    return results

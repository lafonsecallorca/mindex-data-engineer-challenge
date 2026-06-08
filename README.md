# Mindex Data Engineer Code Challenge

## Setup & Run

```bash
pip install -r requirements.txt
python src/pipeline.py
pytest tests/ -v
```

The pipeline produces four artifacts in `output/`:
- `profiling_report.json` — data quality summary for all source files
- `quarantine.csv` — records excluded from the warehouse with reasons
- `warehouse.db` — SQLite star schema
- `analytics.json` — answers to the five business questions

---

## Architecture

```
data/raw/*.csv
     │
     ▼
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│ profiler  │────>│  cleaner  │────>│  loader   │────>│ analytics │
│           │     │           │     │ (SQLite)  │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
     │                 │                  │                  │
     ▼                 ▼                  ▼                  ▼
profiling_       clean DFs +         warehouse.db       analytics.json
report.json      quarantine.csv
```

Each module has a single responsibility:

| Module | Role |
|---|---|
| `profiler.py` | Generic DataFrame quality profiling — shape, nulls, numeric stats, date detection |
| `cleaner.py` | Rule-based cleaning, type coercion, deduplication, quarantine |
| `loader.py` | Star schema DDL and population in SQLite |
| `analytics.py` | SQL queries against the warehouse |
| `pipeline.py` | Orchestrator — runs all phases in sequence |

---

## Data Quality Findings & Decisions

### stores.csv

| Issue | File | Count | Decision | Rationale |
|---|---|---|---|---|
| Near-duplicate store_id (S007 appears with two different names) | stores.csv | 1 | Deduplicated on store_id, kept first | Same ID, city, state, zip, opened_date — only the name differs. First occurrence is authoritative. |
| Malformed zip code (4 digits, leading zero stripped) | stores.csv | 1 | Padded to 5 digits with `str.zfill(5)` | CSV readers interpret numeric-looking zips as integers, dropping leading zeros. Standard US zips are 5 digits. |
| NULL region (S013, S014 — Portland, OR) | stores.csv | 2 | Imputed using state-to-region mapping derived from existing data, with static fallback for unmapped states | All other stores in the same state had a known region. For OR (no other representation), used Census Bureau regional classification ("West"). |

### products.csv

| Issue | File | Count | Decision | Rationale |
|---|---|---|---|---|
| Exact duplicate row (P012) | products.csv | 1 | Dropped | Identical in every column — artifact of a bad data extract. |
| Multiple prices for same product (P005) | products.csv | 1 | Kept highest price, assumed most recent | Dimension table needs one row per product. Fact table preserves the `unit_price` from each transaction, so historical pricing is not lost. |
| NULL category | products.csv | 5 | Filled with "Unknown" | Products are valid catalog entries. "Unknown" is queryable and doesn't break GROUP BY analytics. |
| Zero unit_price (P027) | products.csv | 1 | Kept with `is_zero_price` flag | May be a promotional item or data error. Flag allows downstream filtering without data loss. |

### transactions.csv

| Issue | File | Count | Decision | Rationale |
|---|---|---|---|---|
| Mixed date formats (ISO, US MM/DD/YYYY, EU DD-MM-YYYY) | transactions.csv | 20 | Parsed with ordered format matching: ISO → US → EU | Most rows are ISO. US format tried before EU to resolve ambiguity (e.g., `05/06/2026` → May 6, not June 5). |
| String-formatted amounts (`$X.XX`) | transactions.csv | 25 | Stripped non-numeric characters via regex, coerced with `pd.to_numeric(errors='coerce')` | Currency symbols are display formatting, not data. Values that still can't parse are quarantined rather than crashing the pipeline. |
| Price mismatch / silent discount (total ≠ qty × price) | transactions.csv | 20 | Kept `total_amount` as-is, added `has_price_mismatch` flag | `total_amount` represents what was actually charged. Overwriting it with a recalculation would be a data integrity violation. |
| Orphaned store_id (S016–S019) | transactions.csv | 5 | Quarantined | No matching dimension record — can't attribute to a store. Breaks referential integrity. |
| Orphaned product_id (P031, P032) | transactions.csv | 3 | Quarantined | No matching dimension record. |
| NULL customer_id (guest transactions) | transactions.csv | 40 | Kept with `is_guest` flag | Valid sales events. Guest checkouts are a normal retail pattern. |
| Zero-quantity rows | transactions.csv | 5 | Quarantined | No economic event occurred (quantity = 0, amount = 0). |
| Future-dated transactions | transactions.csv | 3 | Quarantined | Dates after the data extraction date. Can't record sales that haven't happened. |
| Exact duplicate rows (same transaction_id) | transactions.csv | 15 | Deduplicated on transaction_id, kept first | Same transaction recorded twice — extract artifact. |
| Return transactions (negative qty/amount) | transactions.csv | 30 | Kept with `is_return` flag | Valid business events. Returns reduce net revenue and are needed for return rate analytics. |

**Quarantine approach:** Records that violate data integrity constraints (orphaned FKs, impossible dates, zero-quantity, unparseable amounts) are split into a quarantine DataFrame and persisted to `output/quarantine.csv`. Each row carries a semicolon-delimited `quarantine_reason` column that captures *all* violations (a single row can fail multiple checks). This preserves records for audit and reprocessing without polluting the warehouse.

---

## Schema Design

### dim_date

| Column | Type | Description |
|---|---|---|
| date_key | INTEGER PK | YYYYMMDD integer — sortable, human-readable |
| full_date | TEXT | ISO date string |
| year, quarter, month, day | INTEGER | Calendar components |
| month_name, day_of_week | TEXT | Human-readable labels |
| is_weekend | INTEGER | 1 if Saturday/Sunday |

Generated for every date between the earliest and latest transaction in the clean data. No hardcoded date bounds.

### dim_store

| Column | Type | Description |
|---|---|---|
| store_id | TEXT PK | Natural key from source |
| store_name | TEXT | Store name (deduplicated) |
| city, state, zip_code | TEXT | Location |
| region | TEXT | Business region (imputed where missing) |
| opened_date | TEXT | ISO date |

### dim_product

| Column | Type | Description |
|---|---|---|
| product_id | TEXT PK | Natural key from source |
| product_name | TEXT | Display name |
| category | TEXT | "Unknown" where missing |
| unit_price | REAL | Current/highest known price |
| supplier_id | TEXT | Supplier identifier |
| is_zero_price | INTEGER | Flag for $0 products |

### fact_sales

| Column | Type | Description |
|---|---|---|
| transaction_id | TEXT PK | Natural key |
| date_key | INTEGER FK | → dim_date |
| store_id | TEXT FK | → dim_store |
| product_id | TEXT FK | → dim_product |
| customer_id | TEXT | NULL for guest transactions |
| quantity | INTEGER | Negative for returns |
| unit_price | REAL | Price at time of transaction |
| total_amount | REAL | Actual amount charged (negative for returns) |
| is_return | INTEGER | Flag |
| is_guest | INTEGER | Flag |
| has_price_mismatch | INTEGER | Flag: total ≠ qty × price |

### Key Modeling Decisions

**Products with multiple prices:** The dimension holds the highest (assumed most recent) price. The fact table carries the `unit_price` from each transaction, preserving what was actually charged. This avoids the need for SCD Type 2 on the product dimension at this scale.

**Returns:** Kept in `fact_sales` with `is_return = 1` and negative `quantity`/`total_amount`. This allows net revenue calculations (Q1) to include returns as reductions, while other queries (Q4) can exclude them via the flag.

**Excluded records:** 16 transactions quarantined — 5 orphaned stores, 3 orphaned products, 5 zero-quantity, 3 future-dated. All preserved in `output/quarantine.csv` with reasons for auditability.

---

## Analytics Approach

All five queries use SQL via `sqlite3`. SQL is the right choice here because:
- The warehouse is already modeled as a star schema designed for SQL queries
- SQL is declarative, auditable, and readable by both engineers and analysts
- Avoids pulling large result sets into Python memory for operations the database handles natively (joins, aggregations, window functions)

**MoM revenue (Q2) — handling missing months:** The query builds a category×month spine so that months with no sales for a category appear explicitly with `NULL` revenue and `NULL` pct_change, rather than being silently skipped by `LAG`. This is a deliberate choice: `NULL` means "no activity" while `0` would mean "we observed zero revenue." If finance required zero-filled periods (e.g., for forecasting), the `LEFT JOIN` would use `COALESCE(revenue, 0)` instead.

---

## Productionization

With more time and a real deployment target, I would add:

**Orchestration:** Airflow or Databricks Workflows with explicit task dependencies (profile → clean → load → analytics). Each phase as a separate task with retry policies and SLA alerts.

**Incremental loads:** Replace full-refresh with watermark-based incremental processing. Use `transaction_date` as the high-water mark. Delta Lake or SQLite WAL mode for ACID guarantees on partial writes.

**Observability:**
- Structured logging with run IDs, timestamps, and row counts at each stage
- Data quality metrics emitted to a monitoring system (row counts, null rates, quarantine counts)
- Alerting on quarantine spikes or row-count anomalies between runs

**Data contracts:** Validate upstream schema on read — assert expected columns, types, and non-null constraints before processing. Fail fast if the source system changes its export format.

**Reference data management:** Replace the hardcoded state-to-region fallback with a managed reference table that the business maintains.

**Testing:** Add integration tests that run the full pipeline against a known fixture dataset and assert end-to-end output. Add data reconciliation checks (source count = clean count + quarantine count + dedup count).

---

## What I'd Do Differently With More Time

- **SCD Type 2 for products** — track price history in the dimension with effective/end dates rather than just keeping the latest price
- **More granular date parsing** — validate ambiguous dates (where day ≤ 12) against transaction patterns or surrounding records rather than assuming US format
- **Quarantine table in SQLite** — persist quarantined records in the warehouse database alongside the star schema for unified querying, in addition to the current CSV output
- **Data reconciliation assertions** — automated check that source rows = clean rows + quarantined rows + deduplicated rows, failing the pipeline if counts don't balance

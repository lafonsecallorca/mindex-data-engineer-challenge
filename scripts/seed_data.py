"""
seed_data.py  —  Mindex Data Engineer / Data Architect Code Challenge

Generates reproducible raw CSV files in data/raw/ with realistic data quality
issues that candidates must discover, document, and handle.

Usage:
    python scripts/seed_data.py
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

TODAY = datetime(2026, 6, 2)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Electronics", "Apparel", "Home & Garden", "Food & Beverage", "Office Supplies"]


# ── Stores ───────────────────────────────────────────────────────────────────

def build_stores() -> pd.DataFrame:
    """
    15 unique stores.  Intentional issues embedded:
      • S003 : malformed zip code (4 digits instead of 5)
      • S007 : near-duplicate row — same store_id, slightly different name
      • S013, S014 : NULL region field
    """
    rows = [
        ("S001", "Eastview Mall",            "Victor",      "NY", "14564", "Northeast", "2010-03-15"),
        ("S002", "Marketplace Mall",         "Rochester",   "NY", "14623", "Northeast", "2008-11-01"),
        ("S003", "Greece Ridge Center",      "Greece",      "NY", "0938",  "Northeast", "2012-06-20"),  # ← 4-digit zip
        ("S004", "Crossroads Center",        "St. Cloud",   "MN", "56301", "Midwest",   "2015-09-10"),
        ("S005", "Mall of America",          "Bloomington", "MN", "55425", "Midwest",   "2011-04-07"),
        ("S006", "Lakeside Shopping Ctr",    "Metairie",    "LA", "70002", "South",     "2009-08-14"),
        ("S007", "Downtown Rochester",       "Rochester",   "NY", "14604", "Northeast", "2006-01-22"),
        ("S007", "Rochester Downtown",       "Rochester",   "NY", "14604", "Northeast", "2006-01-22"),  # ← near-dupe
        ("S008", "Galleria at Crystal Run",  "Middletown",  "NY", "10941", "Northeast", "2014-02-18"),
        ("S009", "Tucson Mall",              "Tucson",      "AZ", "85705", "West",      "2013-07-30"),
        ("S010", "Park Place Mall",          "Tucson",      "AZ", "85711", "West",      "2016-10-05"),
        ("S011", "Southpark Meadows",        "Austin",      "TX", "78748", "South",     "2017-03-12"),
        ("S012", "The Domain",               "Austin",      "TX", "78758", "South",     "2018-11-28"),
        ("S013", "Cascade Station",          "Portland",    "OR", "97220", None,        "2019-05-14"),  # ← NULL region
        ("S014", "Lloyd Center",             "Portland",    "OR", "97232", None,        "2007-09-09"),  # ← NULL region
        ("S015", "Alderwood Mall",           "Lynnwood",    "WA", "98036", "West",      "2020-01-15"),
    ]
    return pd.DataFrame(
        rows,
        columns=["store_id", "store_name", "city", "state", "zip_code", "region", "opened_date"],
    )


# ── Products ─────────────────────────────────────────────────────────────────

def build_products() -> pd.DataFrame:
    """
    30 unique products.  Intentional issues embedded:
      • P005 : second row with a different unit_price (undocumented price change)
      • P012 : exact duplicate row (bad data extract)
      • P003, P009, P016, P023, P029 : NULL category
      • P027 : unit_price = 0.00
    """
    price_rng = np.random.default_rng(SEED + 10)
    base = []
    for i in range(1, 31):
        pid = f"P{i:03d}"
        cat = CATEGORIES[(i - 1) % len(CATEGORIES)]
        price = round(float(price_rng.uniform(8.99, 249.99)), 2)
        supplier = f"SUP{((i - 1) % 5) + 1:03d}"
        base.append((pid, f"Product {pid}", cat, price, supplier))

    df = pd.DataFrame(base, columns=["product_id", "product_name", "category", "unit_price", "supplier_id"])

    # Issue: undocumented price increase for P005
    p005_new = df.loc[df.product_id == "P005"].iloc[0].copy()
    p005_new["unit_price"] = round(p005_new["unit_price"] + 8.50, 2)
    df = pd.concat([df, pd.DataFrame([p005_new])], ignore_index=True)

    # Issue: exact duplicate for P012
    p012_dup = df.loc[df.product_id == "P012"].iloc[0].copy()
    df = pd.concat([df, pd.DataFrame([p012_dup])], ignore_index=True)

    # Issue: NULL categories
    df.loc[df.product_id.isin(["P003", "P009", "P016", "P023", "P029"]), "category"] = None

    # Issue: zero unit price
    df.loc[df.product_id == "P027", "unit_price"] = 0.00

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ── Transactions ─────────────────────────────────────────────────────────────

def build_transactions() -> pd.DataFrame:
    """
    460 base rows + injected issues = ~505 total rows.

    Issue                                Count   Source rows (pre-shuffle)
    ─────────────────────────────────────────────────────────────────────
    Mixed date formats (US/EU)              20    0 – 19
    String-formatted amounts ($X.XX)        25    20 – 44
    Price mismatches (silent discount)      20    100 – 119
    Orphaned store_id (not in stores)        5    150 – 154
    Orphaned product_id (not in products)    3    155 – 157
    NULL customer_id (guest transactions)   40    200 – 239
    Zero-quantity rows                       5    250 – 254
    Future-dated rows                        3    260 – 262
    Exact duplicate rows                    15    copies of 50 – 64
    Return transactions (neg qty/amount)    30    copies of 65 – 94, new IDs
    """
    valid_stores = [f"S{i:03d}" for i in range(1, 16)]
    valid_products = [f"P{i:03d}" for i in range(1, 31)]
    customers = [f"CUST{i:04d}" for i in range(1, 301)]

    # Derive product prices with the same RNG as build_products() so they match
    price_rng = np.random.default_rng(SEED + 10)
    product_prices: dict[str, float] = {}
    for i in range(1, 31):
        product_prices[f"P{i:03d}"] = round(float(price_rng.uniform(8.99, 249.99)), 2)

    tx_rng = np.random.default_rng(SEED + 20)

    def rand_date() -> str:
        return (TODAY - timedelta(days=int(tx_rng.integers(1, 90)))).strftime("%Y-%m-%d")

    def pick(lst: list) -> str:
        return lst[int(tx_rng.integers(0, len(lst)))]

    # --- Generate 460 clean base rows ---
    rows = []
    for i in range(460):
        product = pick(valid_products)
        qty = int(tx_rng.integers(1, 6))
        price = product_prices[product]
        rows.append({
            "transaction_id":   f"TXN{10001 + i}",
            "transaction_date": rand_date(),
            "store_id":         pick(valid_stores),
            "product_id":       product,
            "customer_id":      pick(customers),
            "quantity":         qty,
            "unit_price":       price,
            "total_amount":     round(qty * price, 2),
        })

    df = pd.DataFrame(rows)

    # --- Inject issues (non-overlapping index ranges) ---

    # 1. Mixed date formats on rows 0–19
    for i in range(10):
        dt = datetime.strptime(df.at[i, "transaction_date"], "%Y-%m-%d")
        df.at[i, "transaction_date"] = dt.strftime("%m/%d/%Y")      # MM/DD/YYYY
    for i in range(10, 20):
        dt = datetime.strptime(df.at[i, "transaction_date"], "%Y-%m-%d")
        df.at[i, "transaction_date"] = dt.strftime("%d-%m-%Y")      # DD-MM-YYYY

    # 2. String-formatted amounts on rows 20–44
    df["total_amount"] = df["total_amount"].astype(object)
    for i in range(20, 45):
        df.at[i, "total_amount"] = f"${df.at[i, 'total_amount']:.2f}"

    # 3. Silent discount: total_amount ≠ qty × unit_price on rows 100–119
    discount_rng = np.random.default_rng(SEED + 30)
    for i in range(100, 120):
        pct_off = round(float(discount_rng.uniform(0.05, 0.20)), 3)
        df.at[i, "total_amount"] = round(df.at[i, "total_amount"] * (1 - pct_off), 2)

    # 4. Orphaned store_ids on rows 150–154
    for i, sid in enumerate(["S016", "S017", "S018", "S016", "S019"]):
        df.at[150 + i, "store_id"] = sid

    # 5. Orphaned product_ids on rows 155–157
    for i, pid in enumerate(["P031", "P032", "P031"]):
        df.at[155 + i, "product_id"] = pid

    # 6. NULL customer_ids (guest transactions) on rows 200–239
    df.loc[200:239, "customer_id"] = None

    # 7. Zero-quantity rows on rows 250–254
    df.loc[250:254, ["quantity", "total_amount"]] = 0

    # 8. Future dates on rows 260–262
    for i, days_fwd in enumerate([8, 16, 25]):
        df.at[260 + i, "transaction_date"] = (TODAY + timedelta(days=days_fwd)).strftime("%Y-%m-%d")

    # 9. Exact duplicates — append copies of rows 50–64 (all clean; same TXN IDs)
    df = pd.concat([df, df.iloc[50:65].copy()], ignore_index=True)

    # 10. Returns — copy rows 65–94 (all clean), negate qty/amount, new TXN IDs
    returns = df.iloc[65:95].copy()
    returns["transaction_id"] = [f"TXN{20001 + i}" for i in range(30)]
    returns["quantity"] = -(returns["quantity"].astype(int))
    returns["total_amount"] = -(returns["total_amount"].astype(float)).round(2)
    df = pd.concat([df, returns], ignore_index=True)

    return df.sample(frac=1, random_state=SEED + 50).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    builders = [
        ("stores",       build_stores),
        ("products",     build_products),
        ("transactions", build_transactions),
    ]
    for name, fn in builders:
        df = fn()
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {path.name}: {len(df)} rows written")
    print(f"\nFiles written to: {DATA_DIR}")

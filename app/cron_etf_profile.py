import os
import orjson
from tqdm import tqdm
import pandas as pd
import sqlite3
import numpy as np

# ---------- IO helpers ----------

def load_prices(symbol, base_dir="json/historical-price/adj"):
    """
    Load JSON list of dicts with keys including 'date' and 'adjClose',
    return a sorted DataFrame with parsed dates.
    """
    path = os.path.join(base_dir, f"{symbol}.json")
    with open(path, "rb") as f:
        data = orjson.loads(f.read())

    df = pd.DataFrame(data)
    if "date" not in df or "adjClose" not in df:
        raise ValueError(f"{symbol}: required keys 'date'/'adjClose' not found")

    # Sort by date ascending and ensure proper dtype
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Ensure numeric
    df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
    df = df.dropna(subset=["adjClose"])
    return df[["date", "adjClose"]]


def save_json(symbol, data):
    base_dir = "json/etf/profile/"
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{symbol}.json")
    with open(path, "wb") as f:
        f.write(orjson.dumps(data))


# ---------- Core math ----------

def compute_beta_vs_spy(symbol, min_overlap=60):
    """
    Returns (beta, n, start_date, end_date)
    beta is rounded to 2 decimals or None if not computable.
    """
    try:
        df = load_prices(symbol, base_dir="json/historical-price/adj")
    except FileNotFoundError:
        return (None, 0, None, None)
    except Exception:
        return (None, 0, None, None)

    # Compute simple daily returns
    try:
        # Avoid division-by-zero inf: treat 0 closes as missing
        df2 = df.copy()
        df2["adjClose"] = df2["adjClose"].replace(0, np.nan)
        df2["r"] = df2["adjClose"].pct_change()

        spy_df_local = spy_df.copy()
        spy_df_local["adjClose"] = spy_df_local["adjClose"].replace(0, np.nan)
        spy_df_local["r_spy"] = spy_df_local["adjClose"].pct_change()

        # Align by date (inner join) and drop NaN rows from either column
        merged = pd.merge(df2[["date", "r"]],
                          spy_df_local[["date", "r_spy"]],
                          on="date", how="inner")

        # Replace ±Inf with NaN then drop
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["r", "r_spy"])
        n = len(merged)
        start = merged["date"].min() if n else None
        end = merged["date"].max() if n else None

        if n < min_overlap:
            return (None, n, start, end)

        var_spy = merged["r_spy"].var(ddof=1)
        if var_spy is None or not np.isfinite(var_spy) or np.isclose(var_spy, 0.0):
            return (None, n, start, end)

        cov = merged["r"].cov(merged["r_spy"])
        if cov is None or not np.isfinite(cov):
            return (None, n, start, end)

        beta = round(float(cov / var_spy), 2)
        return (beta, n, start, end)
    except Exception:
        return (None, 0, None, None)


# ---------- Optional: ETF metadata (unchanged) ----------

def get_data(ticker):
    query = """
        SELECT 
            profile, etfProvider
        FROM 
            etfs
        WHERE
            symbol = ?
    """
    cur = etf_con.cursor()
    cur.execute(query, (ticker,))
    result = cur.fetchone()
    res = []

    try:
        if result is not None:
            res = orjson.loads(result[0])
            for item in res:
                item['etfProvider'] = result[1]
    except Exception as e:
        print(f"[{ticker}] profile parse error:", e)
        res = []
    return res


# ---------- Main ----------

if __name__ == "__main__":
    os.makedirs("json/historical-price/adj", exist_ok=True)

    # Load SPY once (sorted, parsed)
    spy_df = load_prices("SPY", base_dir="json/historical-price/adj")

    # Connect to ETF DB and get symbols
    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")

    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    # Example for testing:
    # etf_symbols = ['QQQ']

    for symbol in tqdm(etf_symbols):
        beta, n, start, end = compute_beta_vs_spy(symbol)

        # Attach beta into first profile item if present; otherwise skip saving
        data = get_data(symbol)
        if isinstance(data, list) and data:
            # Keep extra context; adjust if you only want 'beta'
            data[0]["beta"] = beta
            save_json(symbol, data)

    etf_con.close()

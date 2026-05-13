from __future__ import annotations

import pandas as pd

from .config import TrackerConfig


def load_bse_bulk_block_csv(path: str) -> pd.DataFrame:
    """Best-effort ingestion for manually downloaded BSE bulk/block CSV.

    The BSE site is brittle; this helper accepts a CSV saved by the user and
    normalizes common columns. It will not attempt CAPTCHAs.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "Date": "date",
        "Scrip Code": "symbol",
        "Client Name": "client_name",
        "Deal Type": "deal_type",
        "Buy/Sell": "buy_sell",
        "Quantity": "quantity",
        "Price": "price",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["date", "symbol", "client_name", "deal_type", "buy_sell", "quantity", "price"]:
        if col not in df.columns:
            df[col] = None
    df["source"] = "BSE"
    df["exchange"] = "BSE"
    return df[["date", "symbol", "client_name", "deal_type", "buy_sell", "quantity", "price", "source", "exchange"]]


def filter_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sym = symbol.upper()
    return df[df["symbol"].astype(str).str.upper() == sym]


__all__ = ["load_bse_bulk_block_csv", "filter_symbol"]

from __future__ import annotations

import pandas as pd


def compute_vwap(df: pd.DataFrame, price_source: str = "hlc3") -> pd.Series:
    src = {
        "close": df["close"],
        "hlc3": (df["high"] + df["low"] + df["close"]) / 3,
        "ohlc4": (df["open"] + df["high"] + df["low"] + df["close"]) / 4,
    }.get(price_source, (df["high"] + df["low"] + df["close"]) / 3)
    cum_pv = (src * df["volume"]).groupby(df["datetime"].dt.date).cumsum()
    cum_v = df["volume"].groupby(df["datetime"].dt.date).cumsum()
    # Avoid division by zero when cum_v is 0 (no volume traded)
    return cum_pv.div(cum_v).replace([float("inf"), -float("inf")], float("nan"))


def compute_ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def opening_range(df: pd.DataFrame, start: str = "09:15", end: str = "09:30") -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    rows = []
    for d, g in df.groupby("date"):
        mask = (g["datetime"].dt.time >= pd.to_datetime(start).time()) & (g["datetime"].dt.time <= pd.to_datetime(end).time())
        window = g[mask]
        rows.append({
            "date": d,
            "opening_range_high": window["high"].max() if not window.empty else pd.NA,
            "opening_range_low": window["low"].min() if not window.empty else pd.NA,
        })
    return pd.DataFrame(rows)

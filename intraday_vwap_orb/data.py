from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import StrategyConfig
from .indicators import compute_ema, compute_vwap, opening_range


REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]


@dataclass
class IntradayData:
    df: pd.DataFrame
    symbol: str


def load_intraday(path: Path | str, symbol: Optional[str] = None) -> IntradayData:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Intraday file not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    lower = {c.lower(): c for c in df.columns}
    for col in REQUIRED_COLS:
        if col not in lower:
            raise ValueError(f"Missing required column '{col}' in {p}")
    df = df.rename(columns={v: k for k, v in lower.items()})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    sym = symbol or (df["symbol"].iloc[0] if "symbol" in df.columns and df["symbol"].notna().any() else "UNKNOWN")
    df["symbol"] = sym
    if df.duplicated(subset=["datetime"]).any():
        df = df.drop_duplicates(subset=["datetime"])
    return IntradayData(df=df, symbol=sym)


def ensure_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")
    freq = df["datetime"].diff().median()
    if freq is not None and freq.total_seconds() <= 65:  # allow slight jitter around 1m
        df = df.set_index("datetime").resample("5T", label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "symbol": "first",
        }).dropna(subset=["open", "high", "low", "close", "volume"]).reset_index()
    return df


def prepare_intraday(df: pd.DataFrame, cfg: StrategyConfig, from_date: Optional[str] = None, to_date: Optional[str] = None) -> pd.DataFrame:
    out = ensure_5m(df)
    if from_date:
        out = out[out["datetime"] >= pd.to_datetime(from_date)]
    if to_date:
        out = out[out["datetime"] <= pd.to_datetime(to_date) + pd.Timedelta(days=1)]
    out = out.copy()
    out["vwap"] = compute_vwap(out, cfg.vwap_price_source)
    out["ema20"] = compute_ema(out["close"], cfg.ema_period)
    out["date"] = out["datetime"].dt.date
    out["time"] = out["datetime"].dt.time
    or_df = opening_range(out, cfg.opening_range_start, cfg.opening_range_end)
    out = out.merge(or_df, on="date", how="left")
    out["prev5_avg_vol"] = out["volume"].rolling(cfg.volume_lookback_bars).mean()
    # Safe division: avoid inf/NaN when prev5_avg_vol is 0 or NaN
    prev5_avg = out["prev5_avg_vol"].replace(0, float("nan"))
    out["vol_ratio"] = out["volume"] / prev5_avg
    return out


def split_days(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    grouped = {}
    for date, g in df.groupby("date"):
        grouped[str(date)] = g.reset_index(drop=True)
    return grouped

"""
Data Feeds — Unified OHLCV Fetcher
====================================
Priority chain for every fetch request:
  1. TradingView  (tvdatafeed — FREE, no API key, works for NSE/BSE intraday + daily)
  2. Dhan         (dhanhq SDK — requires DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN in secrets)
  3. Zerodha Kite (kiteconnect SDK — requires KITE_API_KEY + KITE_ACCESS_TOKEN in secrets)
  4. yfinance     (free fallback — may be blocked on cloud, limited intraday history)

Usage
-----
    from data_feeds import fetch_ohlcv, FeedSource

    df, source, err = fetch_ohlcv("MOIL.NS", interval="5m", days=1)
    # df   : pd.DataFrame with columns [open, high, low, close, volume], DatetimeIndex
    # source : FeedSource enum
    # err  : str | None — human-readable error if all feeds failed

TradingView (priority 1 — no credentials needed):
    Uses the open-source `tvdatafeed` library which scrapes TradingView websocket.
    Install: pip install tvdatafeed
    Supports: NSE, BSE equities + indices. Intervals: 1m, 5m, 15m, 1h, 1d.
    Note: TradingView login is NOT required for free/public symbols.

Broker credentials (optional, set via .streamlit/secrets.toml or env vars):
    [dhan]
    client_id    = "..."
    access_token = "..."

    [kite]
    api_key      = "..."
    access_token = "..."
"""
from __future__ import annotations

import os
import datetime
from enum import Enum
from typing import Optional, Tuple

import pandas as pd


# ────────────────────────────────────────────────────────────────────────────────
class FeedSource(str, Enum):
    TRADINGVIEW = "TradingView"
    DHAN        = "Dhan"
    ZERODHA     = "Zerodha Kite"
    YFINANCE    = "yfinance"
    UPLOAD      = "Upload (CSV/Parquet)"
    UNAVAILABLE = "unavailable"


_EMPTY = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, ensure DatetimeIndex, standard OHLCV names."""
    if df is None or df.empty:
        return _EMPTY.copy()
    df = df.copy()
    df.columns = [
        c[0].lower() if isinstance(c, tuple) else c.lower()
        for c in df.columns
    ]
    # reset index if datetime is a column
    if df.index.name and "date" in df.index.name.lower():
        df = df.reset_index()
    dt_col = next(
        (c for c in df.columns if "date" in c or "time" in c),
        None,
    )
    if dt_col and dt_col != df.index.name:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.set_index(dt_col).sort_index()
    # keep only OHLCV
    rename = {}
    for col in df.columns:
        for std in ["open", "high", "low", "close", "volume"]:
            if std in col:
                rename[col] = std
                break
    df = df.rename(columns=rename)
    out_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[out_cols].dropna(subset=["close"])


def _get_secret(section: str, key: str) -> Optional[str]:
    """Read from Streamlit secrets or environment variable."""
    # Streamlit secrets
    try:
        import streamlit as st
        val = st.secrets.get(section, {}).get(key)
        if val:
            return str(val)
    except Exception:
        pass
    # Environment variable: DHAN_CLIENT_ID, KITE_API_KEY, etc.
    env_key = f"{section.upper()}_{key.upper()}"
    return os.environ.get(env_key)


# ────────────────────────────────────────────────────────────────────────────────
# Feed 1 — TradingView (tvdatafeed) — FREE, no credentials required
# ────────────────────────────────────────────────────────────────────────────────

# Map Yahoo-style ticker → (TV symbol, TV exchange)
_TV_SYMBOL_MAP = {
    "MOIL.NS":      ("MOIL",         "NSE"),
    "JSWSTEEL.NS":  ("JSWSTEEL",     "NSE"),
    "SAIL.NS":      ("SAIL",         "NSE"),
    "TATASTEEL.NS": ("TATASTEEL",    "NSE"),
    "HINDZINC.NS":  ("HINDZINC",     "NSE"),
    "^CNXMETAL":    ("NIFTY_METAL",  "NSE"),
    "^NSEI":        ("NIFTY",        "NSE"),
    "^BSESN":       ("SENSEX",       "BSE"),
    "BZ=F":         ("UKOIL",        "NYMEX"),
    "INR=X":        ("USDINR",       "FX_IDC"),
}

_TV_INTERVAL_MAP = {
    "1m":  "in_1_minute",
    "5m":  "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "1h":  "in_1_hour",
    "1d":  "in_daily",
    "1wk": "in_weekly",
    "1mo": "in_monthly",
}


def _fetch_tradingview(
    ticker: str,
    interval: str,
    n_bars: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV via tvdatafeed (TradingView websocket scraper).
    No login or API key needed for free/public symbols on NSE/BSE.
    """
    try:
        from tvDatafeed import TvDatafeed, Interval  # type: ignore  # tvdatafeed-enhanced
    except ImportError:
        try:
            from tvdatafeed import TvDatafeed, Interval  # type: ignore  # legacy
        except ImportError:
            raise ImportError("tvdatafeed not installed. Run: pip install tvdatafeed-enhanced")

    symbol_info = _TV_SYMBOL_MAP.get(ticker)
    if symbol_info is None:
        # Try to extract symbol from Yahoo-style ticker (e.g. "MOIL.NS" -> "MOIL")
        sym = ticker.split(".")[0].replace("^", "")
        exchange = "NSE"
        symbol_info = (sym, exchange)

    symbol, exchange = symbol_info

    tv_interval_str = _TV_INTERVAL_MAP.get(interval, "in_daily")
    tv_interval = getattr(Interval, tv_interval_str, Interval.in_daily)

    tv = TvDatafeed()  # no username/password = anonymous (free tier)
    df = tv.get_hist(
        symbol=symbol,
        exchange=exchange,
        interval=tv_interval,
        n_bars=n_bars,
    )
    if df is None or df.empty:
        raise ValueError(f"TradingView returned no data for {symbol}:{exchange} {interval}")

    df = df.rename(columns=str.lower)
    df.index.name = "datetime"
    return df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]


# ────────────────────────────────────────────────────────────────────────────────
# Feed 2 — Dhan
# ─────────────────────────────────────────────────────────────────────────────

# Dhan instrument map for common MOIL symbols
_DHAN_SECURITY_MAP = {
    "MOIL.NS":     {"security_id": "10576", "exchange_segment": "NSE_EQ"},
    "^CNXMETAL":   {"security_id": "13",    "exchange_segment": "IDX_I"},
    "JSWSTEEL.NS": {"security_id": "3001",  "exchange_segment": "NSE_EQ"},
    "SAIL.NS":     {"security_id": "4963",  "exchange_segment": "NSE_EQ"},
    "TATASTEEL.NS":{"security_id": "14977", "exchange_segment": "NSE_EQ"},
}

_DHAN_INTERVAL_MAP = {
    "1m":  "1",
    "5m":  "5",
    "15m": "15",
    "30m": "30",
    "1h":  "60",
    "1d":  "D",
}


def _fetch_dhan(
    ticker: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV from Dhan Historical Data API."""
    client_id    = _get_secret("dhan", "client_id")
    access_token = _get_secret("dhan", "access_token")
    if not client_id or not access_token:
        raise ValueError("Dhan credentials not configured.")

    try:
        from dhanhq import dhanhq  # type: ignore
    except ImportError:
        raise ImportError("dhanhq SDK not installed. Run: pip install dhanhq")

    meta = _DHAN_SECURITY_MAP.get(ticker)
    if not meta:
        raise ValueError(f"Dhan: no security_id mapping for ticker '{ticker}'.")

    dhan_interval = _DHAN_INTERVAL_MAP.get(interval)
    if not dhan_interval:
        raise ValueError(f"Dhan: unsupported interval '{interval}'.")

    client = dhanhq(client_id, access_token)

    if dhan_interval == "D":
        resp = client.historical_daily_data(
            symbol=meta["security_id"],
            exchange_segment=meta["exchange_segment"],
            instrument_type="EQUITY",
            expiry_code=0,
            from_date=from_date,
            to_date=to_date,
        )
    else:
        resp = client.intraday_minute_data(
            security_id=meta["security_id"],
            exchange_segment=meta["exchange_segment"],
            instrument_type="EQUITY",
            interval=dhan_interval,
            from_date=from_date,
            to_date=to_date,
        )

    if resp.get("status") != "success":
        raise ValueError(f"Dhan API error: {resp.get('remarks', resp)}")

    data = resp.get("data", {})
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data.get("timestamp", []), unit="s", utc=True),
        "open":     data.get("open",   []),
        "high":     data.get("high",   []),
        "low":      data.get("low",    []),
        "close":    data.get("close",  []),
        "volume":   data.get("volume", []),
    }).set_index("datetime").sort_index()
    return df


# ────────────────────────────────────────────────────────────────────────────────
# Feed 3 — Zerodha Kite (optional broker feed)
# ─────────────────────────────────────────────────────────────────────────────

_KITE_INSTRUMENT_MAP = {
    "MOIL.NS":     "NSE:MOIL",
    "JSWSTEEL.NS": "NSE:JSWSTEEL",
    "SAIL.NS":     "NSE:SAIL",
    "TATASTEEL.NS":"NSE:TATASTEEL",
    "^CNXMETAL":   "NSE:NIFTY METAL",
}

_KITE_INTERVAL_MAP = {
    "1m":  "minute",
    "5m":  "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h":  "60minute",
    "1d":  "day",
}


def _fetch_kite(
    ticker: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV from Zerodha Kite historical data API."""
    api_key      = _get_secret("kite", "api_key")
    access_token = _get_secret("kite", "access_token")
    if not api_key or not access_token:
        raise ValueError("Zerodha Kite credentials not configured.")

    try:
        from kiteconnect import KiteConnect  # type: ignore
    except ImportError:
        raise ImportError("kiteconnect SDK not installed. Run: pip install kiteconnect")

    tradingsymbol = _KITE_INSTRUMENT_MAP.get(ticker)
    if not tradingsymbol:
        raise ValueError(f"Kite: no instrument mapping for ticker '{ticker}'.")

    kite_interval = _KITE_INTERVAL_MAP.get(interval)
    if not kite_interval:
        raise ValueError(f"Kite: unsupported interval '{interval}'.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Resolve instrument token
    instruments = kite.instruments("NSE")
    token = None
    sym = tradingsymbol.replace("NSE:", "")
    for inst in instruments:
        if inst["tradingsymbol"] == sym:
            token = inst["instrument_token"]
            break
    if token is None:
        raise ValueError(f"Kite: instrument token not found for '{sym}'.")

    records = kite.historical_data(
        instrument_token=token,
        from_date=from_date,
        to_date=to_date,
        interval=kite_interval,
        continuous=False,
        oi=False,
    )
    df = pd.DataFrame(records)
    df = df.rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


# ────────────────────────────────────────────────────────────────────────────────
# Feed 4 — yfinance (last-resort free fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yfinance(
    ticker: str,
    interval: str,
    from_date: str,
    to_date: str,
    period: Optional[str] = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        raise ImportError("yfinance not installed.")

    if period:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
    else:
        raw = yf.download(ticker, start=from_date, end=to_date,
                          interval=interval, progress=False, auto_adjust=True)

    if raw is None or raw.empty:
        # Retry with period fallback for intraday (date ranges can fail for recent data)
        if interval in ("1m", "5m", "15m", "30m", "1h") and not period:
            raw = yf.download(ticker, period="5d", interval=interval,
                              progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}.")
    # Flatten multi-level columns from yfinance >=0.2.x
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    return raw


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str,
    interval: str = "1d",
    days: int = 365,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    period: Optional[str] = None,
) -> Tuple[pd.DataFrame, FeedSource, Optional[str]]:
    """
    Fetch OHLCV using priority chain: TradingView → Dhan → Zerodha → yfinance.

    Parameters
    ----------
    ticker    : Yahoo-style ticker e.g. "MOIL.NS", "^CNXMETAL"
    interval  : "1m", "5m", "15m", "1h", "1d"
    days      : lookback days (used if from_date/to_date not provided)
    from_date : "YYYY-MM-DD" (overrides days)
    to_date   : "YYYY-MM-DD" (overrides days)
    period    : yfinance period string e.g. "1y", "5y" (yfinance only)

    Returns
    -------
    (df, source, error_string_or_None)
    df is normalised: lowercase columns, DatetimeIndex, [open,high,low,close,volume]
    """
    today = datetime.date.today()
    if from_date is None:
        from_date = (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = today.strftime("%Y-%m-%d")

    errors: list[str] = []

    # Estimate n_bars from date range for TradingView
    try:
        _td = datetime.date.fromisoformat(to_date) - datetime.date.fromisoformat(from_date)
        _days_req = max(_td.days, 1)
    except Exception:
        _days_req = days
    _tv_bars = {
        "1m":  _days_req * 390,  # ~390 min/day NSE
        "5m":  _days_req * 78,
        "15m": _days_req * 26,
        "30m": _days_req * 13,
        "1h":  _days_req * 7,
        "1d":  _days_req + 5,
        "1wk": max(_days_req // 5 + 2, 10),
    }.get(interval, 500)

    # ── 1. TradingView (free, no credentials) ──────────────────────────────────────
    try:
        df = _fetch_tradingview(ticker, interval, n_bars=min(_tv_bars, 5000))
        df = _normalise(df)
        # Filter to requested date range
        if not df.empty:
            _lo = pd.Timestamp(from_date)
            _hi = pd.Timestamp(to_date) + pd.Timedelta(days=1)
            df = df[(df.index >= _lo) & (df.index <= _hi)]
        if not df.empty:
            return df, FeedSource.TRADINGVIEW, None
        errors.append("TradingView: empty after date filter")
    except Exception as e:
        errors.append(f"TradingView: {e}")

    # ── 2. Dhan (if credentials configured) ──────────────────────────────────────
    try:
        df = _fetch_dhan(ticker, interval, from_date, to_date)
        df = _normalise(df)
        if not df.empty:
            return df, FeedSource.DHAN, None
        errors.append("Dhan: empty result")
    except Exception as e:
        errors.append(f"Dhan: {e}")

    # ── 3. Zerodha Kite (if credentials configured) ─────────────────────────────
    try:
        df = _fetch_kite(ticker, interval, from_date, to_date)
        df = _normalise(df)
        if not df.empty:
            return df, FeedSource.ZERODHA, None
        errors.append("Zerodha: empty result")
    except Exception as e:
        errors.append(f"Zerodha: {e}")

    # ── 4. yfinance (last-resort free fallback) ─────────────────────────────────
    try:
        df = _fetch_yfinance(ticker, interval, from_date, to_date, period=period)
        df = _normalise(df)
        if not df.empty:
            return df, FeedSource.YFINANCE, None
        errors.append("yfinance: empty result")
    except Exception as e:
        errors.append(f"yfinance: {e}")

    # ── All failed ────────────────────────────────────────────────────────────────────
    summary = " | ".join(errors)
    return _EMPTY.copy(), FeedSource.UNAVAILABLE, summary


def fetch_daily(
    ticker: str,
    years: int = 2,
    period: Optional[str] = None,
) -> Tuple[pd.DataFrame, FeedSource, Optional[str]]:
    """Convenience wrapper for daily OHLCV."""
    return fetch_ohlcv(
        ticker, interval="1d",
        days=years * 365,
        period=period or f"{years}y",
    )


def source_badge(source: FeedSource) -> str:
    """Return a coloured emoji badge for display."""
    return {
        FeedSource.TRADINGVIEW : "🟢 TradingView (live)",
        FeedSource.DHAN        : "🟢 Dhan",
        FeedSource.ZERODHA     : "🟢 Zerodha Kite",
        FeedSource.YFINANCE    : "🟡 yfinance (fallback)",
        FeedSource.UPLOAD      : "🔵 Upload (CSV/Parquet)",
        FeedSource.UNAVAILABLE : "🔴 No data",
    }.get(source, str(source))

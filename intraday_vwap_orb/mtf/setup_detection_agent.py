"""
Setup Detection Agent
---------------------
Identifies valid trade setups on the daily timeframe.

Supported setups:
  - pullback   : RSI 38–52 in confirmed uptrend, price near EMA20/EMA50
  - breakout   : price closes above recent resistance + volume spike
  - reversal   : price near support + bullish engulfing / hammer candle
  - none       : no valid setup

Output (JSON-serialisable dict):
{
  "setup_type"   : "pullback" | "breakout" | "reversal" | "none",
  "valid"        : bool,
  "setup_quality": 0-100,
  "trigger_zone" : {"low": float, "high": float},
  "rsi"          : float,
  "volume_ratio" : float,
  "candle_pattern": str,
  "log"          : [str, ...]
}
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0


def _volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    if "volume" not in df.columns or len(df) < lookback + 1:
        return 1.0
    avg_vol = float(df["volume"].iloc[-(lookback + 1):-1].mean())
    cur_vol = float(df["volume"].iloc[-1])
    return cur_vol / avg_vol if avg_vol > 0 else 1.0


def _candle_pattern(df: pd.DataFrame) -> str:
    """Classify the last candle."""
    if len(df) < 2:
        return "none"
    o1, h1, l1, c1 = df["open"].iloc[-2], df["high"].iloc[-2], df["low"].iloc[-2], df["close"].iloc[-2]
    o0, h0, l0, c0 = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    body0 = abs(c0 - o0)
    range0 = h0 - l0 if h0 > l0 else 1e-9
    lower_wick = min(o0, c0) - l0
    upper_wick = h0 - max(o0, c0)

    # Bullish engulfing
    if c1 < o1 and c0 > o0 and c0 > o1 and o0 < c1:
        return "bullish_engulfing"
    # Bearish engulfing
    if c1 > o1 and c0 < o0 and c0 < o1 and o0 > c1:
        return "bearish_engulfing"
    # Hammer (bullish reversal)
    if lower_wick > 2 * body0 and upper_wick < 0.3 * range0 and c0 > o0:
        return "hammer"
    # Shooting star
    if upper_wick > 2 * body0 and lower_wick < 0.3 * range0 and c0 < o0:
        return "shooting_star"
    # Doji
    if body0 < 0.1 * range0:
        return "doji"
    return "none"


def run_setup_detection_agent(
    daily_df: pd.DataFrame,
    htf_result: Optional[dict] = None,
) -> dict:
    """
    Parameters
    ----------
    daily_df   : DataFrame with [open, high, low, close, volume]
    htf_result : output from run_htf_trend_agent (used for trend direction + levels)

    Returns
    -------
    dict — see module docstring
    """
    log: list[str] = []
    result: dict = {
        "setup_type": "none",
        "valid": False,
        "setup_quality": 0,
        "trigger_zone": {"low": 0.0, "high": 0.0},
        "rsi": 50.0,
        "volume_ratio": 1.0,
        "candle_pattern": "none",
        "log": log,
    }

    if daily_df is None or daily_df.empty:
        log.append("ERROR: daily_df empty")
        return result

    df = daily_df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        log.append("ERROR: 'close' column missing")
        return result

    close = df["close"].dropna()
    price = float(close.iloc[-1])

    # ── Indicators ───────────────────────────────────────────────────────────
    rsi = _rsi(close)
    vol_ratio = _volume_ratio(df)
    candle = _candle_pattern(df)
    ema20 = float(_ema(close, 20).iloc[-1])
    ema50 = float(_ema(close, 50).iloc[-1]) if len(close) >= 50 else ema20

    # Fix #5: ATR-relative near_ema threshold (replaces fixed 1.5%/2% that was too tight)
    if "high" in df.columns and "low" in df.columns:
        _hi = df["high"].dropna(); _lo = df["low"].dropna()
        _tr = pd.concat([
            _hi - _lo,
            (_hi - close.shift(1)).abs(),
            (_lo - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        _atr_val = float(_tr.rolling(14).mean().iloc[-1]) if len(_tr) >= 14 else float(_tr.mean())
    else:
        _atr_val = float(close.std()) or (price * 0.02)
    _atr_val = max(_atr_val, price * 0.005)  # floor at 0.5% of price

    result["rsi"] = round(rsi, 2)
    result["volume_ratio"] = round(vol_ratio, 2)
    result["candle_pattern"] = candle
    log.append(f"Price={price:.2f}  RSI={rsi:.1f}  VolRatio={vol_ratio:.2f}x  Candle={candle}")

    trend = htf_result.get("trend", "neutral") if htf_result else "neutral"
    resistance = htf_result.get("key_levels", {}).get("resistance", price * 1.05) if htf_result else price * 1.05
    support    = htf_result.get("key_levels", {}).get("support",    price * 0.95) if htf_result else price * 0.95
    log.append(f"HTF trend={trend}  Support={support:.2f}  Resistance={resistance:.2f}")

    # ── Setup rules ──────────────────────────────────────────────────────────
    setup_type = "none"
    quality = 0
    trigger_low = price * 0.99
    trigger_high = price * 1.01

    # 1. Pullback setup — price within 1.5 ATR of EMA20 or EMA50
    near_ema = abs(price - ema20) < 1.5 * _atr_val or abs(price - ema50) < 1.5 * _atr_val
    if trend == "bullish" and 35 <= rsi <= 55 and near_ema:
        setup_type = "pullback"
        quality = 50
        quality += min(20, int((55 - abs(rsi - 45)) * 2))   # sweet spot RSI ~45
        quality += 10 if vol_ratio < 1.0 else 0             # low vol pullback = healthy
        quality += 15 if candle in ("hammer", "bullish_engulfing") else 0
        trigger_low  = min(ema20, ema50) * 0.995
        trigger_high = max(ema20, ema50) * 1.015
        log.append(f"PULLBACK: near_ema={near_ema}  quality={quality}")

    # 2. Breakout setup — price closes above resistance + volume spike
    elif price > resistance * 0.999 and vol_ratio >= 1.4:
        setup_type = "breakout"
        quality = 55
        quality += min(20, int((vol_ratio - 1.4) * 25))    # more vol → higher quality
        quality += 10 if candle in ("bullish_engulfing",) else 0
        quality += 10 if rsi > 55 else 0                   # momentum confirmation
        quality += 5  if trend == "bullish" else 0
        trigger_low  = resistance * 0.998
        trigger_high = resistance * 1.025
        log.append(f"BREAKOUT: price>{resistance:.2f}  vol_ratio={vol_ratio:.2f}x  quality={quality}")

    # 3. Reversal setup — near support + bullish candle + oversold
    # Fix #5b: require structure is NOT a confirmed downtrend (no counter-trend reversals)
    else:
        _structure = htf_result.get("structure", "mixed") if htf_result else "mixed"
        _reversal_allowed = _structure != "LH_LL"
        if abs(price - support) < 1.5 * _atr_val and rsi <= 42 and _reversal_allowed:
            setup_type = "reversal"
            quality = 45
            quality += 15 if candle in ("hammer", "bullish_engulfing") else 0
            quality += min(15, int((42 - rsi) * 1.5))
            quality += 10 if vol_ratio >= 1.2 else 0
            quality -= 15 if trend == "bearish" else 0
            trigger_low  = support * 0.99
            trigger_high = support * 1.025
            log.append(f"REVERSAL: near support={support:.2f}  RSI={rsi:.1f}  quality={quality}  allowed={_reversal_allowed}")

    # Cap at 100
    quality = min(100, quality)
    valid = setup_type != "none" and quality >= 40

    result.update({
        "setup_type": setup_type,
        "valid": valid,
        "setup_quality": quality,
        "trigger_zone": {"low": round(trigger_low, 2), "high": round(trigger_high, 2)},
    })
    log.append(f"Setup={setup_type}  valid={valid}  quality={quality}")
    return result

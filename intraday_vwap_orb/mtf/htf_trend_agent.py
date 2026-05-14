"""
HTF Trend Agent
---------------
Determines macro trend direction from Daily / Weekly OHLCV data.

Indicators  : EMA20, EMA50, EMA200
Structure   : Higher-High / Higher-Low (bullish) or Lower-High / Lower-Low (bearish)
Sector proxy: optional Nifty Metal series passed alongside symbol data

Output (JSON-serialisable dict):
{
  "trend"        : "bullish" | "bearish" | "neutral",
  "strength"     : 0-100,
  "key_levels"   : {"support": float, "resistance": float},
  "trend_quality": "strong" | "weak" | "choppy",
  "ema20"        : float,
  "ema50"        : float,
  "ema200"       : float,
  "structure"    : "HH_HL" | "LH_LL" | "mixed",
  "sector_aligned": bool | None,
  "log"          : [str, ...]
}
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
import numpy as np


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _pivot_levels(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 60) -> dict:
    """
    Fix #7: Identify support/resistance from actual swing high/low pivots.
    A pivot high = bar where high > both neighbours (2-bar each side).
    A pivot low  = bar where low  < both neighbours.
    Returns the most recent significant pivot high and low.
    """
    n = min(lookback, len(close))
    hi_seg = high.iloc[-n:].values
    lo_seg = low.iloc[-n:].values
    cl_seg = close.iloc[-n:].values

    pivot_highs = []
    pivot_lows  = []
    for i in range(2, n - 2):
        if hi_seg[i] > hi_seg[i-1] and hi_seg[i] > hi_seg[i-2] and \
           hi_seg[i] > hi_seg[i+1] and hi_seg[i] > hi_seg[i+2]:
            pivot_highs.append(hi_seg[i])
        if lo_seg[i] < lo_seg[i-1] and lo_seg[i] < lo_seg[i-2] and \
           lo_seg[i] < lo_seg[i+1] and lo_seg[i] < lo_seg[i+2]:
            pivot_lows.append(lo_seg[i])

    price = float(cl_seg[-1])
    # Most recent pivot below price = support; above = resistance
    lows_below  = [v for v in pivot_lows  if v < price]
    highs_above = [v for v in pivot_highs if v > price]

    support    = float(max(lows_below))  if lows_below  else float(min(lo_seg))
    resistance = float(min(highs_above)) if highs_above else float(max(hi_seg))
    return {"support": round(support, 2), "resistance": round(resistance, 2)}


def _detect_structure(close: pd.Series, lookback: int = 20) -> str:
    """Return 'HH_HL', 'LH_LL', or 'mixed' from the last `lookback` closes."""
    if len(close) < lookback + 2:
        return "mixed"
    seg = close.iloc[-lookback:].values
    highs = [seg[i] for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]]
    lows  = [seg[i] for i in range(1, len(seg) - 1) if seg[i] < seg[i - 1] and seg[i] < seg[i + 1]]

    if len(highs) < 2 or len(lows) < 2:
        return "mixed"

    hh = highs[-1] > highs[-2]
    hl = lows[-1]  > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1]  < lows[-2]

    if hh and hl:
        return "HH_HL"
    if lh and ll:
        return "LH_LL"
    return "mixed"


def _trend_strength(
    price: float,
    ema20: float,
    ema50: float,
    ema200: float,
    atr: float,
) -> float:
    """Score 0-100. Distance from EMAs normalised by ATR."""
    if atr <= 0:
        return 50.0

    dist_20  = (price - ema20)  / atr
    dist_50  = (price - ema50)  / atr
    dist_200 = (price - ema200) / atr

    # EMA alignment bonus (each aligned pair adds points)
    align_score = 0.0
    if ema20 > ema50:
        align_score += 20
    if ema50 > ema200:
        align_score += 20
    if ema20 > ema200:
        align_score += 10

    # Distance contribution (capped at ±2 ATR per EMA)
    dist_contrib = (
        max(-2.0, min(2.0, dist_20))  * 10 +
        max(-2.0, min(2.0, dist_50))  * 10 +
        max(-2.0, min(2.0, dist_200)) * 5
    )

    raw = 50.0 + align_score + dist_contrib
    return float(max(0.0, min(100.0, raw)))


def run_htf_trend_agent(
    daily_df: pd.DataFrame,
    weekly_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Parameters
    ----------
    daily_df  : DataFrame with columns [open, high, low, close, volume] (daily bars)
    weekly_df : optional weekly bars (if None, resampled from daily)
    sector_df : optional Nifty Metal daily close series (column 'close')

    Returns
    -------
    dict — see module docstring
    """
    log: list[str] = []
    result: dict = {
        "trend": "neutral",
        "strength": 50,
        "key_levels": {"support": 0.0, "resistance": 0.0},
        "trend_quality": "choppy",
        "ema20": 0.0,
        "ema50": 0.0,
        "ema200": 0.0,
        "structure": "mixed",
        "sector_aligned": None,
        "log": log,
    }

    if daily_df is None or daily_df.empty:
        log.append("ERROR: daily_df empty — cannot run HTF agent")
        return result

    df = daily_df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        log.append("ERROR: 'close' column missing")
        return result

    close = df["close"].dropna()
    if len(close) < 30:
        log.append(f"WARN: only {len(close)} bars — need ≥30 for reliable EMAs")

    # ── EMAs ────────────────────────────────────────────────────────────────
    ema20  = _ema(close, 20).iloc[-1]
    ema50  = _ema(close, 50).iloc[-1] if len(close) >= 50  else ema20
    ema200 = _ema(close, 200).iloc[-1] if len(close) >= 200 else ema50
    price  = float(close.iloc[-1])

    result["ema20"]  = round(ema20,  2)
    result["ema50"]  = round(ema50,  2)
    result["ema200"] = round(ema200, 2)
    log.append(f"Price={price:.2f}  EMA20={ema20:.2f}  EMA50={ema50:.2f}  EMA200={ema200:.2f}")

    # ── ATR (14-bar) ─────────────────────────────────────────────────────────
    if "high" in df.columns and "low" in df.columns:
        hi = df["high"].dropna(); lo = df["low"].dropna()
        tr = pd.concat([
            hi - lo,
            (hi - close.shift(1)).abs(),
            (lo - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())
    else:
        atr = float(close.std()) or 1.0
    log.append(f"ATR(14)={atr:.2f}")

    # ── Trend direction ──────────────────────────────────────────────────────
    if price > ema50 and ema20 > ema50:
        trend = "bullish"
    elif price < ema50 and ema20 < ema50:
        trend = "bearish"
    else:
        trend = "neutral"
    log.append(f"Trend={trend}")

    # ── Market structure ─────────────────────────────────────────────────────
    structure = _detect_structure(close, lookback=min(30, len(close) - 2))
    result["structure"] = structure
    log.append(f"Structure={structure}")

    # ── Trend strength score ─────────────────────────────────────────────────
    strength = _trend_strength(price, ema20, ema50, ema200, atr)
    if trend == "bearish":
        strength = 100.0 - strength   # invert so 100 = very bearish
    log.append(f"Strength={strength:.1f}")

    # ── Trend quality ────────────────────────────────────────────────────────
    if strength >= 70 and structure in ("HH_HL", "LH_LL"):
        quality = "strong"
    elif strength >= 50:
        quality = "weak"
    else:
        quality = "choppy"

    # ── Key levels — Fix #7: pivot-based swing high/low ────────────────────
    if "high" in df.columns and "low" in df.columns:
        key_levels = _pivot_levels(df["high"], df["low"], close, lookback=60)
    else:
        recent = close.iloc[-60:] if len(close) >= 60 else close
        key_levels = {
            "support":    round(float(recent.rolling(10).min().dropna().iloc[-1]), 2),
            "resistance": round(float(recent.rolling(10).max().dropna().iloc[-1]), 2),
        }
    result["key_levels"] = key_levels
    log.append(f"Support={key_levels['support']:.2f}  Resistance={key_levels['resistance']:.2f}")

    # ── Sector alignment — Fix #12: gradient strength not binary ─────────────────
    sector_aligned: bool | None = None
    result["sector_strength"] = 0.0  # pct above/below EMA50 of sector
    if sector_df is not None and not sector_df.empty:
        sc = sector_df.copy()
        sc.columns = [c.lower() for c in sc.columns]
        if "close" in sc.columns:
            sc_close = sc["close"].dropna()
            if len(sc_close) >= 50:
                sc_ema50 = _ema(sc_close, 50).iloc[-1]
                sc_price = float(sc_close.iloc[-1])
                # Fix #12: strength = % deviation from EMA50 (-ve = weak, +ve = strong)
                sc_strength = (sc_price - float(sc_ema50)) / float(sc_ema50) * 100
                result["sector_strength"] = round(sc_strength, 2)
                sector_bull = sc_strength > 0
                sector_aligned = (trend == "bullish" and sector_bull) or (trend == "bearish" and not sector_bull)
                log.append(f"Sector price={sc_price:.2f} EMA50={sc_ema50:.2f} strength={sc_strength:.2f}% aligned={sector_aligned}")

    result.update({
        "trend": trend,
        "strength": round(strength, 1),
        "trend_quality": quality,
        "sector_aligned": sector_aligned,
    })
    return result

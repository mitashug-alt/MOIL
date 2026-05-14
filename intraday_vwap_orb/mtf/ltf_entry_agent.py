"""
LTF Entry Agent
---------------
Times entries using lower timeframe (1H or 15m) bars.

Logic:
  - Candlestick confirmation (engulfing, rejection wicks)
  - RSI bounce above 40 (long) or below 60 (short)
  - Micro-structure breakout (price > last N-bar high)

Rules:
  - Only align with HTF trend direction
  - No counter-trend entries

Output (JSON-serialisable dict):
{
  "entry_signal"          : "buy" | "sell" | "wait",
  "entry_price"           : float,
  "confirmation_strength" : 0-100,
  "invalid_if"            : float,   # price that invalidates setup
  "ltf_rsi"               : float,
  "ltf_candle"            : str,
  "micro_breakout"        : bool,
  "log"                   : [str, ...]
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


def _candle_pattern(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "none"
    o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]
    o0, h0, l0, c0 = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    body0 = abs(c0 - o0)
    range0 = (h0 - l0) or 1e-9
    lower_wick = min(o0, c0) - l0
    upper_wick = h0 - max(o0, c0)

    if c1 < o1 and c0 > o0 and c0 > o1 and o0 < c1:
        return "bullish_engulfing"
    if c1 > o1 and c0 < o0 and c0 < o1 and o0 > c1:
        return "bearish_engulfing"
    if lower_wick > 2 * body0 and upper_wick < 0.3 * range0 and c0 >= o0:  # Fix #4: must close bullish
        return "hammer"
    if upper_wick > 2 * body0 and lower_wick < 0.3 * range0 and c0 <= o0:  # Fix #4: must close bearish
        return "shooting_star"
    if body0 < 0.1 * range0:
        return "doji"
    return "none"


def _micro_breakout(df: pd.DataFrame, side: str, lookback: int = 5) -> bool:
    """True if last bar breaks above (LONG) or below (SHORT) the prior N-bar extreme."""
    if len(df) < lookback + 2:
        return False
    prior = df.iloc[-(lookback + 1):-1]
    current = df.iloc[-1]
    if side == "buy":
        return float(current["close"]) > float(prior["high"].max())
    return float(current["close"]) < float(prior["low"].min())


def run_ltf_entry_agent(
    ltf_df: pd.DataFrame,
    htf_result: Optional[dict] = None,
    setup_result: Optional[dict] = None,
) -> dict:
    """
    Parameters
    ----------
    ltf_df       : 1H or 15m OHLCV DataFrame
    htf_result   : output from run_htf_trend_agent
    setup_result : output from run_setup_detection_agent

    Returns
    -------
    dict — see module docstring
    """
    log: list[str] = []
    result: dict = {
        "entry_signal": "wait",
        "entry_price": 0.0,
        "confirmation_strength": 0,
        "invalid_if": 0.0,
        "ltf_rsi": 50.0,
        "ltf_candle": "none",
        "micro_breakout": False,
        "log": log,
    }

    if ltf_df is None or ltf_df.empty:
        log.append("ERROR: ltf_df empty")
        return result

    df = ltf_df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        log.append("ERROR: 'close' column missing")
        return result

    close = df["close"].dropna()
    price = float(close.iloc[-1])

    trend   = htf_result.get("trend", "neutral") if htf_result else "neutral"
    setup_t = setup_result.get("setup_type", "none") if setup_result else "none"
    setup_v = setup_result.get("valid", False) if setup_result else False

    log.append(f"HTF trend={trend}  setup={setup_t}  valid={setup_v}")

    # No entry if no valid setup or HTF is neutral
    if trend == "neutral" or not setup_v:
        log.append("WAIT: neutral trend or no valid setup")
        result["entry_price"] = round(price, 2)
        return result

    # ── LTF indicators ───────────────────────────────────────────────────────
    ltf_rsi = _rsi(close, 14)
    candle  = _candle_pattern(df)
    desired_side = "buy" if trend == "bullish" else "sell"
    micro_bo = _micro_breakout(df, desired_side)

    result["ltf_rsi"]       = round(ltf_rsi, 2)
    result["ltf_candle"]    = candle
    result["micro_breakout"] = micro_bo
    log.append(f"LTF RSI={ltf_rsi:.1f}  candle={candle}  micro_breakout={micro_bo}")

    # ── Scoring ──────────────────────────────────────────────────────────────
    score = 0

    if desired_side == "buy":
        # Fix #4b: RSI >= 45 required (was 40 — too permissive, fires in downtrend)
        if ltf_rsi >= 45:
            score += 20
        if ltf_rsi >= 55:
            score += 15
        if candle in ("bullish_engulfing", "hammer"):
            score += 30
        if micro_bo:
            score += 25
        if setup_t == "pullback" and ltf_rsi < 55:
            score += 10
        invalid_if = float(df["low"].iloc[-5:].min()) if len(df) >= 5 else price * 0.98
    else:
        if ltf_rsi <= 60:
            score += 25
        if ltf_rsi <= 50:
            score += 10
        if candle in ("bearish_engulfing", "shooting_star"):
            score += 30
        if micro_bo:
            score += 25
        if setup_t == "pullback" and ltf_rsi > 45:
            score += 10
        invalid_if = float(df["high"].iloc[-5:].max()) if len(df) >= 5 else price * 1.02

    score = min(100, score)
    log.append(f"Confirmation score={score}")

    # ── Decision ─────────────────────────────────────────────────────────────
    if score >= 50:
        entry_signal = desired_side
        log.append(f"SIGNAL={entry_signal}")
    else:
        entry_signal = "wait"
        log.append(f"WAIT: score={score} < 50")

    result.update({
        "entry_signal": entry_signal,
        "entry_price": round(price, 2),
        "confirmation_strength": score,
        "invalid_if": round(invalid_if, 2),
    })
    return result

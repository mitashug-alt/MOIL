"""
Risk + Leverage Agent
---------------------
Optimises position sizing and MTF (Margin Trading Facility) leverage.

Inputs  : setup quality, trend strength, ATR volatility, regime context
Rules   :
  weak setup  (composite < 50)  → 1x, reduced size
  medium      (50–69)           → 1.5–2x
  strong      (≥ 70)            → 3–4x, capped by regime

MOIL / commodity-stock regime caps:
  - If sector weak (sector_aligned=False) → max 2x
  - If regime_shift flag → confidence weight boosted in composite

Risk constraints:
  - max_risk_pct : 1–2% of capital per trade
  - mandatory SL : entry ± ATR * sl_atr_mult
  - targets      : T1 = Entry ± ATR * 1.5 * RR, T2 = Entry ± ATR * 2.5 * RR

Output (JSON-serialisable dict):
{
  "capital_allocation_pct": float,
  "leverage"              : float,   # 1.0 – 4.0
  "risk_score"            : 0-100,
  "stop_loss"             : float,
  "targets"               : [t1, t2],
  "position_size_shares"  : int,     # given capital_inr
  "risk_per_trade_inr"    : float,
  "regime_capped"         : bool,
  "log"                   : [str, ...]
}
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
import numpy as np


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if not {"high", "low", "close"}.issubset(df.columns):
        return float(df["close"].std()) if "close" in df.columns else 1.0
    hi = df["high"]; lo = df["low"]; cl = df["close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)
    series = tr.rolling(period).mean()
    return float(series.iloc[-1]) if not series.empty and not math.isnan(series.iloc[-1]) else float(tr.mean())


def run_risk_leverage_agent(
    daily_df: pd.DataFrame,
    htf_result: Optional[dict] = None,
    setup_result: Optional[dict] = None,
    ltf_result: Optional[dict] = None,
    capital_inr: float = 100_000.0,
    max_risk_pct: float = 1.5,
    sl_atr_mult: float = 1.5,
    rr_t1: float = 1.5,
    rr_t2: float = 2.5,
) -> dict:
    """
    Parameters
    ----------
    daily_df     : daily OHLCV DataFrame (for ATR)
    htf_result   : output from run_htf_trend_agent
    setup_result : output from run_setup_detection_agent
    ltf_result   : output from run_ltf_entry_agent
    capital_inr  : total capital in ₹ (default ₹1,00,000)
    max_risk_pct : max % of capital to risk per trade
    sl_atr_mult  : stop = entry ± sl_atr_mult * ATR
    rr_t1 / rr_t2: reward multiples (of risk) for T1 and T2

    Returns
    -------
    dict — see module docstring
    """
    log: list[str] = []
    result: dict = {
        "capital_allocation_pct": 0.0,
        "leverage": 1.0,
        "risk_score": 0,
        "stop_loss": 0.0,
        "targets": [0.0, 0.0],
        "position_size_shares": 0,
        "risk_per_trade_inr": 0.0,
        "regime_capped": False,
        "log": log,
    }

    if daily_df is None or daily_df.empty:
        log.append("ERROR: daily_df empty")
        return result

    trend_strength  = htf_result.get("strength", 50.0)          if htf_result  else 50.0
    setup_quality   = setup_result.get("setup_quality", 0)       if setup_result else 0
    conf_strength   = ltf_result.get("confirmation_strength", 0) if ltf_result  else 0
    entry_signal    = ltf_result.get("entry_signal", "wait")     if ltf_result  else "wait"
    entry_price     = ltf_result.get("entry_price", 0.0)         if ltf_result  else 0.0
    sector_aligned  = htf_result.get("sector_aligned", None)     if htf_result  else None
    trend_dir       = htf_result.get("trend", "neutral")         if htf_result  else "neutral"
    # Fix #3: use actual volume_ratio as 4th factor (not setup_quality proxy)
    raw_vol_ratio   = setup_result.get("volume_ratio", 1.0)      if setup_result else 1.0
    volume_factor   = min(100.0, max(0.0, (raw_vol_ratio - 0.5) / 2.0 * 100))

    atr = _atr(daily_df)
    if entry_price == 0.0:
        df = daily_df.copy(); df.columns = [c.lower() for c in df.columns]
        entry_price = float(df["close"].iloc[-1])

    log.append(f"ATR={atr:.2f}  entry={entry_price:.2f}  signal={entry_signal}")
    log.append(f"TrendStrength={trend_strength:.1f}  SetupQuality={setup_quality}  ConfStrength={conf_strength}")

    # ── Composite risk score (Fix #3: volume_factor replaces setup_quality proxy) ──
    risk_score = (
        0.35 * trend_strength +
        0.30 * setup_quality  +
        0.20 * conf_strength  +
        0.15 * volume_factor
    )
    risk_score = round(min(100.0, risk_score), 1)
    log.append(f"risk_score={risk_score}")

    # ── Leverage tiers ───────────────────────────────────────────────────────
    if risk_score < 50:
        leverage = 1.0
        alloc_pct = max(5.0, risk_score * 0.3)
    elif risk_score < 70:
        leverage = 1.5 + (risk_score - 50) / 20 * 0.5   # 1.5 → 2.0
        alloc_pct = 15.0 + (risk_score - 50) * 0.5
    else:
        leverage = 3.0 + (risk_score - 70) / 30 * 1.0   # 3.0 → 4.0
        alloc_pct = 25.0 + (risk_score - 70) * 0.5

    # ── Regime cap (MOIL-type stocks) ────────────────────────────────────────
    regime_capped = False
    if sector_aligned is False:
        leverage = min(leverage, 2.0)
        alloc_pct = min(alloc_pct, 20.0)
        regime_capped = True
        log.append("REGIME CAP: sector misaligned → max 2x leverage")

    leverage  = round(min(4.0, leverage), 2)
    alloc_pct = round(min(40.0, alloc_pct), 1)
    log.append(f"leverage={leverage}x  alloc_pct={alloc_pct}%")

    # ── Stop loss and targets (Fix #3b: direction driven by entry_signal only) ────
    # Fallback: when LTF has no signal yet, use trend direction
    _is_long = (entry_signal == "buy") or (entry_signal == "wait" and trend_dir == "bullish")
    if _is_long:
        stop_loss = entry_price - sl_atr_mult * atr
        t1 = entry_price + rr_t1 * sl_atr_mult * atr
        t2 = entry_price + rr_t2 * sl_atr_mult * atr
    else:
        stop_loss = entry_price + sl_atr_mult * atr
        t1 = entry_price - rr_t1 * sl_atr_mult * atr
        t2 = entry_price - rr_t2 * sl_atr_mult * atr

    # Fix #3c: guard against degenerate stop (stop must be positive and away from entry)
    min_tick = max(0.05, atr * 0.1)
    if abs(entry_price - stop_loss) < min_tick:
        stop_loss = entry_price - min_tick if _is_long else entry_price + min_tick

    risk_per_share = abs(entry_price - stop_loss)
    log.append(f"stop={stop_loss:.2f}  T1={t1:.2f}  T2={t2:.2f}  risk/share=₹{risk_per_share:.2f}")

    # ── Position sizing ──────────────────────────────────────────────────────
    capital_deployed = capital_inr * (alloc_pct / 100)
    max_risk_inr     = capital_inr * (max_risk_pct / 100)
    if risk_per_share > 0:
        size_by_risk = math.floor(max_risk_inr / risk_per_share)
        size_by_capital = math.floor((capital_deployed * leverage) / entry_price)
        pos_size = min(size_by_risk, size_by_capital)
    else:
        pos_size = 0

    risk_per_trade_inr = pos_size * risk_per_share
    log.append(f"pos_size={pos_size} shares  risk_inr=₹{risk_per_trade_inr:.0f}")

    result.update({
        "capital_allocation_pct": alloc_pct,
        "leverage": leverage,
        "risk_score": risk_score,
        "stop_loss": round(stop_loss, 2),
        "targets": [round(t1, 2), round(t2, 2)],
        "position_size_shares": pos_size,
        "risk_per_trade_inr": round(risk_per_trade_inr, 2),
        "regime_capped": regime_capped,
    })
    return result

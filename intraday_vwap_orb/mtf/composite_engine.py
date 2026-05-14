"""
Composite MTF Engine
--------------------
Combines all four agent outputs into a single actionable signal.

Scoring formula (per spec):
  confidence_score =
      (0.35 * trend_strength)       +
      (0.30 * setup_quality)        +
      (0.20 * confirmation_strength)+
      (0.15 * volume_factor)

  volume_factor is derived from setup_result["volume_ratio"] mapped to 0-100.

Decision rules:
  score > 75  → strong trade
  60–75       → moderate trade
  < 60        → avoid

MOIL regime override:
  - sector misaligned  → max confidence capped at 72
  - regime_capped      → add note

Final output (JSON-serialisable):
{
  "symbol"          : str,
  "bias"            : "bullish" | "bearish" | "neutral",
  "confidence_score": float,
  "decision"        : "strong_trade" | "moderate_trade" | "avoid",
  "action"          : "buy" | "sell" | "wait",
  "entry"           : float,
  "stop_loss"       : float,
  "targets"         : [t1, t2],
  "leverage"        : float,
  "position_size"   : int,
  "risk_per_trade"  : float,
  "notes"           : str,
  "agent_outputs"   : { htf, setup, ltf, risk },
  "log"             : [str, ...]
}
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from .htf_trend_agent      import run_htf_trend_agent
from .setup_detection_agent import run_setup_detection_agent
from .ltf_entry_agent      import run_ltf_entry_agent
from .risk_leverage_agent  import run_risk_leverage_agent
from .adaptive_layer       import AdaptiveWeights


def run_composite_engine(
    daily_df: pd.DataFrame,
    ltf_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    symbol: str = "MOIL",
    capital_inr: float = 100_000.0,
    max_risk_pct: float = 1.5,
    adaptive_weights: Optional[AdaptiveWeights] = None,
) -> dict:
    """
    Run all four agents and combine into a final trading decision.

    Parameters
    ----------
    daily_df         : Daily OHLCV (primary timeframe for HTF + setup)
    ltf_df           : 1H / 15m OHLCV for LTF entry timing (optional)
    sector_df        : Nifty Metal daily for regime check (optional)
    symbol           : ticker label
    capital_inr      : total capital in ₹
    max_risk_pct     : max % capital at risk per trade
    adaptive_weights : AdaptiveWeights instance (optional; uses defaults if None)

    Returns
    -------
    dict — see module docstring
    """
    log: list[str] = []
    aw = adaptive_weights or AdaptiveWeights()
    weights = aw.get_weights()
    log.append(f"Adaptive weights: {weights}")

    # ── Run agents ───────────────────────────────────────────────────────────
    htf   = run_htf_trend_agent(daily_df, sector_df=sector_df)
    setup = run_setup_detection_agent(daily_df, htf_result=htf)
    ltf   = run_ltf_entry_agent(ltf_df if ltf_df is not None else daily_df, htf_result=htf, setup_result=setup)
    risk  = run_risk_leverage_agent(
        daily_df,
        htf_result=htf,
        setup_result=setup,
        ltf_result=ltf,
        capital_inr=capital_inr,
        max_risk_pct=max_risk_pct,
    )

    # ── Volume factor ────────────────────────────────────────────────────────
    vol_ratio     = setup.get("volume_ratio", 1.0)
    volume_factor = min(100.0, max(0.0, (vol_ratio - 0.5) / 2.0 * 100))

    # ── Composite score with adaptive weights ────────────────────────────────
    trend_strength   = htf.get("strength", 50.0)
    setup_quality    = setup.get("setup_quality", 0.0)
    conf_strength    = ltf.get("confirmation_strength", 0.0)

    raw_score = (
        weights["trend_weight"]  * trend_strength  +
        weights["setup_weight"]  * setup_quality   +
        weights["conf_weight"]   * conf_strength   +
        weights["volume_weight"] * volume_factor
    )
    confidence_score = round(min(100.0, max(0.0, raw_score)), 1)

    # Fix #12: Gradient regime cap based on sector_strength (% vs EMA50)
    # Weak sector  (-1% to -3%)  → cap 72 | Very weak (<-3%) → cap 65 | Misaligned → cap 72
    regime_note = ""
    sc_strength = htf.get("sector_strength", 0.0)
    if htf.get("sector_aligned") is False:
        if sc_strength < -3.0:
            cap = 65.0
            regime_note = f" [capped at 65: sector very weak {sc_strength:.1f}%]"
        else:
            cap = 72.0
            regime_note = f" [capped at 72: sector misaligned {sc_strength:.1f}%]"
        confidence_score = min(confidence_score, cap)
        log.append(f"REGIME CAP: sector_strength={sc_strength:.1f}% → cap={cap}")

    log.append(f"Raw composite={raw_score:.1f}  final={confidence_score}")

    # ── Decision ─────────────────────────────────────────────────────────────
    if confidence_score > 75:
        decision = "strong_trade"
    elif confidence_score >= 60:
        decision = "moderate_trade"
    else:
        decision = "avoid"

    raw_action = ltf.get("entry_signal", "wait")
    # Fix #2: never emit a directional action when decision is avoid
    action = raw_action if decision != "avoid" else "wait"
    bias   = htf.get("trend", "neutral")
    log.append(f"decision={decision}  action={action}  bias={bias}")

    # ── Notes ────────────────────────────────────────────────────────────────
    setup_t = setup.get("setup_type", "none")
    quality_lbl = htf.get("trend_quality", "")
    notes = (
        f"{symbol} | {bias.title()} {quality_lbl} trend | "
        f"Setup: {setup_t} (quality {setup_quality:.0f}/100) | "
        f"LTF: {ltf.get('ltf_candle','?')} candle, RSI {ltf.get('ltf_rsi',0):.1f} | "
        f"Leverage: {risk.get('leverage',1.0):.1f}x{regime_note}"
    )

    entry     = ltf.get("entry_price", 0.0)
    stop_loss = risk.get("stop_loss", 0.0)
    targets   = risk.get("targets", [0.0, 0.0])
    leverage  = risk.get("leverage", 1.0)

    return {
        "symbol"          : symbol,
        "bias"            : bias,
        "confidence_score": confidence_score,
        "decision"        : decision,
        "action"          : action,
        "entry"           : entry,
        "stop_loss"       : stop_loss,
        "targets"         : targets,
        "leverage"        : leverage,
        "position_size"   : risk.get("position_size_shares", 0),
        "risk_per_trade"  : risk.get("risk_per_trade_inr", 0.0),
        "notes"           : notes,
        "agent_outputs"   : {
            "htf"  : htf,
            "setup": setup,
            "ltf"  : ltf,
            "risk" : risk,
        },
        "log": log,
    }

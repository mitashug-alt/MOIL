"""
MTF Backtest Engine
-------------------
Walks forward through a daily price history, running the composite engine
on each bar's preceding window, collecting signals and resolving outcomes.

Usage:
    results = run_mtf_backtest(daily_df, sector_df=None, capital_inr=100_000)

Returns a dict:
{
  "trades"       : list[MTFPaperTrade],
  "trades_df"    : pd.DataFrame,
  "equity_curve" : pd.Series  (cumulative PnL points),
  "metrics"      : dict,
  "signal_log"   : list[dict],
}
"""
from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

from .composite_engine import run_composite_engine
from .adaptive_layer import AdaptiveWeights, update_weights_from_trade_log
from .paper_trading import (
    MTFPaperTrade,
    resolve_mtf_paper_trade,
    mtf_paper_log_to_df,
    summarise_mtf_track,
)


def run_mtf_backtest(
    daily_df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None,
    capital_inr: float = 100_000.0,
    max_risk_pct: float = 1.5,
    lookback: int = 120,       # min bars of history before first signal
    forward_bars: int = 5,     # bars to simulate outcome
    min_confidence: float = 60.0,
    adaptive: bool = True,     # dynamically update weights per trade
    # Fix #11: transaction cost parameters
    brokerage_pct: float = 0.03,    # % per side (0.03% = typical discount broker)
    slippage_pct: float = 0.05,     # % per side (half-spread estimate)
    mtf_rate_pa: float = 18.0,      # annual MTF interest rate (%)
) -> dict:
    """
    Walk-forward backtest over daily_df.

    Parameters
    ----------
    daily_df      : full daily OHLCV (columns: open/high/low/close/volume)
    sector_df     : Nifty Metal daily for regime check (optional)
    capital_inr   : capital for position sizing
    max_risk_pct  : max % capital at risk per trade
    lookback      : bars of history before first signal
    forward_bars  : bars ahead to resolve outcome
    min_confidence: skip signals below this score
    adaptive      : if True, update weights as trades resolve

    Returns
    -------
    dict — see module docstring
    """
    df = daily_df.copy()
    df.columns = [c.lower() for c in df.columns]

    if len(df) < lookback + forward_bars + 1:
        return {
            "trades": [], "trades_df": pd.DataFrame(),
            "equity_curve": pd.Series(dtype=float),
            "metrics": {}, "signal_log": [],
            "error": f"Need ≥ {lookback + forward_bars + 1} bars, got {len(df)}",
        }

    trades: list[MTFPaperTrade] = []
    signal_log: list[dict]      = []
    aw = AdaptiveWeights()
    cumulative_pnl: list[float] = [0.0]
    bar_dates: list = []
    open_until_bar: int = -1  # Fix #6: track when current position closes

    for i in range(lookback, len(df) - forward_bars):
        window = df.iloc[:i + 1].copy()
        date_label = window.index[i] if hasattr(window.index, '__iter__') else i

        # Sector window aligned to same index length if provided
        sec_window = None
        if sector_df is not None:
            sec_df = sector_df.copy()
            sec_df.columns = [c.lower() for c in sec_df.columns]
            sec_window = sec_df.iloc[:i + 1].copy()

        result = run_composite_engine(
            daily_df=window,
            ltf_df=None,
            sector_df=sec_window,
            symbol="MOIL",
            capital_inr=capital_inr,
            max_risk_pct=max_risk_pct,
            adaptive_weights=aw,
        )

        score  = result["confidence_score"]
        action = result["action"]

        signal_log.append({
            "bar"        : i,
            "date"       : str(date_label),
            "score"      : score,
            "action"     : action,
            "decision"   : result["decision"],
            "bias"       : result["bias"],
            "setup_type" : result["agent_outputs"]["setup"].get("setup_type", "none"),
        })

        # Skip low-confidence or wait signals
        if score < min_confidence or action == "wait":
            continue
        # Fix #6: skip if there is an open position that hasn't resolved yet
        if i <= open_until_bar:
            continue

        # Build paper trade
        lev = result["leverage"]
        tt  = "intraday" if lev <= 1.0 else "margin"
        trade = MTFPaperTrade(
            logged_at         = str(date_label),
            trade_type        = tt,
            action            = action,
            bias              = result["bias"],
            confidence_score  = score,
            decision          = result["decision"],
            leverage          = lev,
            entry             = result["entry"],
            stop              = result["stop_loss"],
            target1           = result["targets"][0],
            target2           = result["targets"][1],
            position_size     = result["position_size"],
            risk_per_trade    = result["risk_per_trade"],
            setup_type        = result["agent_outputs"]["setup"].get("setup_type", "none"),
            trend_score       = result["agent_outputs"]["htf"].get("strength", 50),
            setup_score       = result["agent_outputs"]["setup"].get("setup_quality", 50),
            conf_score        = result["agent_outputs"]["ltf"].get("confirmation_strength", 50),
            volume_ratio      = result["agent_outputs"]["setup"].get("volume_ratio", 1.0),
            track             = "auto",
            notes             = result["notes"],
        )

        future = df.iloc[i + 1: i + 1 + forward_bars].copy()
        trade  = resolve_mtf_paper_trade(trade, future, margin_bars=forward_bars)
        # Fix #6: mark position as open until forward_bars resolved
        open_until_bar = i + forward_bars
        trades.append(trade)

        raw_pnl = trade.pnl_points or 0.0

        # Fix #11: deduct transaction costs
        entry_px = trade.entry or 1.0
        cost_pct  = (brokerage_pct + slippage_pct) / 100.0  # per side
        tx_cost   = entry_px * cost_pct * 2  # round-trip in price points
        # MTF interest: daily rate × forward_bars × (leverage - 1) × entry_price
        lev_above_1 = max(0.0, trade.leverage - 1.0)
        daily_rate  = mtf_rate_pa / 365 / 100
        interest_cost = entry_px * lev_above_1 * daily_rate * forward_bars
        net_pnl = raw_pnl - tx_cost - interest_cost
        # Store net PnL back so metrics reflect real costs
        if trade.status == "closed":
            trade.pnl_points = round(net_pnl, 2)
            risk = abs(trade.entry - trade.stop) if abs(trade.entry - trade.stop) > 0 else 1.0
            trade.r_multiple = round(net_pnl / risk, 2)

        pnl = trade.pnl_points or 0.0
        cumulative_pnl.append(cumulative_pnl[-1] + pnl)
        bar_dates.append(str(date_label))

        # Adaptive weight update
        if adaptive and trade.status == "closed":
            outcome = "win" if pnl > 0 else "loss"
            aw = aw.update(outcome, {
                "trend" : trade.trend_score / 100,
                "setup" : trade.setup_score / 100,
                "conf"  : trade.conf_score  / 100,
                "volume": min(1.0, trade.volume_ratio / 2.0),
            })

    trades_df = mtf_paper_log_to_df(trades)
    equity    = pd.Series(cumulative_pnl[1:], index=bar_dates, name="cumulative_pnl")

    # ── Metrics ──────────────────────────────────────────────────────────────
    closed = trades_df[trades_df["status"] == "closed"] if not trades_df.empty else pd.DataFrame()
    pnl_s  = pd.to_numeric(closed["pnl_points"], errors="coerce") if not closed.empty else pd.Series(dtype=float)
    r_s    = pd.to_numeric(closed["r_multiple"], errors="coerce").dropna() if not closed.empty else pd.Series(dtype=float)
    conf_s = pd.to_numeric(closed["confidence_score"], errors="coerce").dropna() if not closed.empty else pd.Series(dtype=float)

    wins   = int((pnl_s > 0).sum())
    losses = int((pnl_s <= 0).sum())
    total  = wins + losses

    # Max drawdown from equity curve
    eq_arr  = np.array(cumulative_pnl)
    peak    = np.maximum.accumulate(eq_arr)
    dd      = eq_arr - peak
    max_dd  = float(dd.min()) if len(dd) > 0 else 0.0

    # Profit factor
    gross_profit = float(pnl_s[pnl_s > 0].sum()) if not pnl_s.empty else 0.0
    gross_loss   = abs(float(pnl_s[pnl_s < 0].sum())) if not pnl_s.empty else 1.0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    metrics = {
        "total_signals"  : len(signal_log),
        "trades_taken"   : len(trades),
        "trades_closed"  : total,
        "wins"           : wins,
        "losses"         : losses,
        "win_rate"       : round(wins / total, 3) if total > 0 else 0.0,
        "total_pnl_pts"  : round(float(pnl_s.sum()), 2) if not pnl_s.empty else 0.0,
        "avg_r"          : round(float(r_s.mean()), 2)   if not r_s.empty  else 0.0,
        "max_r"          : round(float(r_s.max()), 2)    if not r_s.empty  else 0.0,
        "max_dd_pts"     : round(max_dd, 2),
        "profit_factor"  : profit_factor,
        "avg_confidence" : round(float(conf_s.mean()), 1) if not conf_s.empty else 0.0,
        "final_weights"  : aw.get_weights(),
    }

    return {
        "trades"      : trades,
        "trades_df"   : trades_df,
        "equity_curve": equity,
        "metrics"     : metrics,
        "signal_log"  : signal_log,
    }


def build_validation_table(
    trades_df: pd.DataFrame,
    confidence_bins: list[tuple] = None,
) -> pd.DataFrame:
    """
    Group closed trades by confidence bin and compute forward stats.

    confidence_bins: list of (label, low, high) e.g. [("60-70",60,70), ...]
    Default: [60-70, 70-80, 80-100]
    """
    if confidence_bins is None:
        confidence_bins = [("60–70", 60, 70), ("70–80", 70, 80), ("80–100", 80, 100)]

    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    closed = trades_df[trades_df["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame()

    closed["confidence_score"] = pd.to_numeric(closed["confidence_score"], errors="coerce")
    closed["pnl_points"]       = pd.to_numeric(closed["pnl_points"],       errors="coerce")
    closed["r_multiple"]       = pd.to_numeric(closed["r_multiple"],        errors="coerce")

    rows = []
    for label, lo, hi in confidence_bins:
        seg = closed[(closed["confidence_score"] >= lo) & (closed["confidence_score"] < hi)]
        if seg.empty:
            rows.append({"Conf band": label, "Trades": 0, "Win rate": "—",
                         "Avg R": "—", "Total PnL (pts)": "—", "Avg conf": "—"})
            continue
        pnl = seg["pnl_points"].dropna()
        r   = seg["r_multiple"].dropna()
        wins = int((pnl > 0).sum())
        rows.append({
            "Conf band"      : label,
            "Trades"         : len(seg),
            "Win rate"       : f"{wins/len(seg)*100:.1f}%",
            "Avg R"          : f"{r.mean():.2f}R" if not r.empty else "—",
            "Total PnL (pts)": f"{pnl.sum():+.2f}",
            "Avg conf"       : f"{seg['confidence_score'].mean():.1f}",
        })
    return pd.DataFrame(rows)

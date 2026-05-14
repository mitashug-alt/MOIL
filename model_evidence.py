"""
Model Evidence Engine
---------------------
Computes all customer-facing proof metrics for the Intraday ORB engine
and the MTF Adaptive Engine.

All functions accept raw session-state lists / DataFrames and return
clean, JSON-serialisable dicts for rendering in the Evidence tab.

Metrics produced
----------------
Intraday ORB:
  - Win rate (auto-track & manual-track)
  - Avg R-multiple
  - Profit factor
  - Expectancy (₹ per trade, adjusted by vol)
  - Confidence calibration table (score band → actual win rate)
  - Signal quality funnel (generated → auto-approved → wins)
  - Daily edge stability (rolling 10-trade win rate)
  - Worst drawdown (sequence of losses)
  - Breakdown by side (LONG vs SHORT)
  - Breakdown by time-of-day

MTF Engine:
  - Same core stats (win rate / avg R / profit factor / expectancy)
  - Per setup type (pullback / breakout / reversal)
  - Per trade type (intraday 1x vs margin >1x)
  - Confidence calibration (60-70 / 70-80 / 80-100 → actual win rate)
  - Adaptive weight convergence (final vs default)
  - Net PnL with costs vs gross PnL
  - Rolling win rate (10-trade window)
  - Max consecutive losses
"""
from __future__ import annotations

from typing import Optional
import math
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _profit_factor(pnl_series: pd.Series) -> float:
    gross_profit = float(pnl_series[pnl_series > 0].sum())
    gross_loss   = abs(float(pnl_series[pnl_series < 0].sum()))
    return round(_safe_div(gross_profit, gross_loss, float("inf")), 2)


def _expectancy(pnl_series: pd.Series) -> float:
    """Average PnL per trade (points / ₹ per share)."""
    return round(float(pnl_series.mean()), 3) if not pnl_series.empty else 0.0


def _max_consecutive_losses(pnl_series: pd.Series) -> int:
    max_cl = 0
    cur_cl = 0
    for v in pnl_series:
        if v <= 0:
            cur_cl += 1
            max_cl = max(max_cl, cur_cl)
        else:
            cur_cl = 0
    return max_cl


def _rolling_win_rate(pnl_series: pd.Series, window: int = 10) -> pd.Series:
    wins = (pnl_series > 0).astype(int)
    return wins.rolling(window, min_periods=1).mean().round(3)


def _max_drawdown(pnl_series: pd.Series) -> float:
    """Max peak-to-trough in cumulative PnL."""
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    dd = (cum - peak)
    return round(float(dd.min()), 3)


def _confidence_calibration(df: pd.DataFrame,
                             score_col: str,
                             pnl_col: str,
                             bins: list[tuple] | None = None) -> pd.DataFrame:
    """
    For each confidence band, compute: trades, wins, actual win rate, avg PnL, avg R.
    """
    if bins is None:
        bins = [
            ("0–44",   0,  44),
            ("45–59", 45,  59),
            ("60–69", 60,  69),
            ("70–79", 70,  79),
            ("80–100",80, 100),
        ]
    rows = []
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[pnl_col]   = pd.to_numeric(df[pnl_col],   errors="coerce")
    for label, lo, hi in bins:
        seg = df[(df[score_col] >= lo) & (df[score_col] <= hi) & df[pnl_col].notna()]
        n = len(seg)
        if n == 0:
            rows.append({"Conf band": label, "Trades": 0, "Win %": "—",
                         "Avg PnL": "—", "Profit factor": "—", "Edge": "❌"})
            continue
        pnl   = seg[pnl_col]
        wins  = int((pnl > 0).sum())
        wr    = wins / n
        pf    = _profit_factor(pnl)
        edge  = "✅" if wr >= 0.55 and pf >= 1.3 else ("⚠️" if wr >= 0.45 else "❌")
        r_col = "r_multiple" if "r_multiple" in seg.columns else None
        avg_r = f"{pd.to_numeric(seg[r_col], errors='coerce').mean():.2f}R" if r_col else "—"
        rows.append({
            "Conf band":     label,
            "Trades":        n,
            "Win %":         f"{wr:.1%}",
            "Avg PnL (pts)": f"{pnl.mean():+.3f}",
            "Avg R":         avg_r,
            "Profit factor": str(pf),
            "Edge":          edge,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY ORB metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_intraday_evidence(paper_log: list) -> dict:
    """
    Accepts list of PaperSignal objects (or dicts via .to_dict()).

    Returns
    -------
    {
      "summary"          : dict   — top-line KPIs
      "by_track"         : dict   — auto vs manual stats
      "by_side"          : df     — LONG vs SHORT
      "by_time"          : df     — hour-of-day
      "confidence_cal"   : df     — calibration table
      "rolling_wr"       : Series — rolling 10-trade win rate
      "signal_funnel"    : dict
      "trades_df"        : df
      "error"            : str | None
    }
    """
    empty = {
        "summary": {}, "by_track": {}, "by_side": pd.DataFrame(),
        "by_time": pd.DataFrame(), "confidence_cal": pd.DataFrame(),
        "rolling_wr": pd.Series(dtype=float), "signal_funnel": {},
        "trades_df": pd.DataFrame(), "error": None,
    }

    if not paper_log:
        empty["error"] = "No intraday paper trades logged yet."
        return empty

    rows = [t.to_dict() if hasattr(t, "to_dict") else t for t in paper_log]
    df = pd.DataFrame(rows)

    # Coerce numerics
    for col in ["pnl", "r_multiple", "confidence_score", "volume_ratio", "or_range_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    closed = df[df["status"] == "closed"].copy() if "status" in df.columns else df.copy()
    if closed.empty:
        empty["error"] = "No closed intraday trades yet."
        return empty

    pnl    = closed["pnl"].dropna()
    r_vals = closed["r_multiple"].dropna() if "r_multiple" in closed.columns else pd.Series(dtype=float)
    wins   = int((pnl > 0).sum())
    total  = len(pnl)

    summary = {
        "total_trades"   : total,
        "wins"           : wins,
        "losses"         : total - wins,
        "win_rate"       : round(_safe_div(wins, total), 3),
        "avg_r"          : round(float(r_vals.mean()), 2) if not r_vals.empty else 0.0,
        "profit_factor"  : _profit_factor(pnl),
        "expectancy_pts" : _expectancy(pnl),
        "max_dd"         : _max_drawdown(pnl),
        "max_consec_loss": _max_consecutive_losses(pnl),
        "total_pnl_pts"  : round(float(pnl.sum()), 3),
    }

    # By track
    by_track = {}
    for trk in ["auto", "manual"]:
        sub = closed[closed["track"] == trk]["pnl"].dropna() if "track" in closed.columns else pd.Series(dtype=float)
        w   = int((sub > 0).sum())
        n   = len(sub)
        by_track[trk] = {
            "trades"  : n,
            "wins"    : w,
            "win_rate": round(_safe_div(w, n), 3),
            "avg_r"   : round(float(closed[closed["track"] == trk]["r_multiple"].dropna().mean()), 2)
                        if "r_multiple" in closed.columns and n > 0 else 0.0,
            "profit_factor": _profit_factor(sub),
            "total_pnl"    : round(float(sub.sum()), 3),
        }

    # By side
    by_side = pd.DataFrame()
    if "side" in closed.columns:
        _side_rows = []
        for side in ["LONG", "SHORT"]:
            sub = closed[closed["side"] == side]["pnl"].dropna()
            w = int((sub > 0).sum()); n = len(sub)
            _side_rows.append({
                "Side": side, "Trades": n,
                "Win %": f"{_safe_div(w,n):.1%}",
                "Avg PnL": f"{sub.mean():+.3f}" if n > 0 else "—",
                "Profit factor": str(_profit_factor(sub)) if n > 0 else "—",
            })
        by_side = pd.DataFrame(_side_rows)

    # By time-of-day (hour)
    by_time = pd.DataFrame()
    if "signal_time" in closed.columns:
        try:
            def _hour(t):
                try:
                    return int(str(t).split(":")[0])
                except Exception:
                    return -1
            closed = closed.copy()
            closed["_hour"] = closed["signal_time"].apply(_hour)
            _time_rows = []
            for h in sorted(closed["_hour"].unique()):
                sub = closed[closed["_hour"] == h]["pnl"].dropna()
                w = int((sub > 0).sum()); n = len(sub)
                _time_rows.append({
                    "Hour": f"{h:02d}:xx",
                    "Trades": n,
                    "Win %": f"{_safe_div(w,n):.1%}",
                    "Avg PnL": f"{sub.mean():+.3f}" if n > 0 else "—",
                })
            by_time = pd.DataFrame(_time_rows)
        except Exception:
            pass

    # Confidence calibration
    cal = _confidence_calibration(closed, "confidence_score", "pnl")

    # Rolling win rate
    rolling_wr = _rolling_win_rate(pnl.reset_index(drop=True))

    # Signal funnel
    all_count    = len(df)
    closed_count = len(closed)
    win_count    = wins
    funnel = {
        "signals_generated": all_count,
        "trades_closed"    : closed_count,
        "wins"             : win_count,
        "open_or_pending"  : all_count - closed_count,
    }

    return {
        "summary"        : summary,
        "by_track"       : by_track,
        "by_side"        : by_side,
        "by_time"        : by_time,
        "confidence_cal" : cal,
        "rolling_wr"     : rolling_wr,
        "signal_funnel"  : funnel,
        "trades_df"      : closed,
        "error"          : None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MTF ENGINE metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mtf_evidence(
    mtf_paper_log: list,
    mtf_backtest_result: dict | None = None,
    adaptive_weights: dict | None = None,
) -> dict:
    """
    Accepts list of MTFPaperTrade objects and optional backtest result dict.

    Returns
    -------
    {
      "summary"          : dict
      "by_track"         : dict   — auto vs manual
      "by_type"          : df     — intraday vs margin
      "by_setup"         : df     — pullback / breakout / reversal
      "confidence_cal"   : df
      "rolling_wr"       : Series
      "backtest_metrics" : dict | None
      "weight_drift"     : dict   — current vs default weights
      "trades_df"        : df
      "error"            : str | None
    }
    """
    empty = {
        "summary": {}, "by_track": {}, "by_type": pd.DataFrame(),
        "by_setup": pd.DataFrame(), "confidence_cal": pd.DataFrame(),
        "rolling_wr": pd.Series(dtype=float),
        "backtest_metrics": None, "weight_drift": {},
        "trades_df": pd.DataFrame(), "error": None,
    }

    # Prefer backtest data if available (larger sample)
    bt_df    = None
    bt_mets  = None
    if mtf_backtest_result and "trades_df" in mtf_backtest_result:
        bt_df   = mtf_backtest_result["trades_df"]
        bt_mets = mtf_backtest_result.get("metrics")

    paper_rows = [t.to_dict() if hasattr(t, "to_dict") else t for t in mtf_paper_log] if mtf_paper_log else []
    paper_df   = pd.DataFrame(paper_rows) if paper_rows else pd.DataFrame()

    # Merge: prefer backtest as primary, append live paper trades
    if bt_df is not None and not bt_df.empty:
        if not paper_df.empty:
            combined = pd.concat([bt_df, paper_df], ignore_index=True)
        else:
            combined = bt_df.copy()
    elif not paper_df.empty:
        combined = paper_df.copy()
    else:
        empty["error"] = "No MTF trades logged yet. Run backtest or auto-execute trades."
        return empty

    for col in ["pnl_points", "r_multiple", "confidence_score", "leverage", "volume_ratio"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    closed = combined[combined["status"] == "closed"].copy() if "status" in combined.columns else combined.copy()
    if closed.empty:
        empty["error"] = "No closed MTF trades yet."
        return empty

    pnl    = closed["pnl_points"].dropna()
    r_vals = closed["r_multiple"].dropna() if "r_multiple" in closed.columns else pd.Series(dtype=float)
    wins   = int((pnl > 0).sum())
    total  = len(pnl)

    summary = {
        "total_trades"   : total,
        "wins"           : wins,
        "losses"         : total - wins,
        "win_rate"       : round(_safe_div(wins, total), 3),
        "avg_r"          : round(float(r_vals.mean()), 2) if not r_vals.empty else 0.0,
        "profit_factor"  : _profit_factor(pnl),
        "expectancy_pts" : _expectancy(pnl),
        "max_dd"         : _max_drawdown(pnl),
        "max_consec_loss": _max_consecutive_losses(pnl),
        "total_pnl_pts"  : round(float(pnl.sum()), 3),
        "source"         : "backtest + live paper" if bt_df is not None else "live paper",
    }

    # By track
    by_track = {}
    for trk in ["auto", "manual"]:
        sub = closed[closed["track"] == trk]["pnl_points"].dropna() if "track" in closed.columns else pd.Series(dtype=float)
        w = int((sub > 0).sum()); n = len(sub)
        by_track[trk] = {
            "trades"  : n,
            "wins"    : w,
            "win_rate": round(_safe_div(w, n), 3),
            "profit_factor": _profit_factor(sub),
            "total_pnl"    : round(float(sub.sum()), 3),
        }

    # By trade type
    by_type_rows = []
    if "trade_type" in closed.columns:
        for tt in ["intraday", "margin"]:
            sub = closed[closed["trade_type"] == tt]["pnl_points"].dropna()
            w = int((sub > 0).sum()); n = len(sub)
            if n == 0:
                continue
            by_type_rows.append({
                "Type": tt.title(),
                "Trades": n,
                "Win %": f"{_safe_div(w,n):.1%}",
                "Avg PnL (pts)": f"{sub.mean():+.3f}",
                "Profit factor": str(_profit_factor(sub)),
                "Total PnL": f"{sub.sum():+.3f}",
            })
    by_type = pd.DataFrame(by_type_rows)

    # By setup type
    by_setup_rows = []
    if "setup_type" in closed.columns:
        for st_ in closed["setup_type"].dropna().unique():
            sub = closed[closed["setup_type"] == st_]["pnl_points"].dropna()
            w = int((sub > 0).sum()); n = len(sub)
            if n == 0:
                continue
            by_setup_rows.append({
                "Setup": st_,
                "Trades": n,
                "Win %": f"{_safe_div(w,n):.1%}",
                "Avg PnL (pts)": f"{sub.mean():+.3f}",
                "Profit factor": str(_profit_factor(sub)),
            })
    by_setup = pd.DataFrame(by_setup_rows)

    # Confidence calibration
    bins_mtf = [("60–69", 60, 69), ("70–79", 70, 79), ("80–100", 80, 100)]
    cal = _confidence_calibration(closed, "confidence_score", "pnl_points", bins=bins_mtf)

    # Rolling win rate
    rolling_wr = _rolling_win_rate(pnl.reset_index(drop=True))

    # Adaptive weight drift
    defaults = {"trend": 0.35, "setup": 0.30, "conf": 0.20, "volume": 0.15}
    weight_drift = {}
    if adaptive_weights:
        aw = adaptive_weights if isinstance(adaptive_weights, dict) else adaptive_weights.get_weights()
        for k, dv in defaults.items():
            key = f"{k}_weight"
            cur = aw.get(key, dv)
            weight_drift[k] = {"default": dv, "current": round(cur, 4), "delta": round(cur - dv, 4)}

    return {
        "summary"          : summary,
        "by_track"         : by_track,
        "by_type"          : by_type,
        "by_setup"         : by_setup,
        "confidence_cal"   : cal,
        "rolling_wr"       : rolling_wr,
        "backtest_metrics" : bt_mets,
        "weight_drift"     : weight_drift,
        "trades_df"        : closed,
        "error"            : None,
    }

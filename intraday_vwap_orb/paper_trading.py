from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .confidence import AUTO_TRADE_THRESHOLD, MANUAL_REVIEW_THRESHOLD

# ── Wallet constants ─────────────────────────────────────────────────────────
INTRADAY_WALLET_INITIAL = 100_000   # ₹1 lakh per wallet (auto + manual)
MTF_WALLET_INITIAL      = 2_500_000 # ₹25 lakh for MTF wallet

TRADE_LOG_SCHEMA = [
    "id", "trade_date", "signal_time", "exit_time", "side",
    "entry_price", "exit_price", "stop", "target",
    "qty", "pnl_pts", "pnl_inr",
    "confidence_score", "confidence_label",
    "volume_ratio", "or_range_pct", "gap_pct",
    "regime_band",
    "track", "status", "exit_reason", "r_multiple",
    "approved_by", "approved_at",
]


@dataclass
class PaperSignal:
    id: str
    trade_date: str
    signal_time: str
    side: str
    entry_price: float
    stop: float
    target: float
    confidence_score: int
    confidence_label: str
    volume_ratio: float
    or_range_pct: float
    gap_pct: Optional[float]
    track: str
    regime_band: str = "unknown"         # RT1: trend-regime tag at trade entry
    qty: int = 1                         # shares traded (computed from wallet)
    status: str = "open"
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None          # points (price difference)
    pnl_inr: Optional[float] = None      # pnl × qty in ₹
    r_multiple: Optional[float] = None
    approved_by: str = "auto"
    approved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trade_date": self.trade_date,
            "signal_time": self.signal_time,
            "exit_time": self.exit_time,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop": self.stop,
            "target": self.target,
            "qty": self.qty,
            "pnl_pts": self.pnl,
            "pnl_inr": self.pnl_inr,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "volume_ratio": self.volume_ratio,
            "or_range_pct": self.or_range_pct,
            "gap_pct": self.gap_pct,
            "regime_band": self.regime_band,
            "track": self.track,
            "status": self.status,
            "exit_reason": self.exit_reason,
            "r_multiple": self.r_multiple,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }


# ── Wallet helpers ────────────────────────────────────────────────────────────

MAX_DAY_LOSS_INR = 10_000   # Stop auto-trading for the day if total loss hits this

# ── Confidence → max-loss budget table ───────────────────────────────────────
# Bands: (score_from_inclusive, score_to_exclusive, max_loss_inr)
# Linear interpolation within each band; capped at ₹10,000 @ score=100.
_CONF_LOSS_BANDS = [
    (70,  75,  5_000,  5_000),   # 70–74: flat ₹5,000
    (75,  80,  5_000,  5_500),   # 75–79: ₹5,000 → ₹5,500  (+10%)
    (80,  85,  5_500,  6_000),   # 80–84: ₹5,500 → ₹6,000  (+20% of base)
    (85,  90,  6_000,  6_500),   # 85–89: ₹6,000 → ₹6,500  (+30% of base)
    (90,  95,  6_500,  7_500),   # 90–94: ₹6,500 → ₹7,500  (+50% of base)
    (95, 101,  7_500, 10_000),   # 95–100: ₹7,500 → ₹10,000
]


def max_loss_for_confidence(confidence_score: int) -> float:
    """
    Risk budget (max ₹ loss) per trade, scaled by confidence band.

    Band summary (linear interp within each):
      score 70–74  → ₹5,000 (flat)
      score 75–79  → ₹5,000–₹5,500  (+10%)
      score 80–84  → ₹5,500–₹6,000  (+20%)
      score 85–89  → ₹6,000–₹6,500  (+30%)
      score 90–94  → ₹6,500–₹7,500  (+50%)
      score 95–100 → ₹7,500–₹10,000
      below 70     → ₹5,000 (floor)
    """
    score = max(0, min(100, confidence_score))
    if score >= 100:
        return 10_000
    for lo, hi, v_lo, v_hi in _CONF_LOSS_BANDS:
        if lo <= score < hi:
            t = (score - lo) / (hi - lo)
            return v_lo + t * (v_hi - v_lo)
    return 5_000  # below auto threshold floor


def compute_qty(
    entry_price: float,
    stop_price: float,
    capital: float,
    confidence_score: int = AUTO_TRADE_THRESHOLD,
) -> int:
    """
    Risk-based position sizing.

    qty = floor(max_loss_budget / risk_per_share)
    Capped so total position value ≤ available capital.
    """
    if entry_price <= 0:
        return 1
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        risk_per_share = entry_price * 0.005   # 0.5% fallback
    budget = max_loss_for_confidence(confidence_score)
    qty = int(budget // risk_per_share)
    max_by_capital = int(capital // entry_price)
    return max(1, min(qty, max_by_capital))


# ── Intraday auto-trade guards ────────────────────────────────────────────────

def today_pnl_inr(log: list, track: str = "auto") -> float:
    """
    Sum of INR P&L for closed trades of `track` logged today (IST).
    Works for PaperSignal (pnl_inr) and MTFPaperTrade (pnl_points * position_size).
    """
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    total = 0.0
    for t in log:
        if t.track != track or t.status != "closed":
            continue
        pnl = getattr(t, "pnl_inr", None)
        if pnl is None:
            pts  = getattr(t, "pnl_points", None)
            size = getattr(t, "position_size", 1) or 1
            pnl  = pts * size if pts is not None else None
        if pnl is None:
            continue
        date_str = getattr(t, "trade_date", None) or getattr(t, "logged_at", None)
        try:
            if date_str and pd.Timestamp(str(date_str)[:10]).date() == today:
                total += pnl
        except Exception:
            pass
    return total


def has_open_trade(log: list, track: str = "auto") -> bool:
    """True if any trade for `track` is currently open (not yet closed)."""
    return any(t.track == track and t.status == "open" for t in log)


def auto_trade_blocked(log: list, track: str = "auto") -> tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).
    Blocks new auto trades when:
      1. An open trade already exists (sequential gate)
      2. Today's cumulative loss >= MAX_DAY_LOSS_INR
    """
    if has_open_trade(log, track):
        return True, "open_trade_exists"
    day_pnl = today_pnl_inr(log, track)
    if day_pnl <= -MAX_DAY_LOSS_INR:
        return True, f"daily_loss_limit_hit (₹{day_pnl:,.0f})"
    return False, ""


def wallet_balance(
    log: list,
    track: str,
    initial: float = INTRADAY_WALLET_INITIAL,
    return_stats: bool = False,
):
    """
    Compute current wallet balance for a given track ('auto'|'manual').
    - Starts at `initial`.
    - Each closed trade adds pnl_inr.
    - If balance hits 0 or below → auto-reload to initial (paper trading).

    B6: If return_stats=True returns dict with balance, reload_count, capital_injected.
    """
    balance          = initial
    reload_count     = 0
    capital_injected = 0.0
    for t in log:
        if t.track != track or t.status != "closed" or t.pnl_inr is None:
            continue
        balance += t.pnl_inr
        if balance <= 0:
            reload_count     += 1
            capital_injected += initial - balance   # amount topped up
            balance           = initial
    if return_stats:
        return {"balance": balance, "reload_count": reload_count,
                "capital_injected": capital_injected}
    return balance


def classify_signal(confidence_score: int) -> str:
    """Returns 'auto', 'manual_review', or 'rejected'."""
    if confidence_score >= AUTO_TRADE_THRESHOLD:
        return "auto"
    elif confidence_score >= MANUAL_REVIEW_THRESHOLD:
        return "manual_review"
    else:
        return "rejected"


_SQUARE_OFF_TIME = time(15, 15)   # hard NSE intraday cutoff


def resolve_paper_trade(
    paper_signal: PaperSignal,
    day_bars: pd.DataFrame,
    square_off_time: time = _SQUARE_OFF_TIME,
) -> PaperSignal:
    """
    Forward-simulate outcome on day_bars after entry.

    Exit priority per bar (whichever fires first):
      1. Stop-loss hit  (bar low ≤ stop for LONG, bar high ≥ stop for SHORT)
      2. Target hit     (bar high ≥ target for LONG, bar low ≤ target for SHORT)
      3. 15:15 hard square-off (bar time ≥ square_off_time → exit at bar open)

    If none of the above fire and bars run out → close at last bar close (EOD).
    """
    if day_bars.empty:
        paper_signal.status = "open"
        paper_signal.exit_reason = "awaiting_market"
        return paper_signal

    entry = paper_signal.entry_price
    stop  = paper_signal.stop
    target = paper_signal.target
    side  = paper_signal.side
    risk  = abs(entry - stop)
    if risk <= 0:
        paper_signal.status = "invalid"
        paper_signal.exit_reason = "zero_risk"
        return paper_signal

    # Detect datetime column (DatetimeIndex or column)
    dt_col = None
    if hasattr(day_bars.index, "hour"):
        dt_col = "__index__"
    else:
        for _c in day_bars.columns:
            if "datetime" in _c.lower() or (_c.lower() == "time" and
                    pd.api.types.is_datetime64_any_dtype(day_bars[_c])):
                dt_col = _c
                break

    def _bar_ts(bar) -> Optional[str]:
        try:
            if dt_col == "__index__":
                return pd.Timestamp(bar.name).strftime("%H:%M:%S")
            if dt_col:
                return pd.Timestamp(bar[dt_col]).strftime("%H:%M:%S")
        except Exception:
            pass
        return None

    def _set_exit(price: float, reason: str, bar=None) -> PaperSignal:
        paper_signal.status      = "closed"
        paper_signal.exit_price  = round(price, 2)
        paper_signal.exit_reason = reason
        pnl_pts = round((price - entry) if side == "LONG" else (entry - price), 2)
        paper_signal.pnl         = pnl_pts
        paper_signal.pnl_inr     = round(pnl_pts * paper_signal.qty, 2)
        paper_signal.r_multiple  = round(pnl_pts / risk, 3)
        paper_signal.exit_time   = _bar_ts(bar) if bar is not None else None
        return paper_signal

    entry_ts: Optional[pd.Timestamp] = None
    bars_since_entry = 0

    for _, bar in day_bars.iterrows():
        h = float(bar["high"])
        l = float(bar["low"])
        o = float(bar.get("open", bar["close"]))
        c = float(bar["close"])

        # Resolve bar timestamp
        bar_time: Optional[time] = None
        bar_ts:   Optional[pd.Timestamp] = None
        if dt_col == "__index__":
            try:
                bar_ts   = pd.Timestamp(bar.name)
                bar_time = bar_ts.time()
            except Exception:
                pass
        elif dt_col:
            try:
                bar_ts   = pd.Timestamp(bar[dt_col])
                bar_time = bar_ts.time()
            except Exception:
                pass

        if entry_ts is None and bar_ts is not None:
            entry_ts = bar_ts
        bars_since_entry += 1

        # 1. Hard square-off at 15:15
        if bar_time is not None and bar_time >= square_off_time:
            return _set_exit(o, "square_off_15:15", bar)

        # 2. Same-bar conflict resolution (B1: look-ahead bias fix)
        # If both stop and target are touched in the same bar, the level
        # closer to the bar open is assumed to have been hit first.
        _stop_touched   = (side == "LONG" and l <= stop)   or (side == "SHORT" and h >= stop)
        _target_touched = (side == "LONG" and h >= target) or (side == "SHORT" and l <= target)
        if _stop_touched and _target_touched:
            _dist_stop   = abs(o - stop)
            _dist_target = abs(o - target)
            if _dist_stop <= _dist_target:
                return _set_exit(stop,   "stop_hit",   bar)
            else:
                return _set_exit(target, "target_hit", bar)

        # 3. Stop-loss only
        if _stop_touched:
            return _set_exit(stop, "stop_hit", bar)

        # 3b. Early profit: +5% of entry within first 15 bars (≈15 min on 1m)
        # Checked before target so momentum captures are recorded correctly (B5)
        profit_pct = ((c - entry) / entry) if side == "LONG" else ((entry - c) / entry)
        if bars_since_entry <= 15 and profit_pct >= 0.05 and not _target_touched:
            return _set_exit(c, "early_profit_5pct", bar)

        # 4. Target
        if _target_touched:
            return _set_exit(target, "target_hit", bar)

    # Bars exhausted — EOD close
    return _set_exit(float(day_bars.iloc[-1]["close"]), "eod_square_off", day_bars.iloc[-1])


EXIT_LABEL_MAP = {
    "awaiting_market":   "⏳ open",
    "stop_hit":          "🛑 stop hit",
    "target_hit":        "🎯 target hit",
    "square_off_15:15":  "🔔 sq-off 15:15",
    "eod_square_off":    "🔔 EOD sq-off",
    "early_profit_5pct": "💰 early +5%",
    "prob_target_low":   "📉 prob<50% close",
    "prob_stop_high":    "⚡ stop-prob>75% close",
    "force_closed":      "🔴 force closed",
    "zero_risk":         "⚠️ invalid",
}


def live_trade_probability(
    trade: PaperSignal,
    recent_bars: pd.DataFrame,
) -> dict:
    """
    Estimate real-time probability of target hit and stop hit
    from the most recent candle data (1m or 5m — whichever is passed).

    Method:
    - ATR(last N bars) → volatility gauge
    - Momentum: close vs entry (directional drift)
    - Distance ratio: how far target/stop is relative to ATR

    Returns dict with keys:
        target_prob  : float 0–1  (probability of reaching target)
        stop_prob    : float 0–1  (probability of hitting stop)
        current_price: float
        pnl_pts      : float (unrealised points)
        pnl_inr      : float (unrealised INR)
    """
    if recent_bars.empty or trade.entry_price <= 0:
        return {"target_prob": 0.5, "stop_prob": 0.5,
                "current_price": trade.entry_price,
                "pnl_pts": 0.0, "pnl_inr": 0.0}

    bars = recent_bars.tail(20).copy()
    try:
        highs  = pd.to_numeric(bars["high"],  errors="coerce")
        lows   = pd.to_numeric(bars["low"],   errors="coerce")
        closes = pd.to_numeric(bars["close"], errors="coerce")
        atr    = (highs - lows).mean()
        if atr <= 0:
            atr = trade.entry_price * 0.003
    except Exception:
        atr = trade.entry_price * 0.003

    current_price = float(recent_bars.iloc[-1]["close"])
    entry  = trade.entry_price
    target = trade.target
    stop   = trade.stop
    side   = trade.side

    # Directional unrealised P&L
    pnl_pts = (current_price - entry) if side == "LONG" else (entry - current_price)
    pnl_inr = round(pnl_pts * trade.qty, 2)

    # Distance to target and stop (in ATR units)
    dist_to_target = abs(target - current_price)
    dist_to_stop   = abs(current_price - stop)

    # Momentum: fraction of last N bars that moved in trade direction
    if len(closes) >= 2:
        diffs = closes.diff().dropna()
        favour = (diffs > 0).sum() if side == "LONG" else (diffs < 0).sum()
        momentum = favour / len(diffs)  # 0–1
    else:
        momentum = 0.5

    # Probability model (B3: normalised so target_prob + stop_prob = 1):
    # Raw scores from momentum × distance decay, then softmax-normalise
    # so they are complementary and interpretable as true probabilities.
    _raw_target = momentum       * math.exp(-dist_to_target / atr)
    _raw_stop   = (1.0 - momentum) * math.exp(-dist_to_stop   / atr)
    _total = _raw_target + _raw_stop
    if _total <= 0:
        _total = 1.0
    target_prob = _raw_target / _total
    stop_prob   = _raw_stop   / _total
    # Clamp to [0.05, 0.95] to avoid degenerate certainty
    target_prob = max(0.05, min(0.95, target_prob))
    stop_prob   = max(0.05, min(0.95, stop_prob))
    # Re-normalise after clamping
    _total2 = target_prob + stop_prob
    target_prob = round(target_prob / _total2, 3)
    stop_prob   = round(stop_prob   / _total2, 3)

    return {
        "target_prob":   round(target_prob, 3),
        "stop_prob":     round(stop_prob,   3),
        "current_price": round(current_price, 2),
        "pnl_pts":       round(pnl_pts, 2),
        "pnl_inr":       pnl_inr,
    }


def force_close_trade(trade: PaperSignal, current_price: float) -> PaperSignal:
    """Close an open trade immediately at current_price with reason 'force_closed'."""
    entry  = trade.entry_price
    side   = trade.side
    risk   = abs(entry - trade.stop)
    trade.status      = "closed"
    trade.exit_price  = round(current_price, 2)
    trade.exit_reason = "force_closed"
    trade.exit_time   = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S")
    pnl_pts           = round((current_price - entry) if side == "LONG" else (entry - current_price), 2)
    trade.pnl         = pnl_pts
    trade.pnl_inr     = round(pnl_pts * trade.qty, 2)
    trade.r_multiple  = round(pnl_pts / risk, 3) if risk > 0 else 0.0
    return trade


# ── RT1: Regime classifier ────────────────────────────────────────────────────

# Regime bands defined by OR-range pct (opening-range breakout width relative to price)
# and volume ratio at entry.
# Labels: "trending_strong" | "trending_mild" | "range_bound" | "choppy"
def classify_regime(or_range_pct: float, volume_ratio: float) -> str:
    """
    Tag market regime at trade entry using two structural features:
      - or_range_pct: opening range width as % of price (narrow = range-bound)
      - volume_ratio: entry bar volume vs 20-bar average (high = participation)

    Regime map:
      high vol + wide OR  → trending_strong
      high vol + narrow OR → trending_mild  (breakout attempt)
      low vol  + wide OR  → range_bound
      low vol  + narrow OR → choppy
    """
    try:
        orp = float(or_range_pct or 0)
        vr  = float(volume_ratio or 1)
    except (TypeError, ValueError):
        return "unknown"

    high_vol  = vr  >= 1.5
    wide_or   = orp >= 0.5   # 0.5% of price is a meaningful intraday range

    if high_vol and wide_or:
        return "trending_strong"
    elif high_vol and not wide_or:
        return "trending_mild"
    elif not high_vol and wide_or:
        return "range_bound"
    else:
        return "choppy"


# ── LR1: Logistic regression probability engine ───────────────────────────────

# Minimum closed trades required before activating LR model
LR_MIN_TRADES = 100

# Feature names used by the LR model (must be numeric, present in trade log)
LR_FEATURE_COLS = [
    "confidence_score",
    "volume_ratio",
    "or_range_pct",
]


def _build_lr_model(closed_df: pd.DataFrame):
    """
    Fit a logistic regression model on closed trade history.
    Target: 1 if trade was a win (pnl_inr > 0), 0 otherwise.
    Features: LR_FEATURE_COLS.

    Returns (model, scaler, feature_cols_used, brier_score) or None if sklearn unavailable.
    Uses Platt scaling (sigmoid calibration) via CalibratedClassifierCV.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import brier_score_loss
    except ImportError:
        return None

    pnl_col = "pnl_inr" if "pnl_inr" in closed_df.columns else "pnl_pts"
    feat_cols = [c for c in LR_FEATURE_COLS if c in closed_df.columns]
    if not feat_cols:
        return None

    df = closed_df.copy()
    df["_target"] = (pd.to_numeric(df[pnl_col], errors="coerce") > 0).astype(int)
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df = df[feat_cols + ["_target"]].dropna()

    if len(df) < LR_MIN_TRADES or df["_target"].nunique() < 2:
        return None

    X = df[feat_cols].values
    y = df["_target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = LogisticRegression(max_iter=500, random_state=42)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_scaled, y)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    brier  = round(float(brier_score_loss(y, y_prob)), 4)

    return {"model": model, "scaler": scaler, "features": feat_cols, "brier": brier,
            "n_train": len(df), "win_rate": float(y.mean())}


def lr_trade_probability(
    trade: "PaperSignal",
    recent_bars: pd.DataFrame,
    lr_model_bundle: Optional[dict],
) -> dict:
    """
    Compute target/stop probabilities using LR model if available and trained,
    otherwise fall back to the heuristic live_trade_probability().

    Returns same dict schema as live_trade_probability() plus 'model_used' key.
    """
    base = live_trade_probability(trade, recent_bars)

    if lr_model_bundle is None:
        base["model_used"] = "heuristic"
        return base

    try:
        feat_vals = []
        feat_map  = {
            "confidence_score": float(trade.confidence_score),
            "volume_ratio":     float(trade.volume_ratio),
            "or_range_pct":     float(trade.or_range_pct or 0),
        }
        for f in lr_model_bundle["features"]:
            feat_vals.append(feat_map.get(f, 0.0))

        X = np.array(feat_vals).reshape(1, -1)
        X_sc = lr_model_bundle["scaler"].transform(X)
        win_prob = float(lr_model_bundle["model"].predict_proba(X_sc)[0, 1])

        # Map win_prob → target_prob / stop_prob
        # Win ≈ target hit, so use win_prob as target_prob
        # Blend 30% heuristic to retain distance-decay signal
        blend = 0.7
        target_prob = blend * win_prob + (1 - blend) * base["target_prob"]
        stop_prob   = 1.0 - target_prob
        # Clamp and re-normalise
        target_prob = max(0.05, min(0.95, target_prob))
        stop_prob   = max(0.05, min(0.95, stop_prob))
        _t = target_prob + stop_prob
        base["target_prob"] = round(target_prob / _t, 3)
        base["stop_prob"]   = round(stop_prob   / _t, 3)
        base["model_used"]  = "logistic_regression"
    except Exception:
        base["model_used"] = "heuristic_fallback"

    return base


# ── WF1: Walk-forward validation helper ──────────────────────────────────────

def walk_forward_threshold_sweep(
    closed_df: pd.DataFrame,
    pnl_col: str = "pnl_inr",
    train_pct: float = 0.70,
    thresholds: Optional[list] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Walk-forward validation for threshold sweep.

    Splits closed trades chronologically: first train_pct% as training,
    remaining as out-of-sample test. For each threshold candidate,
    reports in-sample and out-of-sample win rate and Sharpe ratio.

    Returns (results_df, warning_message).
    """
    if thresholds is None:
        thresholds = list(range(55, 96, 5))

    if "confidence_score" not in closed_df.columns or len(closed_df) < 10:
        return pd.DataFrame(), "Need at least 10 closed trades for walk-forward validation."

    # Sort by trade_date chronologically
    df = closed_df.copy()
    df["_date_parsed"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    split_idx = int(len(df) * train_pct)
    if split_idx < 5 or (len(df) - split_idx) < 3:
        return pd.DataFrame(), (
            f"Not enough trades for a reliable split "
            f"(train={split_idx}, test={len(df)-split_idx}). Need ≥10 total."
        )

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    train_cutoff = str(train_df["_date_parsed"].max().date())
    test_start   = str(test_df["_date_parsed"].min().date())

    scores_tr  = pd.to_numeric(train_df["confidence_score"], errors="coerce")
    scores_te  = pd.to_numeric(test_df["confidence_score"],  errors="coerce")
    pnls_tr    = pd.to_numeric(train_df[pnl_col], errors="coerce")
    pnls_te    = pd.to_numeric(test_df[pnl_col],  errors="coerce")

    def _sharpe(pnl_series: pd.Series) -> float:
        s = pnl_series.dropna()
        if len(s) < 2 or s.std() == 0:
            return 0.0
        return round(float(s.mean() / s.std() * math.sqrt(252)), 2)

    rows = []
    for thresh in thresholds:
        tr_sub  = train_df[scores_tr >= thresh]
        te_sub  = test_df[scores_te  >= thresh]
        tr_pnl  = pd.to_numeric(tr_sub[pnl_col], errors="coerce")
        te_pnl  = pd.to_numeric(te_sub[pnl_col], errors="coerce")

        tr_wr   = (tr_pnl > 0).mean() if len(tr_sub) else float("nan")
        te_wr   = (te_pnl > 0).mean() if len(te_sub) else float("nan")
        tr_sh   = _sharpe(tr_pnl)
        te_sh   = _sharpe(te_pnl)

        overfit = (
            not (math.isnan(tr_wr) or math.isnan(te_wr))
            and tr_wr - te_wr > 0.15
        )

        rows.append({
            "threshold":       thresh,
            "train_n":         len(tr_sub),
            "train_wr":        f"{tr_wr:.0%}" if not math.isnan(tr_wr) else "—",
            "train_sharpe":    tr_sh,
            "test_n":          len(te_sub),
            "test_wr":         f"{te_wr:.0%}" if not math.isnan(te_wr) else "—",
            "test_sharpe":     te_sh,
            "overfit_warning": "⚠️ overfit" if overfit else "✅",
        })

    warning = (
        f"Training: trades up to {train_cutoff} (n={split_idx})  |  "
        f"Out-of-sample test: from {test_start} (n={len(df)-split_idx}).  "
        f"Rows where train win rate > test win rate by >15pp are flagged ⚠️ overfit."
    )
    return pd.DataFrame(rows), warning


def paper_log_to_df(log: list) -> pd.DataFrame:
    if not log:
        return pd.DataFrame(columns=TRADE_LOG_SCHEMA)
    return pd.DataFrame([t.to_dict() for t in log])


def summarise_track(df: pd.DataFrame, track: str) -> dict:
    sub = df[df["track"] == track].copy() if not df.empty and "track" in df.columns else pd.DataFrame()
    if sub.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl_inr": 0.0, "avg_pnl_inr": 0.0, "avg_r": 0.0}
    closed = sub[sub["status"] == "closed"]
    pnl_col = "pnl_inr" if "pnl_inr" in closed.columns else "pnl_pts" if "pnl_pts" in closed.columns else "pnl"
    pnl_vals = pd.to_numeric(closed[pnl_col], errors="coerce")
    wins   = closed[pnl_vals > 0]
    losses = closed[pnl_vals <= 0]
    r_vals = pd.to_numeric(closed["r_multiple"], errors="coerce").dropna()
    n = len(closed)
    return {
        "trades":       n,
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     len(wins) / n if n > 0 else 0.0,
        "total_pnl_inr": float(pnl_vals.sum()),
        "avg_pnl_inr":   float(pnl_vals.mean()) if n > 0 else 0.0,
        "avg_r":         float(r_vals.mean()) if not r_vals.empty else 0.0,
    }

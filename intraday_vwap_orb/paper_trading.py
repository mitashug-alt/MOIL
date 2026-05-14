from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Optional

import pandas as pd

from .confidence import AUTO_TRADE_THRESHOLD, MANUAL_REVIEW_THRESHOLD


TRADE_LOG_SCHEMA = [
    "id", "trade_date", "signal_time", "side",
    "entry_price", "stop", "target", "confidence_score", "confidence_label",
    "volume_ratio", "or_range_pct", "gap_pct",
    "track", "status", "exit_price", "exit_reason",
    "pnl", "r_multiple", "approved_by", "approved_at",
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
    status: str = "open"
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    approved_by: str = "auto"
    approved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trade_date": self.trade_date,
            "signal_time": self.signal_time,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop": self.stop,
            "target": self.target,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "volume_ratio": self.volume_ratio,
            "or_range_pct": self.or_range_pct,
            "gap_pct": self.gap_pct,
            "track": self.track,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "r_multiple": self.r_multiple,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }


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

    def _set_exit(price: float, reason: str) -> PaperSignal:
        paper_signal.status     = "closed"
        paper_signal.exit_price = round(price, 2)
        paper_signal.exit_reason = reason
        paper_signal.pnl        = round((price - entry) if side == "LONG" else (entry - price), 2)
        paper_signal.r_multiple = round(paper_signal.pnl / risk, 3)
        return paper_signal

    # Detect datetime column (support DatetimeIndex or column named datetime/time)
    dt_col = None
    if hasattr(day_bars.index, "hour"):
        dt_col = "__index__"
    else:
        for _c in day_bars.columns:
            if "datetime" in _c.lower() or (_c.lower() == "time" and
                    pd.api.types.is_datetime64_any_dtype(day_bars[_c])):
                dt_col = _c
                break

    for _, bar in day_bars.iterrows():
        h = float(bar["high"])
        l = float(bar["low"])
        o = float(bar.get("open", bar["close"]))
        c = float(bar["close"])

        # Resolve bar timestamp
        bar_time: Optional[time] = None
        if dt_col == "__index__":
            bar_time = pd.Timestamp(bar.name).time()
        elif dt_col:
            bar_time = pd.Timestamp(bar[dt_col]).time()

        # 1. Hard square-off at 15:15 — exit at bar open (best approximation)
        if bar_time is not None and bar_time >= square_off_time:
            return _set_exit(o, "square_off_15:15")

        # 2. Stop-loss
        if side == "LONG" and l <= stop:
            return _set_exit(stop, "stop_hit")
        if side == "SHORT" and h >= stop:
            return _set_exit(stop, "stop_hit")

        # 3. Target
        if side == "LONG" and h >= target:
            return _set_exit(target, "target_hit")
        if side == "SHORT" and l <= target:
            return _set_exit(target, "target_hit")

    # Bars exhausted without hitting any exit — close at last bar close
    return _set_exit(float(day_bars.iloc[-1]["close"]), "eod_square_off")


def paper_log_to_df(log: list) -> pd.DataFrame:
    if not log:
        return pd.DataFrame(columns=TRADE_LOG_SCHEMA)
    return pd.DataFrame([t.to_dict() for t in log])


def summarise_track(df: pd.DataFrame, track: str) -> dict:
    sub = df[df["track"] == track].copy() if not df.empty and "track" in df.columns else pd.DataFrame()
    if sub.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_r": 0.0}
    closed = sub[sub["status"] == "closed"]
    wins = closed[pd.to_numeric(closed["pnl"], errors="coerce") > 0]
    losses = closed[pd.to_numeric(closed["pnl"], errors="coerce") <= 0]
    r_vals = pd.to_numeric(closed["r_multiple"], errors="coerce").dropna()
    return {
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) if len(closed) > 0 else 0.0,
        "total_pnl": float(pd.to_numeric(closed["pnl"], errors="coerce").sum()),
        "avg_r": float(r_vals.mean()) if not r_vals.empty else 0.0,
    }

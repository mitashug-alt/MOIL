from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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


def resolve_paper_trade(
    paper_signal: PaperSignal,
    day_bars: pd.DataFrame,
) -> PaperSignal:
    """
    Forward-simulate outcome on day_bars after entry.
    Updates status, exit_price, exit_reason, pnl, r_multiple in-place.
    """
    if day_bars.empty:
        paper_signal.status = "unresolved"
        return paper_signal

    entry = paper_signal.entry_price
    stop = paper_signal.stop
    target = paper_signal.target
    side = paper_signal.side
    risk = abs(entry - stop)
    if risk <= 0:
        paper_signal.status = "unresolved"
        return paper_signal

    for _, bar in day_bars.iterrows():
        h, l = float(bar["high"]), float(bar["low"])
        if side == "LONG":
            if l <= stop:
                paper_signal.status = "closed"
                paper_signal.exit_price = stop
                paper_signal.exit_reason = "stop_hit"
                paper_signal.pnl = (stop - entry)
                paper_signal.r_multiple = paper_signal.pnl / risk
                return paper_signal
            if h >= target:
                paper_signal.status = "closed"
                paper_signal.exit_price = target
                paper_signal.exit_reason = "target_hit"
                paper_signal.pnl = (target - entry)
                paper_signal.r_multiple = paper_signal.pnl / risk
                return paper_signal
        else:
            if h >= stop:
                paper_signal.status = "closed"
                paper_signal.exit_price = stop
                paper_signal.exit_reason = "stop_hit"
                paper_signal.pnl = (entry - stop)
                paper_signal.r_multiple = paper_signal.pnl / risk
                return paper_signal
            if l <= target:
                paper_signal.status = "closed"
                paper_signal.exit_price = target
                paper_signal.exit_reason = "target_hit"
                paper_signal.pnl = (entry - target)
                paper_signal.r_multiple = paper_signal.pnl / risk
                return paper_signal

    last_close = float(day_bars.iloc[-1]["close"])
    paper_signal.status = "closed"
    paper_signal.exit_price = last_close
    paper_signal.exit_reason = "eod_square_off"
    if side == "LONG":
        paper_signal.pnl = last_close - entry
    else:
        paper_signal.pnl = entry - last_close
    paper_signal.r_multiple = paper_signal.pnl / risk
    return paper_signal


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

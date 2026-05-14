from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd


AUTO_TRADE_THRESHOLD: int = 70
MANUAL_REVIEW_THRESHOLD: int = 45


@dataclass
class ConfidenceResult:
    score: int
    label: str
    breakdown: Dict[str, Any]
    auto_trade: bool
    manual_review: bool


def score_signal_confidence(
    row: pd.Series,
    volume_ratio: float,
    or_range_pct: float,
    gap_pct: float | None,
    side: str,
    signal_time: Any,
    or_high: float,
    or_low: float,
) -> ConfidenceResult:
    """
    Score a single signal 0-100 across five pillars.
    Returns ConfidenceResult with per-pillar breakdown.
    """
    breakdown: Dict[str, Any] = {}
    total = 0

    # ── Pillar 1: Volume strength (0-25 pts) ────────────────────────────
    if volume_ratio >= 4.0:
        vol_pts = 25
    elif volume_ratio >= 3.0:
        vol_pts = 20
    elif volume_ratio >= 2.5:
        vol_pts = 17
    elif volume_ratio >= 2.0:
        vol_pts = 13
    elif volume_ratio >= 1.5:
        vol_pts = 8
    else:
        vol_pts = 3
    breakdown["volume_ratio"] = {"value": round(volume_ratio, 2), "pts": vol_pts, "max": 25}
    total += vol_pts

    # ── Pillar 2: VWAP distance (0-20 pts) ──────────────────────────────
    vwap = row.get("vwap") if hasattr(row, "get") else getattr(row, "vwap", None)
    close = row.get("close") if hasattr(row, "get") else getattr(row, "close", None)
    if vwap and vwap > 0 and close:
        vwap_dist_pct = abs(close - vwap) / vwap * 100
        if side == "LONG":
            above_vwap = close > vwap
        else:
            above_vwap = close < vwap
        if not above_vwap:
            vwap_pts = 0
        elif vwap_dist_pct >= 0.5:
            vwap_pts = 20
        elif vwap_dist_pct >= 0.25:
            vwap_pts = 14
        elif vwap_dist_pct >= 0.1:
            vwap_pts = 8
        else:
            vwap_pts = 4
    else:
        # Fix #9: no VWAP data → 0 pts (was 10, which inflated scores on missing data)
        vwap_dist_pct = None
        vwap_pts = 0
    breakdown["vwap_distance"] = {"value": round(vwap_dist_pct, 3) if vwap_dist_pct is not None else None, "pts": vwap_pts, "max": 20}
    total += vwap_pts

    # ── Pillar 3: OR range quality (0-20 pts) ────────────────────
    # Fix #10: Tight ORs produce cleaner breakouts — score inverted.
    # Optimal OR = 0.3–0.8% (tight, well-defined); wide OR ≥ 1.5% is messy/fakeout-prone.
    if or_range_pct <= 0.3:
        or_pts = 20   # very tight OR — best R:R
    elif or_range_pct <= 0.6:
        or_pts = 16
    elif or_range_pct <= 1.0:
        or_pts = 10
    elif or_range_pct <= 1.5:
        or_pts = 5
    else:
        or_pts = 2    # wide OR (≥1.5%) — high false-breakout risk
    breakdown["or_range_pct"] = {"value": round(or_range_pct, 3), "pts": or_pts, "max": 20}
    total += or_pts

    # ── Pillar 4: Gap risk (0-15 pts — penalise large gaps) ──────────────
    if gap_pct is None:
        gap_pts = 10
    elif gap_pct <= 0.3:
        gap_pts = 15
    elif gap_pct <= 0.8:
        gap_pts = 12
    elif gap_pct <= 1.5:
        gap_pts = 7
    elif gap_pct <= 2.0:
        gap_pts = 3
    else:
        gap_pts = 0
    breakdown["gap_pct"] = {"value": round(gap_pct, 3) if gap_pct is not None else None, "pts": gap_pts, "max": 15}
    total += gap_pts

    # ── Pillar 5: EMA20 alignment (0-10 pts) ────────────────────────────
    ema20 = row.get("ema20") if hasattr(row, "get") else getattr(row, "ema20", None)
    if ema20 and ema20 > 0 and close:
        if side == "LONG":
            ema_aligned = close > ema20
        else:
            ema_aligned = close < ema20
        ema_pts = 10 if ema_aligned else 0
    else:
        ema_pts = 5
        ema_aligned = None
    breakdown["ema20_aligned"] = {"value": ema_aligned, "pts": ema_pts, "max": 10}
    total += ema_pts

    # ── Pillar 6: Time-of-day quality (0-10 pts) ────────────────────────
    try:
        if hasattr(signal_time, "hour"):
            h, m = signal_time.hour, signal_time.minute
        else:
            parts = str(signal_time).split(":")
            h, m = int(parts[0]), int(parts[1])
        total_mins = h * 60 + m
        if 570 <= total_mins <= 595:
            time_pts = 10
        elif 595 < total_mins <= 630:
            time_pts = 7
        elif 630 < total_mins <= 660:
            time_pts = 4
        else:
            time_pts = 1
    except Exception:
        time_pts = 5
    breakdown["time_of_day"] = {"value": str(signal_time), "pts": time_pts, "max": 10}
    total += time_pts

    total = min(100, max(0, total))

    if total >= 80:
        label = "Very High"
    elif total >= AUTO_TRADE_THRESHOLD:
        label = "High"
    elif total >= MANUAL_REVIEW_THRESHOLD:
        label = "Medium"
    elif total >= 25:
        label = "Low"
    else:
        label = "Very Low"

    return ConfidenceResult(
        score=total,
        label=label,
        breakdown=breakdown,
        auto_trade=total >= AUTO_TRADE_THRESHOLD,
        manual_review=MANUAL_REVIEW_THRESHOLD <= total < AUTO_TRADE_THRESHOLD,
    )

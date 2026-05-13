from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from datetime import time
from typing import List, Tuple, Dict, Any

from .config import StrategyConfig


@dataclass
class Signal:
    idx: int
    side: str  # LONG or SHORT
    reason: str


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["vwap"] = out["vwap"]
    out["ema20"] = out["ema20"]
    out["date"] = out["datetime"].dt.date
    out["time"] = out["datetime"].dt.time
    return out


def volume_ratio_ok(df: pd.DataFrame, i: int, lookback: int, thresh: float) -> tuple[bool, float, float]:
    start = max(0, i - lookback)
    window = df.iloc[start:i]
    avg_vol = window["volume"].mean() if not window.empty else float("nan")
    cur_vol = df.iloc[i]["volume"]
    ratio = cur_vol / avg_vol if avg_vol and avg_vol == avg_vol and avg_vol > 0 else 0
    return ratio >= thresh, cur_vol, avg_vol


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig, or_high: float, or_low: float) -> List[Signal]:
    signals, stats = generate_signals_with_stats(df, cfg, or_high, or_low)
    return signals


def generate_signals_with_stats(df: pd.DataFrame, cfg: StrategyConfig, or_high: float, or_low: float) -> Tuple[List[Signal], Dict[str, Any]]:
    signals: List[Signal] = []
    entry_start = pd.to_datetime(cfg.entry_start).time()
    entry_end = pd.to_datetime(cfg.entry_end).time()
    rejected_low_volume = 0
    rejected_below_vwap_long = 0
    rejected_above_vwap_short = 0
    first_breakout_side = None
    first_breakout_time = None
    first_breakout_volume_ratio = None
    for i, row in df.iterrows():
        t: time = row["time"]
        if t < entry_start or t > entry_end:
            continue
        ok_vol, cur_vol, avg_vol = volume_ratio_ok(df, i, cfg.volume_lookback_bars, cfg.volume_multiplier)
        ratio = cur_vol / avg_vol if avg_vol and avg_vol == avg_vol else 0
        if not ok_vol:
            rejected_low_volume += 1
            continue
        if cfg.allow_long and row["close"] > or_high:
            if row["close"] > row["vwap"]:
                if not cfg.require_ema_filter or row["close"] > row.get("ema20", row["close"]):
                    signals.append(Signal(i, "LONG", f"breakout high vol {cur_vol:.0f}/{avg_vol:.0f}"))
                    if first_breakout_side is None:
                        first_breakout_side = "LONG"
                        first_breakout_time = row["time"]
                        first_breakout_volume_ratio = ratio
                else:
                    rejected_below_vwap_long += 1
            else:
                rejected_below_vwap_long += 1
        if cfg.allow_short and row["close"] < or_low:
            if row["close"] < row["vwap"]:
                if not cfg.require_ema_filter or row["close"] < row.get("ema20", row["close"]):
                    signals.append(Signal(i, "SHORT", f"breakdown high vol {cur_vol:.0f}/{avg_vol:.0f}"))
                    if first_breakout_side is None:
                        first_breakout_side = "SHORT"
                        first_breakout_time = row["time"]
                        first_breakout_volume_ratio = ratio
                else:
                    rejected_above_vwap_short += 1
            else:
                rejected_above_vwap_short += 1
    stats = {
        "valid_long_signal_count": len([s for s in signals if s.side == "LONG"]),
        "valid_short_signal_count": len([s for s in signals if s.side == "SHORT"]),
        "rejected_low_volume_signal_count": rejected_low_volume,
        "rejected_below_vwap_long_count": rejected_below_vwap_long,
        "rejected_above_vwap_short_count": rejected_above_vwap_short,
        "first_breakout_side": first_breakout_side,
        "first_breakout_time": first_breakout_time,
        "first_breakout_volume_ratio": first_breakout_volume_ratio,
        "vwap_breakout_signal": len(signals) > 0,
    }
    return signals, stats

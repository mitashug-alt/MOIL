from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Tuple

import pandas as pd

from .config import StrategyConfig
from .signals import Signal, generate_signals_with_stats


@dataclass
class Trade:
    symbol: str
    trade_date: str
    entry_datetime: datetime
    exit_datetime: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    opening_range_high: float
    opening_range_low: float
    entry_vwap: float
    entry_ema20: float
    breakout_volume: float
    previous_5_candle_avg_volume: float
    breakout_volume_ratio: float
    initial_stop: float
    final_stop: float
    target: float
    risk_per_share: float
    planned_r_multiple: float
    realized_r_multiple: float
    gross_pnl: float
    estimated_costs: float
    net_pnl: float
    exit_reason: str
    bars_held: int
    one_trade_per_day_flag: bool
    sector_filter_passed: bool
    notes: str = ""


def _within(t: time, start: time, end: time) -> bool:
    return start <= t <= end


def _price_hits(price: float, high: float, low: float) -> bool:
    return low <= price <= high


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    trades: List[Trade] = []
    entry_start = pd.to_datetime(cfg.entry_start).time()
    entry_end = pd.to_datetime(cfg.entry_end).time()
    square_off = pd.to_datetime(cfg.square_off_time).time()

    taken_dates: set[str] = set()
    stats_by_date: dict[str, dict] = {}

    # generate signals per day to honor one-trade-per-day
    prev_close: dict[str, float] = {}  # symbol -> prior day close for gap filter

    for trade_date, day_df in df.groupby(df["datetime"].dt.date):
        day_df = day_df.reset_index(drop=True)
        or_high = day_df["opening_range_high"].iloc[0]
        or_low = day_df["opening_range_low"].iloc[0]
        sym = day_df["symbol"].iloc[0]

        # ── OR range filter ──────────────────────────────────────────────
        or_range_pct = (or_high - or_low) / or_low * 100 if pd.notna(or_high) and pd.notna(or_low) and or_low and or_low != 0 else 0
        if cfg.min_or_range_pct > 0 and or_range_pct < cfg.min_or_range_pct:
            prev_close[sym] = float(day_df["close"].iloc[-1])
            stats_by_date[str(trade_date)] = {
                "skipped": True, "skip_reason": f"OR range {or_range_pct:.2f}% < min {cfg.min_or_range_pct}%",
                "orb_high": or_high, "orb_low": or_low, "orb_range_pct": or_range_pct,
                "gap_pct": None, "valid_long_signal_count": 0, "valid_short_signal_count": 0,
                "rejected_low_volume_signal_count": 0, "rejected_below_vwap_long_count": 0,
                "rejected_above_vwap_short_count": 0, "first_breakout_side": None,
                "first_breakout_time": None, "first_breakout_volume_ratio": None,
                "vwap_breakout_signal": False,
            }
            continue

        # ── Gap filter ───────────────────────────────────────────────────
        day_open = float(day_df["open"].iloc[0])
        prior_close = prev_close.get(sym)
        gap_pct = abs(day_open - prior_close) / prior_close * 100 if prior_close and prior_close != 0 else None
        if cfg.max_gap_pct > 0 and prior_close and prior_close != 0 and gap_pct > cfg.max_gap_pct:
            prev_close[sym] = float(day_df["close"].iloc[-1])
            stats_by_date[str(trade_date)] = {
                "skipped": True, "skip_reason": f"Gap {gap_pct:.2f}% > max {cfg.max_gap_pct}%",
                "orb_high": or_high, "orb_low": or_low, "orb_range_pct": or_range_pct, "gap_pct": gap_pct,
                "valid_long_signal_count": 0, "valid_short_signal_count": 0,
                "rejected_low_volume_signal_count": 0, "rejected_below_vwap_long_count": 0,
                "rejected_above_vwap_short_count": 0, "first_breakout_side": None,
                "first_breakout_time": None, "first_breakout_volume_ratio": None,
                "vwap_breakout_signal": False,
            }
            continue

        sigs, stats = generate_signals_with_stats(day_df, cfg, or_high, or_low)
        stats_by_date[str(trade_date)] = stats | {
            "orb_high": or_high,
            "orb_low": or_low,
            "orb_range_pct": or_range_pct,
            "gap_pct": gap_pct,
        }

        for sig in sigs:
            trade_date_key = str(trade_date)
            if cfg.one_trade_per_day and trade_date_key in taken_dates:
                continue

            sig_row = day_df.iloc[sig.idx]
            trade_date_str = str(sig_row["datetime"].date())
            t: time = sig_row["time"]
            if not _within(t, entry_start, entry_end):
                continue

            # ── Entry price: next bar open (no look-ahead bias) ──────────
            if cfg.entry_on_next_bar_open:
                entry_bar_idx = sig.idx + 1
                if entry_bar_idx >= len(day_df):
                    continue  # signal on last bar — no entry bar
                entry_bar = day_df.iloc[entry_bar_idx]
                entry = entry_bar["open"]
                entry_datetime = entry_bar["datetime"]
                # same-bar guard: skip if entry bar already triggered stop/target
                if cfg.conservative_same_bar_exit:
                    pass  # will be enforced in forward simulation starting at entry_bar_idx+1
                sim_start = entry_bar_idx + 1
            else:
                entry_bar_idx = sig.idx
                entry_bar = sig_row
                entry = entry_bar["close"]
                entry_datetime = entry_bar["datetime"]
                sim_start = sig.idx + 1

            vwap = sig_row.get("vwap", entry)
            ema20 = sig_row.get("ema20", entry)
            or_high_r = sig_row.get("opening_range_high")
            or_low_r = sig_row.get("opening_range_low")

            # ── Stop calculation: avoid look-ahead bias ────────────────────────────────
            # When entering on next bar open, use signal bar's low/high for stop
            # (known at decision time), not entry bar's low/high (unknown at entry time)
            if sig.side == "LONG":
                stop_candidates = [vwap, sig_row["low"]]
                stop_candidates = [s for s in stop_candidates if s < entry]
                if not stop_candidates:
                    continue
                stop = max(stop_candidates)
                risk = entry - stop
                target = entry + cfg.risk_reward * risk
            else:
                stop_candidates = [vwap, sig_row["high"]]
                stop_candidates = [s for s in stop_candidates if s > entry]
                if not stop_candidates:
                    continue
                stop = min(stop_candidates)
                risk = stop - entry
                target = entry - cfg.risk_reward * risk
            if risk <= 0:
                continue
            planned_r = cfg.risk_reward

            # ── Sizing ───────────────────────────────────────────────────
            if cfg.sizing_method == "fixed_quantity" and cfg.fixed_quantity:
                qty = int(cfg.fixed_quantity)
            elif cfg.sizing_method == "risk_based" and cfg.max_risk_per_trade_inr:
                qty = int(cfg.max_risk_per_trade_inr // risk)
            else:
                qty = int(cfg.fixed_notional_inr // entry)
            if cfg.max_position_value_inr:
                qty = min(qty, int(cfg.max_position_value_inr // entry))
            if qty <= 0:
                continue

            exit_price = day_df["close"].iloc[-1]  # default: square-off at last bar
            exit_reason = "square_off"
            bars_held = 0
            breakeven_moved = False
            stop_price = stop
            exit_bar_idx = len(day_df) - 1

            # ── Forward simulation from entry bar ────────────────────────
            for j in range(sim_start, len(day_df)):
                bar = day_df.iloc[j]
                bars_held += 1
                cur_high = bar["high"]
                cur_low = bar["low"]
                cur_close = bar["close"]
                cur_time: time = bar["time"]

                # Breakeven after N×R
                if cfg.enable_breakeven and not breakeven_moved:
                    if sig.side == "LONG" and cur_high >= entry + risk * cfg.breakeven_after_r:
                        # compute costs on entry+entry (round-trip estimate at entry)
                        be_costs = 2 * entry * qty * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000
                        stop_price = entry if not cfg.cost_aware_breakeven else entry + be_costs / qty
                        breakeven_moved = True
                    if sig.side == "SHORT" and cur_low <= entry - risk * cfg.breakeven_after_r:
                        be_costs = 2 * entry * qty * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000
                        stop_price = entry if not cfg.cost_aware_breakeven else entry - be_costs / qty
                        breakeven_moved = True

                # EMA trailing stop after breakeven
                if cfg.enable_ema_trailing_stop and breakeven_moved:
                    ema = bar.get("ema20", cur_close)
                    if sig.side == "LONG" and ema < cur_close:
                        stop_price = max(stop_price, ema)
                    if sig.side == "SHORT" and ema > cur_close:
                        stop_price = min(stop_price, ema)

                # Exit priority: stop → target → square-off
                if sig.side == "LONG":
                    if _price_hits(stop_price, cur_high, cur_low):
                        exit_price = stop_price; exit_reason = "stop"; exit_bar_idx = j; break
                    if _price_hits(target, cur_high, cur_low):
                        exit_price = target; exit_reason = "target"; exit_bar_idx = j; break
                else:
                    if _price_hits(stop_price, cur_high, cur_low):
                        exit_price = stop_price; exit_reason = "stop"; exit_bar_idx = j; break
                    if _price_hits(target, cur_high, cur_low):
                        exit_price = target; exit_reason = "target"; exit_bar_idx = j; break
                if cur_time >= square_off:
                    exit_price = cur_close; exit_reason = "square_off"; exit_bar_idx = j; break

            # ── Costs on actual round-trip ───────────────────────────────
            costs = (entry + exit_price) * qty * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000

            realized_r = (exit_price - entry) / risk if sig.side == "LONG" else (entry - exit_price) / risk
            gross_pnl = (exit_price - entry) * qty if sig.side == "LONG" else (entry - exit_price) * qty
            net_pnl = gross_pnl - costs

            exit_dt = day_df.iloc[exit_bar_idx]["datetime"]

            notes = f"{sig.side} entry @ {entry:.2f}, signal: {sig.reason}, exit: {exit_reason}"
            trades.append(
                Trade(
                    symbol=sym,
                    trade_date=trade_date,
                    entry_datetime=entry_datetime,
                    exit_datetime=exit_dt,
                    side=sig.side,
                    entry_price=entry,
                    exit_price=exit_price,
                    quantity=qty,
                    opening_range_high=or_high_r,
                    opening_range_low=or_low_r,
                    entry_vwap=vwap,
                    entry_ema20=ema20,
                    breakout_volume=sig_row["volume"],
                    previous_5_candle_avg_volume=sig_row.get("prev5_avg_vol", float("nan")),
                    breakout_volume_ratio=sig_row.get("vol_ratio", float("nan")),
                    initial_stop=stop,
                    final_stop=stop_price,
                    target=target,
                    risk_per_share=risk,
                    planned_r_multiple=planned_r,
                    realized_r_multiple=realized_r,
                    gross_pnl=gross_pnl,
                    estimated_costs=costs,
                    net_pnl=net_pnl,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                    one_trade_per_day_flag=cfg.one_trade_per_day,
                    sector_filter_passed=True,
                    notes=notes,
                )
            )
            taken_dates.add(trade_date_str)

        prev_close[sym] = float(day_df["close"].iloc[-1])

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    daily_df = trades_df.copy().groupby(["trade_date", "symbol"], as_index=False).agg(
        trades=("side", "count"),
        winners=("net_pnl", lambda s: (s > 0).sum()),
        losers=("net_pnl", lambda s: (s < 0).sum()),
        gross_pnl=("gross_pnl", "sum"),
        costs=("estimated_costs", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_r=("realized_r_multiple", "mean"),
    ) if not trades_df.empty else pd.DataFrame(columns=["trade_date", "symbol", "trades", "winners", "losers", "gross_pnl", "costs", "net_pnl", "avg_r"])
    if not daily_df.empty:
        daily_df["cumulative_net_pnl"] = daily_df.groupby("symbol")["net_pnl"].cumsum()

    # simple drawdown from daily net pnl
    if not daily_df.empty:
        daily_df["equity"] = daily_df.groupby("symbol")["net_pnl"].cumsum()
        daily_df["peak"] = daily_df.groupby("symbol")["equity"].cummax()
        daily_df["drawdown"] = daily_df["equity"] - daily_df["peak"]
        max_drawdown = float(daily_df["drawdown"].min())
    else:
        max_drawdown = 0.0

    summary = {
        "total_trades": int(len(trades_df)),
        "win_rate": float((trades_df["net_pnl"] > 0).mean()) if not trades_df.empty else 0.0,
        "gross_pnl": float(trades_df.get("gross_pnl", pd.Series(dtype=float)).sum()) if not trades_df.empty else 0.0,
        "net_pnl": float(trades_df.get("net_pnl", pd.Series(dtype=float)).sum()) if not trades_df.empty else 0.0,
        "avg_r": float(trades_df.get("realized_r_multiple", pd.Series(dtype=float)).mean()) if not trades_df.empty else 0.0,
        "profit_factor": float(abs(trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()) / abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum())) if not trades_df.empty and (trades_df["net_pnl"] < 0).any() else None,
        "max_drawdown": max_drawdown,
    }
    return trades_df, daily_df, summary, stats_by_date

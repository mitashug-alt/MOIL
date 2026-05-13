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
    for trade_date, day_df in df.groupby(df["datetime"].dt.date):
        day_df = day_df.reset_index(drop=True)
        or_high = day_df["opening_range_high"].iloc[0]
        or_low = day_df["opening_range_low"].iloc[0]
        sigs, stats = generate_signals_with_stats(day_df, cfg, or_high, or_low)
        stats_by_date[str(trade_date)] = stats | {
            "orb_high": or_high,
            "orb_low": or_low,
            "orb_range_pct": (or_high - or_low) / or_low * 100 if pd.notna(or_high) and pd.notna(or_low) and or_low else None,
        }
        for sig in sigs:
            trade_date_key = str(trade_date)
            if cfg.one_trade_per_day and trade_date_key in taken_dates:
                continue
            row = day_df.iloc[sig.idx]
            trade_date_str = str(row["datetime"].date())
            t: time = row["time"]
            if not _within(t, entry_start, entry_end):
                continue
            entry = row["close"]
            vwap = row.get("vwap", entry)
            ema20 = row.get("ema20", entry)
            or_high = row.get("opening_range_high")
            or_low = row.get("opening_range_low")
            if sig.side == "LONG":
                stop_candidates = [vwap, row["low"]]
                stop_candidates = [s for s in stop_candidates if s < entry]
                if not stop_candidates:
                    continue
                stop = max(stop_candidates)
                risk = entry - stop
                target = entry + cfg.risk_reward * risk
            else:
                stop_candidates = [vwap, row["high"]]
                stop_candidates = [s for s in stop_candidates if s > entry]
                if not stop_candidates:
                    continue
                stop = min(stop_candidates)
                risk = stop - entry
                target = entry - cfg.risk_reward * risk
            if risk <= 0:
                continue
            planned_r = cfg.risk_reward

            # sizing
            if cfg.sizing_method == "fixed_quantity" and cfg.fixed_quantity:
                qty = int(cfg.fixed_quantity)
            elif cfg.sizing_method == "risk_based" and cfg.max_risk_per_trade_inr:
                qty = int(cfg.max_risk_per_trade_inr // risk)
            else:
                # fixed notional
                qty = int(cfg.fixed_notional_inr // entry)
            if cfg.max_position_value_inr:
                qty = min(qty, int(cfg.max_position_value_inr // entry))
            if qty <= 0:
                continue

            costs = (entry + target) * qty * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000

            exit_price = target
            exit_reason = "target"
            bars_held = 0
            breakeven_moved = False
            stop_price = stop

            # simulate forward bars until square-off
            for j in range(sig.idx + 1, len(day_df)):
                bar = day_df.iloc[j]
                bars_held += 1
                cur_high = bar["high"]
                cur_low = bar["low"]
                cur_close = bar["close"]
                cur_time: time = bar["time"]

                # Breakeven after 1R
                if cfg.enable_breakeven and not breakeven_moved:
                    if sig.side == "LONG" and cur_high >= entry + risk * cfg.breakeven_after_r:
                        stop_price = entry if not cfg.cost_aware_breakeven else entry + costs / qty
                        breakeven_moved = True
                    if sig.side == "SHORT" and cur_low <= entry - risk * cfg.breakeven_after_r:
                        stop_price = entry if not cfg.cost_aware_breakeven else entry - costs / qty
                        breakeven_moved = True

                # Trailing with EMA20 after 1R
                if cfg.enable_ema_trailing_stop and breakeven_moved:
                    ema = bar.get("ema20", cur_close)
                    if sig.side == "LONG" and ema < cur_close:
                        stop_price = max(stop_price, ema)
                    if sig.side == "SHORT" and ema > cur_close:
                        stop_price = min(stop_price, ema)

                # check exits priority: stop -> target -> trailing (already applied) -> square off
                if sig.side == "LONG":
                    if _price_hits(stop_price, cur_high, cur_low):
                        exit_price = stop_price
                        exit_reason = "stop"
                        break
                    if _price_hits(target, cur_high, cur_low):
                        exit_price = target
                        exit_reason = "target"
                        break
                else:
                    if _price_hits(stop_price, cur_high, cur_low):
                        exit_price = stop_price
                        exit_reason = "stop"
                        break
                    if _price_hits(target, cur_high, cur_low):
                        exit_price = target
                        exit_reason = "target"
                        break
                if cur_time >= square_off:
                    exit_price = cur_close
                    exit_reason = "square_off"
                    break

            realized_r = (exit_price - entry) / risk if sig.side == "LONG" else (entry - exit_price) / risk
            gross_pnl = (exit_price - entry) * qty if sig.side == "LONG" else (entry - exit_price) * qty
            net_pnl = gross_pnl - costs

            trades.append(
                Trade(
                    symbol=row["symbol"],
                    trade_date=trade_date,
                    entry_datetime=row["datetime"],
                    exit_datetime=day_df.iloc[sig.idx + bars_held]["datetime"] if sig.idx + bars_held < len(day_df) else row["datetime"],
                    side=sig.side,
                    entry_price=entry,
                    exit_price=exit_price,
                    quantity=qty,
                    opening_range_high=or_high,
                    opening_range_low=or_low,
                    entry_vwap=vwap,
                    entry_ema20=ema20,
                    breakout_volume=row["volume"],
                    previous_5_candle_avg_volume=row.get("prev5_avg_vol", float("nan")),
                    breakout_volume_ratio=row.get("vol_ratio", float("nan")),
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
                )
            )
            taken_dates.add(trade_date_str)

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

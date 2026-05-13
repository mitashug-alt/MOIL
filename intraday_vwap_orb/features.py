from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "date",
    "symbol",
    "orb_high",
    "orb_low",
    "orb_range_pct",
    "first_breakout_side",
    "first_breakout_time",
    "first_breakout_volume_ratio",
    "vwap_breakout_signal",
    "valid_long_signal_count",
    "valid_short_signal_count",
    "rejected_low_volume_signal_count",
    "rejected_below_vwap_long_count",
    "rejected_above_vwap_short_count",
    "strategy_trade_taken",
    "strategy_side",
    "strategy_net_pnl",
    "strategy_realized_r",
    "strategy_exit_reason",
    "close_vs_vwap_pct_at_1100",
]


def daily_features(trades: pd.DataFrame, stats_by_date: dict[str, dict], prepared_df: pd.DataFrame) -> pd.DataFrame:
    # ensure one row per symbol/date even if no trades
    rows = []
    grouped = prepared_df.groupby(prepared_df["datetime"].dt.date)
    for date, day_df in grouped:
        sym = day_df["symbol"].iloc[0]
        stats = stats_by_date.get(str(date), {})
        if trades is None or trades.empty or "trade_date" not in trades.columns:
            trade_rows = pd.DataFrame()
        else:
            trade_rows = trades[trades["trade_date"] == str(date)]
        strategy_trade_taken = not trade_rows.empty
        strategy_side = trade_rows.iloc[0]["side"] if strategy_trade_taken else None
        strategy_net_pnl = float(trade_rows["net_pnl"].sum()) if strategy_trade_taken else 0.0
        strategy_realized_r = float(trade_rows["realized_r_multiple"].mean()) if strategy_trade_taken else 0.0
        strategy_exit_reason = trade_rows.iloc[0]["exit_reason"] if strategy_trade_taken else None
        vwap_1100 = day_df[day_df["time"] <= pd.to_datetime("11:00").time()]
        last_1100 = vwap_1100.iloc[-1] if not vwap_1100.empty else None
        close_vs_vwap_pct_at_1100 = None
        if last_1100 is not None and last_1100.get("vwap"):
            close_vs_vwap_pct_at_1100 = (last_1100["close"] - last_1100["vwap"]) / last_1100["vwap"] * 100
        rows.append({
            "date": pd.to_datetime(date),
            "symbol": sym,
            "orb_high": stats.get("orb_high"),
            "orb_low": stats.get("orb_low"),
            "orb_range_pct": stats.get("orb_range_pct"),
            "first_breakout_side": stats.get("first_breakout_side"),
            "first_breakout_time": stats.get("first_breakout_time"),
            "first_breakout_volume_ratio": stats.get("first_breakout_volume_ratio"),
            "vwap_breakout_signal": stats.get("vwap_breakout_signal", False),
            "valid_long_signal_count": stats.get("valid_long_signal_count", 0),
            "valid_short_signal_count": stats.get("valid_short_signal_count", 0),
            "rejected_low_volume_signal_count": stats.get("rejected_low_volume_signal_count", 0),
            "rejected_below_vwap_long_count": stats.get("rejected_below_vwap_long_count", 0),
            "rejected_above_vwap_short_count": stats.get("rejected_above_vwap_short_count", 0),
            "strategy_trade_taken": strategy_trade_taken,
            "strategy_side": strategy_side,
            "strategy_net_pnl": strategy_net_pnl,
            "strategy_realized_r": strategy_realized_r,
            "strategy_exit_reason": strategy_exit_reason,
            "close_vs_vwap_pct_at_1100": close_vs_vwap_pct_at_1100,
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def merge_with_institutional(intraday_features: pd.DataFrame, institutional_features: pd.DataFrame) -> pd.DataFrame:
    if intraday_features is None or intraday_features.empty:
        return institutional_features
    if institutional_features is None or institutional_features.empty:
        return intraday_features
    inst = institutional_features.copy()
    if "date" in inst.columns:
        inst["date"] = pd.to_datetime(inst["date"])
    intr = intraday_features.copy()
    intr["date"] = pd.to_datetime(intr["date"])
    merged = pd.merge(inst, intr, on=["date", "symbol"], how="outer", suffixes=("_inst", "_intraday"))
    return merged

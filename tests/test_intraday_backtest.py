import pandas as pd
from intraday_vwap_orb.config import StrategyConfig
from intraday_vwap_orb.data import prepare_intraday
from intraday_vwap_orb.backtest import run_backtest


def make_day():
    # simple day where breakout triggers and hits target
    rows = [
        # OR window
        {"datetime": "2026-05-12 09:15", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:20", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1100, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:25", "open": 102, "high": 103, "low": 101, "close": 102.5, "volume": 1200, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:30", "open": 102.5, "high": 103, "low": 102, "close": 102.8, "volume": 1200, "symbol": "MOIL"},
        # breakout candle 09:35 with big volume
        {"datetime": "2026-05-12 09:35", "open": 103, "high": 110, "low": 102.5, "close": 109, "volume": 3000, "symbol": "MOIL"},
        # follow-through
        {"datetime": "2026-05-12 09:40", "open": 109, "high": 113, "low": 108, "close": 112, "volume": 1500, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:45", "open": 112, "high": 115, "low": 111, "close": 114, "volume": 1500, "symbol": "MOIL"},
    ]
    return pd.DataFrame(rows)


def test_long_breakout_hits_target():
    cfg = StrategyConfig(volume_multiplier=1.5)
    df = prepare_intraday(make_day(), cfg)
    trades, daily, summary, stats = run_backtest(df, cfg)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert summary["total_trades"] == 1


def test_stop_chosen_below_entry():
    cfg = StrategyConfig(volume_multiplier=1.5)
    df = prepare_intraday(make_day(), cfg)
    trades, daily, summary, stats = run_backtest(df, cfg)
    t = trades.iloc[0]
    assert t["initial_stop"] < t["entry_price"]


def test_one_trade_per_day():
    cfg = StrategyConfig(one_trade_per_day=True, volume_multiplier=1.0)
    day1 = make_day()
    day2 = make_day()
    day2["datetime"] = pd.to_datetime(day2["datetime"]) + pd.Timedelta(days=1)
    df = prepare_intraday(pd.concat([day1, day2], ignore_index=True), cfg)
    trades, daily, summary, stats = run_backtest(df, cfg)
    # two days, one trade each
    assert len(trades) == 2

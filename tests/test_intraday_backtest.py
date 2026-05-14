import pandas as pd
from intraday_vwap_orb.config import StrategyConfig
from intraday_vwap_orb.data import prepare_intraday
from intraday_vwap_orb.backtest import run_backtest


def make_day():
    # simple day where breakout triggers and hits target
    # with correct no-look-ahead stop placement (using signal bar low)
    rows = [
        # OR window
        {"datetime": "2026-05-12 09:15", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:20", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1100, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:25", "open": 102, "high": 103, "low": 101, "close": 102.5, "volume": 1200, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:30", "open": 102.5, "high": 103, "low": 102, "close": 102.8, "volume": 1200, "symbol": "MOIL"},
        # breakout candle 09:35 with big volume (signal bar, low=102.5 used for stop)
        {"datetime": "2026-05-12 09:35", "open": 103, "high": 110, "low": 102.5, "close": 109, "volume": 3000, "symbol": "MOIL"},
        # entry at 09:40 open=109, stop=max(vwap,102.5) ~102.5, risk=6.5, target=109+13=122
        {"datetime": "2026-05-12 09:40", "open": 109, "high": 118, "low": 108, "close": 117, "volume": 1500, "symbol": "MOIL"},
        # hits target 122
        {"datetime": "2026-05-12 09:45", "open": 117, "high": 125, "low": 116, "close": 124, "volume": 1500, "symbol": "MOIL"},
    ]
    return pd.DataFrame(rows)


_TEST_CFG_KWARGS = dict(volume_multiplier=1.5, min_or_range_pct=0.0, max_gap_pct=0.0)


def test_long_breakout_hits_target():
    cfg = StrategyConfig(**_TEST_CFG_KWARGS)
    df = prepare_intraday(make_day(), cfg)
    trades, daily, summary, stats = run_backtest(df, cfg)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert summary["total_trades"] == 1


def test_stop_chosen_below_entry():
    cfg = StrategyConfig(**_TEST_CFG_KWARGS)
    df = prepare_intraday(make_day(), cfg)
    trades, daily, summary, stats = run_backtest(df, cfg)
    t = trades.iloc[0]
    assert t["initial_stop"] < t["entry_price"]


def test_one_trade_per_day():
    cfg = StrategyConfig(one_trade_per_day=True, volume_multiplier=1.0, min_or_range_pct=0.0, max_gap_pct=0.0)
    day1 = make_day()
    day2 = make_day()
    day2["datetime"] = pd.to_datetime(day2["datetime"]) + pd.Timedelta(days=1)
    df = prepare_intraday(pd.concat([day1, day2], ignore_index=True), cfg)
    trades, daily, summary, stats = run_backtest(df, cfg)
    # two days, one trade each
    assert len(trades) == 2


def test_missing_vwap_column_uses_close_fallback():
    """Regression test: vwap column missing should not raise AttributeError."""
    from intraday_vwap_orb.signals import add_indicators, generate_signals_with_stats
    rows = [
        {"datetime": "2026-05-12 09:15", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "symbol": "MOIL", "ema20": 100},
        {"datetime": "2026-05-12 09:20", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1100, "symbol": "MOIL", "ema20": 101},
    ]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    # Add time column via add_indicators (which now validates vwap exists)
    cfg = StrategyConfig(volume_multiplier=1.0, min_or_range_pct=0.0, max_gap_pct=0.0)
    # add_indicators requires vwap column, so add a dummy one
    df["vwap"] = df["close"]  # dummy vwap
    df = add_indicators(df, cfg)
    # Now remove vwap to test the getattr fallback in signal generation
    df_no_vwap = df.drop(columns=["vwap"])
    or_high, or_low = 103, 99
    signals, stats = generate_signals_with_stats(df_no_vwap, cfg, or_high, or_low)
    # Should complete without AttributeError even without vwap column
    assert isinstance(stats, dict)


def test_or_range_skip_populates_all_stats_keys():
    """Regression test: skipped days must have consistent stats schema."""
    cfg = StrategyConfig(volume_multiplier=1.0, min_or_range_pct=5.0, max_gap_pct=0.0)  # High min OR range to trigger skip
    rows = [
        # OR window - tiny range to trigger skip
        {"datetime": "2026-05-12 09:15", "open": 100, "high": 100.1, "low": 100, "close": 100.05, "volume": 1000, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:20", "open": 100.05, "high": 100.1, "low": 100, "close": 100.08, "volume": 1000, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:25", "open": 100.08, "high": 100.1, "low": 100, "close": 100.09, "volume": 1000, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:30", "open": 100.09, "high": 100.1, "low": 100, "close": 100.1, "volume": 1000, "symbol": "MOIL"},
        {"datetime": "2026-05-12 09:35", "open": 100.1, "high": 100.2, "low": 100, "close": 100.15, "volume": 1000, "symbol": "MOIL"},
    ]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    from intraday_vwap_orb.data import prepare_intraday
    df = prepare_intraday(df, cfg)
    trades, daily, summary, stats_by_date = run_backtest(df, cfg)
    
    date_key = "2026-05-12"
    assert date_key in stats_by_date
    day_stats = stats_by_date[date_key]
    assert day_stats["skipped"] is True
    # All signal-related keys must exist (regression: these were missing before)
    expected_keys = [
        "gap_pct", "valid_long_signal_count", "valid_short_signal_count",
        "rejected_low_volume_signal_count", "rejected_below_vwap_long_count",
        "rejected_above_vwap_short_count", "first_breakout_side",
        "first_breakout_time", "first_breakout_volume_ratio", "vwap_breakout_signal"
    ]
    for key in expected_keys:
        assert key in day_stats, f"Missing key: {key}"


def test_first_day_gap_pct_is_none():
    """Regression test: first trading day should have gap_pct=None (not 0)."""
    cfg = StrategyConfig(volume_multiplier=1.0, min_or_range_pct=0.0, max_gap_pct=0.0)
    # Single day - no prior close available
    day = make_day()
    df = prepare_intraday(day, cfg)
    trades, daily, summary, stats_by_date = run_backtest(df, cfg)
    
    date_key = "2026-05-12"
    assert date_key in stats_by_date
    assert stats_by_date[date_key]["gap_pct"] is None


def test_trade_notes_populated():
    """Regression test: Trade.notes field should be populated with context."""
    cfg = StrategyConfig(**_TEST_CFG_KWARGS)
    df = prepare_intraday(make_day(), cfg)
    trades, daily, summary, stats = run_backtest(df, cfg)
    assert len(trades) == 1
    t = trades.iloc[0]
    # notes should contain entry price, signal info, exit reason
    assert "notes" in t
    assert len(t["notes"]) > 0
    assert "entry @" in t["notes"]
    assert "exit:" in t["notes"]

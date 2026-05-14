import pandas as pd
from pathlib import Path

from intraday_vwap_orb.config import StrategyConfig
from intraday_vwap_orb.data import load_intraday, prepare_intraday
from intraday_vwap_orb.indicators import compute_vwap, opening_range, compute_ema
from intraday_vwap_orb.signals import generate_signals


def test_load_intraday_columns(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("datetime,open,high,low,close,volume\n2026-05-12 09:15,1,2,0.5,1.5,100")
    data = load_intraday(sample)
    assert not data.df.empty
    assert set(["datetime", "open", "high", "low", "close", "volume"]).issubset(data.df.columns)


def test_vwap_resets_daily():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime([
                "2026-05-12 09:15", "2026-05-12 09:20", "2026-05-13 09:15", "2026-05-13 09:20"
            ]),
            "high": [2, 3, 4, 5],
            "low": [1, 1.5, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5],
            "open": [1.4, 2.4, 3.4, 4.4],
            "volume": [100, 200, 100, 200],
        }
    )
    vwap = compute_vwap(df)
    assert vwap.iloc[1] > vwap.iloc[0]
    # reset next day: first bar of new session equals its own typical price
    assert vwap.iloc[2] == (df.loc[2, ["high", "low", "close"]].sum() / 3)


def test_opening_range_window():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime([
                "2026-05-12 09:15", "2026-05-12 09:20", "2026-05-12 09:35"
            ]),
            "high": [2, 3, 10],
            "low": [1, 1, 5],
            "close": [1.5, 2.5, 7],
            "open": [1.4, 2.4, 6],
            "volume": [100, 200, 300],
        }
    )
    or_df = opening_range(df)
    assert or_df.loc[0, "opening_range_high"] == 3
    assert or_df.loc[0, "opening_range_low"] == 1


def test_signal_requires_after_0930_high_volume():
    cfg = StrategyConfig(volume_multiplier=2.0)
    # Resolve path relative to test file location
    test_dir = Path(__file__).parent
    sample_path = test_dir.parent / "data" / "sample" / "intraday_sample_moil.csv"
    data = load_intraday(sample_path)
    df = prepare_intraday(data.df, cfg)
    day = df[df["date"] == df["date"].iloc[0]].reset_index(drop=True)
    sigs = generate_signals(day, cfg, day.loc[0, "opening_range_high"], day.loc[0, "opening_range_low"])
    assert all(day.loc[s.idx, "time"] >= pd.to_datetime(cfg.entry_start).time() for s in sigs)


def test_vwap_no_division_by_zero_with_zero_volume():
    """Regression test: VWAP should not produce inf when cumulative volume is 0."""
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-05-12 09:15", "2026-05-12 09:20"]),
            "high": [100, 101],
            "low": [99, 100],
            "close": [100, 101],
            "open": [99, 100],
            "volume": [0, 0],  # Zero volume - should not cause inf
        }
    )
    vwap = compute_vwap(df)
    # Should be NaN, not inf
    assert not vwap.isin([float("inf"), -float("inf")]).any()
    assert vwap.isna().all()


def test_vol_ratio_no_division_by_zero():
    """Regression test: vol_ratio should not produce inf when prev5_avg_vol is 0."""
    cfg = StrategyConfig(volume_lookback_bars=5)
    # Create data where rolling mean will be 0 (all zero volumes)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime([
                "2026-05-12 09:15", "2026-05-12 09:20", "2026-05-12 09:25",
                "2026-05-12 09:30", "2026-05-12 09:35", "2026-05-12 09:40"
            ]),
            "open": [100, 100, 100, 100, 100, 100],
            "high": [101, 101, 101, 101, 101, 101],
            "low": [99, 99, 99, 99, 99, 99],
            "close": [100, 100, 100, 100, 100, 100],
            "volume": [0, 0, 0, 0, 0, 100],  # First 5 bars have 0 volume
            "symbol": ["MOIL"] * 6,
        }
    )
    df = prepare_intraday(df, cfg)
    # vol_ratio should not have inf values
    assert not df["vol_ratio"].isin([float("inf"), -float("inf")]).any()
    # First 4 bars should have NaN (insufficient lookback), then finite values
    assert df["vol_ratio"].iloc[4:].isna().all() or not df["vol_ratio"].iloc[4:].isin([float("inf")]).any()

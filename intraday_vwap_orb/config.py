from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    timeframe: str = "5min"
    market_open: str = "09:15"
    opening_range_start: str = "09:15"
    opening_range_end: str = "09:30"
    entry_start: str = "09:30"
    entry_end: str = "11:00"
    square_off_time: str = "15:00"
    volume_lookback_bars: int = 5
    volume_multiplier: float = 2.0
    ema_period: int = 20
    vwap_price_source: str = "hlc3"  # hlc3 | close | ohlc4
    risk_reward: float = 2.0
    allow_long: bool = True
    allow_short: bool = True
    require_ema_filter: bool = False
    use_sector_filter: bool = False
    one_trade_per_day: bool = True
    enable_breakeven: bool = True
    breakeven_after_r: float = 1.0
    cost_aware_breakeven: bool = False
    enable_ema_trailing_stop: bool = True
    conservative_same_bar_exit: bool = True
    entry_on_next_bar_open: bool = True       # enter at next bar open, not signal bar close
    min_or_range_pct: float = 0.3             # skip days where OR range < this % of OR low
    max_gap_pct: float = 2.0                  # skip days where gap vs prior close > this %
    slippage_bps: float = 5.0
    transaction_cost_bps: float = 2.0
    sizing_method: str = "fixed_notional"  # fixed_quantity | fixed_notional | risk_based
    fixed_quantity: int | None = None
    fixed_notional_inr: float = 200_000.0
    max_risk_per_trade_inr: float = 50_000.0
    max_position_value_inr: float | None = None
    price_col: str = "close"


DEFAULT_CONFIG = StrategyConfig()

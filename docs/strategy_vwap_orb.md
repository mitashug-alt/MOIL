---
description: Intraday VWAP + Opening Range Breakout research module
---

## Overview
Research-only VWAP + 15-minute Opening Range Breakout strategy on 5-minute NSE equity candles. No live trading or execution.

## Defaults
- Timeframe: 5m
- Market open: 09:15; OR window: 09:15–09:30; entries 09:30–11:00; square-off 15:00
- VWAP price source: hlc3; EMA20 optional filter/trailing
- Volume filter: current volume >= 2x average of previous 5 bars
- Risk: 1:2 RR by default; one trade per symbol per day by default

## Inputs
CSV/Parquet with columns: `datetime, open, high, low, close, volume` (+ optional `symbol`). Loader validates and resamples 1m->5m if needed.

## Key Rules
- Long: close > OR high and > VWAP, volume ratio >= threshold, optional EMA20/sector filter
- Short: close < OR low and < VWAP, volume ratio >= threshold, optional EMA20/sector filter
- Stops: pick closest valid stop (VWAP or breakout bar extreme) beyond entry; skip if invalid
- Target: RR * risk; breakeven after 1R (optional cost-aware); optional EMA20 trailing after breakeven
- Exit priority: stop → target → trailing → force square-off (15:00); conservative same-bar stop-first
- Sizing: fixed quantity | fixed notional | risk-based (cap optional)
- Costs: simple bps slippage + transaction cost placeholder

## CLI
```bash
# Backtest
a) python -m intraday_vwap_orb.cli intraday-backtest \
     --symbol MOIL \
     --input data/sample/intraday_sample_moil.csv \
     --volume-multiplier 2.0 \
     --risk-reward 2.0 \
     --output-dir reports/intraday

# Feature export (daily strategy features)
b) python -m intraday_vwap_orb.cli intraday-features \
     --symbol MOIL \
     --input data/sample/intraday_sample_moil.csv \
     --output data/processed/intraday/MOIL_vwap_orb_features.csv

# Report from existing CSVs
c) python -m intraday_vwap_orb.cli intraday-report \
     --symbol MOIL \
     --trades reports/intraday/MOIL_trades.csv \
     --daily reports/intraday/MOIL_daily_pnl.csv \
     --output reports/intraday/MOIL_summary.md
```
Outputs include trades, daily P&L, features, and a markdown summary.

## Integration with institutional features
Use `intraday_vwap_orb.features.merge_with_institutional(intraday_features, institutional_features)` to produce a combined daily dataset (suggested path: `data/processed/MOIL_combined_daily_features.csv`).

## Caveats
- Research-only; no live orders.
- Requires complete 09:15–09:30 window per day; incomplete days are included but OR values may be NaN.
- Data quality: missing volume/timestamps will raise validation errors; duplicated timestamps are dropped.
- Costs/slippage are simplified placeholders; configure bps inputs appropriately.

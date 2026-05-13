from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dataclasses import replace

from .backtest import run_backtest
from .config import DEFAULT_CONFIG, StrategyConfig
from .data import load_intraday, prepare_intraday
from .features import daily_features, merge_with_institutional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday VWAP + ORB backtest")
    sub = p.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("intraday-backtest")
    bt.add_argument("--symbol", default="MOIL")
    bt.add_argument("--input", required=True)
    bt.add_argument("--from", dest="from_date", default=None)
    bt.add_argument("--to", dest="to_date", default=None)
    bt.add_argument("--volume-multiplier", type=float, default=2.0)
    bt.add_argument("--risk-reward", type=float, default=2.0)
    bt.add_argument("--output-dir", default="reports/intraday")

    feats = sub.add_parser("intraday-features")
    feats.add_argument("--symbol", default="MOIL")
    feats.add_argument("--input", required=True)
    feats.add_argument("--from", dest="from_date", default=None)
    feats.add_argument("--to", dest="to_date", default=None)
    feats.add_argument("--output", default="data/processed/intraday_features.csv")

    rep = sub.add_parser("intraday-report")
    rep.add_argument("--symbol", default="MOIL")
    rep.add_argument("--trades", required=True)
    rep.add_argument("--daily", required=True)
    rep.add_argument("--output", default="reports/intraday/summary.md")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(DEFAULT_CONFIG)
    if args.command == "intraday-backtest":
        cfg.volume_multiplier = args.volume_multiplier
        cfg.risk_reward = args.risk_reward
        intraday = load_intraday(args.input)
        df = prepare_intraday(intraday.df, cfg, args.from_date, args.to_date)
        trades, daily, summary, stats = run_backtest(df, cfg)
        feats = daily_features(trades, stats, df)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_dir / f"{args.symbol}_trades.csv", index=False)
        daily.to_csv(out_dir / f"{args.symbol}_daily_pnl.csv", index=False)
        feats.to_csv(out_dir / f"{args.symbol}_features.csv", index=False)
        # summary markdown
        summary_md = out_dir / f"{args.symbol}_summary.md"
        summary_md.write_text(
            "\n".join([
                f"# Intraday VWAP+ORB summary for {args.symbol}",
                f"Total trades: {summary['total_trades']}",
                f"Win rate: {summary['win_rate']:.2%}",
                f"Net PnL: {summary['net_pnl']:.2f}",
                f"Avg R: {summary['avg_r']:.2f}",
                f"Profit factor: {summary['profit_factor'] if summary['profit_factor'] is not None else 'n/a'}",
                f"Max drawdown: {summary['max_drawdown']:.2f}",
            ])
        )
        print(f"Wrote trades to {out_dir}/{args.symbol}_trades.csv, daily PnL to {out_dir}/{args.symbol}_daily_pnl.csv, features to {out_dir}/{args.symbol}_features.csv, summary to {summary_md}")
    elif args.command == "intraday-features":
        cfg = replace(DEFAULT_CONFIG)
        intraday = load_intraday(args.input)
        df = prepare_intraday(intraday.df, cfg, args.from_date, args.to_date)
        trades, daily, summary, stats = run_backtest(df, cfg)
        feats = daily_features(trades, stats, df)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(args.output, index=False)
        print(f"Wrote features to {args.output}")
    elif args.command == "intraday-report":
        trades = pd.read_csv(args.trades)
        daily = pd.read_csv(args.daily)
        profit_factor = abs(trades[trades["net_pnl"] > 0]["net_pnl"].sum()) / abs(trades[trades["net_pnl"] < 0]["net_pnl"].sum()) if (trades["net_pnl"] < 0).any() else None
        md = [
            f"# Intraday VWAP+ORB report for {args.symbol}",
            f"Total trades: {len(trades)}",
            f"Win rate: {(trades['net_pnl']>0).mean():.2% if len(trades)>0 else 0}",
            f"Net PnL: {trades['net_pnl'].sum():.2f}",
            f"Profit factor: {profit_factor if profit_factor is not None else 'n/a'}",
        ]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(md))
        print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()

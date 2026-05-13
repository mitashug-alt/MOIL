from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .bse_client import filter_symbol as bse_filter_symbol, load_bse_bulk_block_csv
from .config import tracker_config
from .disclosures import classify_trigger, load_disclosure_csv
from .features import combine_features
from .mf_holdings import load_mf_manual_file
from .moil_shp import compare_quarters, load_manual_shp_csv, parse_supplemental_csv, save_template_csv
from .nse_client import NSEClient, iso_today
from .report import build_summary, write_features, write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOIL institutional flow tracker (offline-friendly)")
    p.add_argument("command", choices=["thresholds", "fetch-deals", "fetch-shareholding", "fetch-mf", "build-features", "report"], help="Command to run")
    p.add_argument("--symbol", default="MOIL")
    p.add_argument("--from", dest="from_date", default=None, help="From date YYYY-MM-DD for deals")
    p.add_argument("--to", dest="to_date", default=None, help="To date YYYY-MM-DD for deals")
    p.add_argument("--mf-file", dest="mf_file", default=None, help="Manual MF CSV/XLSX path")
    p.add_argument("--deals-file", dest="deals_file", default=None, help="Manual deals CSV path (bulk/block)")
    p.add_argument("--bse-file", dest="bse_file", default=None, help="Manual BSE bulk/block CSV path")
    p.add_argument("--shp-file", dest="shp_file", default=None, help="Manual SHP CSV path")
    p.add_argument("--disclosure-file", dest="disclosure_file", default=None, help="Manual disclosures CSV path")
    p.add_argument("--out", dest="out_dir", default="reports", help="Output directory")
    p.add_argument("--disable-network", action="store_true", help="Force offline mode (no NSE GET)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = tracker_config({"symbol": args.symbol, "enable_network": not args.disable_network})
    cmd = args.command

    if cmd == "thresholds":
        print(f"Bulk threshold (0.5%): {cfg.bulk_threshold_shares} shares")
        print(f"Named holder (1%): {cfg.named_holder_threshold_shares} shares")
        print(f"SAST 5%: {cfg.sast_five_percent_shares} shares")
        print(f"SAST 2% change: {cfg.sast_two_percent_shares} shares")
        return

    deals = pd.DataFrame()
    shp = pd.DataFrame()
    mf = pd.DataFrame()
    disclosures = pd.DataFrame()

    if cmd in {"fetch-deals", "build-features", "report"}:
        client = NSEClient.create(cfg)
        from_date = args.from_date or iso_today()
        to_date = args.to_date or iso_today()
        bulk, diag1 = client.fetch_bulk_deals(from_date, to_date)
        block, diag2 = client.fetch_block_deals(from_date, to_date)
        deals = pd.concat([bulk, block], ignore_index=True) if not bulk.empty or not block.empty else pd.DataFrame()
        if args.deals_file:
            try:
                manual_deals = pd.read_csv(Path(args.deals_file))
                deals = pd.concat([deals, manual_deals], ignore_index=True) if not manual_deals.empty else deals
            except Exception as exc:  # noqa: BLE001
                if cfg.verbose:
                    print(f"Failed to load manual deals file {args.deals_file}: {exc}")
        if args.bse_file:
            bse_df = bse_filter_symbol(load_bse_bulk_block_csv(args.bse_file), cfg.symbol)
            deals = pd.concat([deals, bse_df], ignore_index=True) if not bse_df.empty else deals
        if cfg.verbose:
            print("Diagnostics:", diag1 + diag2)

    if cmd in {"fetch-shareholding", "build-features", "report"}:
        if args.shp_file:
            shp = load_manual_shp_csv(Path(args.shp_file))
        else:
            template_path = Path(cfg.raw_dir) / "manual_shp_template.csv"
            save_template_csv(template_path)
            shp = load_manual_shp_csv(template_path)

    if cmd in {"fetch-mf", "build-features", "report"} and args.mf_file:
        mf = load_mf_manual_file(Path(args.mf_file))

    if cmd in {"build-features", "report"} and args.disclosure_file:
        disclosures = load_disclosure_csv(Path(args.disclosure_file), cfg.symbol)
        disclosures["trigger_type"] = disclosures.apply(lambda r: classify_trigger(r, cfg.outstanding_shares), axis=1)

    if cmd == "fetch-deals":
        print(deals.head())
        return
    if cmd == "fetch-shareholding":
        print(shp.head())
        return
    if cmd == "fetch-mf":
        print(mf.head())
        return

    features_df = combine_features(deals, shp, mf, disclosures, cfg)

    if cmd == "build-features":
        out_dir = Path(args.out_dir)
        write_features(features_df, out_dir / "moil_institutional_features.csv")
        print(features_df.head())
        return

    if cmd == "report":
        report_md = build_summary(cfg, deals, shp, mf, disclosures)
        out_dir = Path(args.out_dir)
        write_report(report_md, out_dir / "moil_institutional_flow_summary.md")
        write_features(features_df, out_dir / "moil_institutional_features.csv", out_dir / "moil_institutional_features.parquet")
        print(report_md)
        return


if __name__ == "__main__":
    main()

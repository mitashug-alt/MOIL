from __future__ import annotations

import pandas as pd
from typing import Dict

from .config import TrackerConfig


def deal_features(deals: pd.DataFrame, cfg: TrackerConfig) -> pd.DataFrame:
    if deals is None or deals.empty:
        return pd.DataFrame(columns=[
            "date", "symbol", "bulk_buy_qty", "bulk_sell_qty", "bulk_net_qty", "bulk_buy_value", "bulk_sell_value", "bulk_net_value",
            "block_buy_qty", "block_sell_qty", "block_net_qty", "block_buy_value", "block_sell_value", "block_net_value",
            "large_deal_flag", "moil_bulk_threshold_hit", "moil_block_deal_flag", "repeat_institution_5d", "repeat_institution_20d",
        ])
    df = deals.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["deal_type"] = df.get("deal_type", "").str.lower()
    df["buy_sell"] = df.get("buy_sell", "").str.upper()
    df["value"] = df.get("quantity", 0) * df.get("price", 0)
    grouped = []
    for dt, grp in df.groupby(df["date"].dt.date):
        bulk = grp[grp["deal_type"].str.contains("bulk", case=False, na=False)]
        block = grp[grp["deal_type"].str.contains("block", case=False, na=False)]
        def agg_side(frame, side):
            subset = frame[frame["buy_sell"].str.contains(side, na=False)]
            qty = subset["quantity"].sum()
            val = subset["value"].sum()
            return qty, val
        bbuy_qty, bbuy_val = agg_side(bulk, "BUY")
        bsell_qty, bsell_val = agg_side(bulk, "SELL")
        bkbuy_qty, bkbuy_val = agg_side(block, "BUY")
        bksell_qty, bksell_val = agg_side(block, "SELL")
        large_deal_flag = (bulk["quantity"] >= cfg.bulk_threshold_shares).any() if not bulk.empty else False
        block_flag = not block.empty
        grouped.append({
            "date": dt,
            "symbol": cfg.symbol,
            "bulk_buy_qty": bbuy_qty,
            "bulk_sell_qty": bsell_qty,
            "bulk_net_qty": bbuy_qty - bsell_qty,
            "bulk_buy_value": bbuy_val,
            "bulk_sell_value": bsell_val,
            "bulk_net_value": bbuy_val - bsell_val,
            "block_buy_qty": bkbuy_qty,
            "block_sell_qty": bksell_qty,
            "block_net_qty": bkbuy_qty - bksell_qty,
            "block_buy_value": bkbuy_val,
            "block_sell_value": bksell_val,
            "block_net_value": bkbuy_val - bksell_val,
            "large_deal_flag": bool(large_deal_flag),
            "moil_bulk_threshold_hit": bool(large_deal_flag),
            "moil_block_deal_flag": bool(block_flag),
            "repeat_institution_5d": False,
            "repeat_institution_20d": False,
        })
    return pd.DataFrame(grouped)


def shareholding_features(shareholding: pd.DataFrame, cfg: TrackerConfig) -> pd.DataFrame:
    if shareholding is None or shareholding.empty:
        return pd.DataFrame(columns=[
            "quarter_end", "symbol", "promoter_pct", "public_pct", "mf_pct", "insurance_pct", "fpi_pct", "dii_pct_if_available",
            "named_1pct_holder_count", "qoq_change_mf_pct", "qoq_change_fpi_pct", "qoq_change_insurance_pct", "qoq_change_public_pct",
        ])
    df = shareholding.copy()
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], errors="coerce").dt.date
    latest_q = df["quarter_end"].max()
    prev_q = sorted(df["quarter_end"].dropna().unique())[:-1]
    prev_q = prev_q[-1] if prev_q else None
    def pct_for(cat: str) -> float:
        sub = df[(df["quarter_end"] == latest_q) & (df["category"].str.contains(cat, case=False, na=False))]
        return float(sub["percentage"].sum()) if not sub.empty else 0.0
    named_count = len(df[(df["quarter_end"] == latest_q) & (df["shares"] >= cfg.named_holder_threshold_shares)])
    row = {
        "quarter_end": latest_q,
        "symbol": cfg.symbol,
        "promoter_pct": pct_for("Promoter"),
        "public_pct": pct_for("Public"),
        "mf_pct": pct_for("Mutual Fund"),
        "insurance_pct": pct_for("Insurance"),
        "fpi_pct": pct_for("FPI"),
        "dii_pct_if_available": pct_for("Insurance") + pct_for("Mutual Fund"),
        "named_1pct_holder_count": named_count,
        "qoq_change_mf_pct": None,
        "qoq_change_fpi_pct": None,
        "qoq_change_insurance_pct": None,
        "qoq_change_public_pct": None,
    }
    if prev_q:
        prev = df[df["quarter_end"] == prev_q]
        def delta(cat: str) -> float | None:
            latest = row.get(f"{cat}_pct")
            prev_val = float(prev["percentage"].sum()) if not prev.empty else None
            return None if prev_val is None else latest - prev_val
        row["qoq_change_mf_pct"] = delta("mf")
        row["qoq_change_fpi_pct"] = delta("fpi")
        row["qoq_change_insurance_pct"] = delta("insurance")
        row["qoq_change_public_pct"] = delta("public")
    return pd.DataFrame([row])


def mf_features(mf: pd.DataFrame, cfg: TrackerConfig) -> pd.DataFrame:
    if mf is None or mf.empty:
        return pd.DataFrame(columns=[
            "month", "symbol", "mf_net_shares", "mf_buyer_count", "mf_seller_count",
            "mf_buyer_count_minus_seller_count", "top_mf_holder_shares", "top_mf_holder_name", "mf_total_moil_shares",
        ])
    df = mf.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["month"])
    df["shares"] = pd.to_numeric(df.get("shares", 0), errors="coerce").fillna(0)
    grouped = []
    for month, grp in df.groupby(df["month"].dt.date):
        total = grp["shares"].sum()
        top_idx = grp["shares"].idxmax()
        top_name = grp.loc[top_idx, "scheme_name"] if len(grp) else None
        top_shares = grp.loc[top_idx, "shares"] if len(grp) else None
        grouped.append({
            "month": month,
            "symbol": cfg.symbol,
            "mf_net_shares": total,
            "mf_buyer_count": len(grp[grp["shares"] > 0]),
            "mf_seller_count": len(grp[grp["shares"] < 0]),
            "mf_buyer_count_minus_seller_count": len(grp[grp["shares"] > 0]) - len(grp[grp["shares"] < 0]),
            "top_mf_holder_shares": top_shares,
            "top_mf_holder_name": top_name,
            "mf_total_moil_shares": total,
        })
    return pd.DataFrame(grouped)


def disclosures_features(disclosures: pd.DataFrame, cfg: TrackerConfig) -> pd.DataFrame:
    if disclosures is None or disclosures.empty:
        return pd.DataFrame(columns=[
            "event_date", "symbol", "sast_event_flag", "pit_event_flag", "shares_disclosed_in_events", "value_disclosed_in_events",
        ])
    df = disclosures.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    df = df.dropna(subset=["event_date"])
    df["shares"] = pd.to_numeric(df.get("shares", 0), errors="coerce").fillna(0)
    df["value_inr"] = pd.to_numeric(df.get("value_inr", 0), errors="coerce").fillna(0)
    rows = []
    for dt, grp in df.groupby("event_date"):
        trig = grp.get("trigger_type", "")
        sast_flag = bool((grp.get("trigger_type") == "SAST_5_PERCENT").any() or (grp.get("trigger_type") == "SAST_2_PERCENT_CHANGE").any()) if "trigger_type" in grp.columns else False
        pit_flag = bool((grp.get("trigger_type") == "PIT_10_LAKH_QUARTER").any()) if "trigger_type" in grp.columns else False
        rows.append({
            "event_date": dt,
            "symbol": cfg.symbol,
            "sast_event_flag": sast_flag,
            "pit_event_flag": pit_flag,
            "shares_disclosed_in_events": grp["shares"].sum(),
            "value_disclosed_in_events": grp["value_inr"].sum(),
        })
    return pd.DataFrame(rows)


def combine_features(deal_df: pd.DataFrame, shp_df: pd.DataFrame, mf_df: pd.DataFrame, disc_df: pd.DataFrame, cfg: TrackerConfig) -> pd.DataFrame:
    daily = deal_features(deal_df, cfg)
    quarterly = shareholding_features(shp_df, cfg)
    monthly = mf_features(mf_df, cfg)
    events = disclosures_features(disc_df, cfg)
    # Outer joins on date-like keys
    # daily (by date), events (date), monthly (month), quarterly (quarter_end)
    # For simplicity, align by date where available
    combined = daily.copy() if not daily.empty else pd.DataFrame()

    def safe_merge(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
        if left is None or left.empty:
            tmp = right.copy()
            tmp = tmp.rename(columns={right_on: left_on}) if right_on in tmp.columns else tmp
            return tmp
        merged = left.merge(right, left_on=left_on, right_on=right_on, how="outer", suffixes=("", "_dup"))
        # drop duplicate symbol columns if created
        for col in ["symbol_dup", "symbol_y", "symbol_x"]:
            if col in merged.columns:
                merged = merged.drop(columns=[col])
        # if both symbol and symbol_dup exist with same values, keep original
        return merged

    if not events.empty:
        combined = safe_merge(combined, events, "date", "event_date")
    if not monthly.empty:
        combined = safe_merge(combined, monthly, "date", "month")
    if not quarterly.empty:
        combined = safe_merge(combined, quarterly, "date", "quarter_end")

    combined["symbol"] = cfg.symbol
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.sort_values("date") if "date" in combined.columns else combined
    return combined


__all__ = [
    "deal_features",
    "shareholding_features",
    "mf_features",
    "disclosures_features",
    "combine_features",
]

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .config import TrackerConfig

SHP_TEMPLATE = [
    {
        "quarter_end": "2026-03-31",
        "category": "Promoter & Promoter Group",
        "sub_category": "Promoter",
        "holder_name": "Government of India",
        "shares": 131_624_088,
        "percentage": 64.68,
    },
    {
        "quarter_end": "2026-03-31",
        "category": "Public",
        "sub_category": "Mutual Funds",
        "holder_name": "MF total",
        "shares": 4_960_456,
        "percentage": 2.44,
    },
    {
        "quarter_end": "2026-03-31",
        "category": "Public",
        "sub_category": "Insurance Companies",
        "holder_name": "LIC",
        "shares": 7_010_761,
        "percentage": 3.45,
    },
    {
        "quarter_end": "2026-03-31",
        "category": "Public",
        "sub_category": "FPI Category I+II",
        "holder_name": "FPI total",
        "shares": 8_478_486,
        "percentage": 4.17,
    },
    {
        "quarter_end": "2026-03-31",
        "category": "Public",
        "sub_category": "Named Holder",
        "holder_name": "Bandhan Small Cap Fund",
        "shares": 2_125_382,
        "percentage": 1.04,
    },
]


@dataclass(slots=True)
class ShareholdingRow:
    quarter_end: str
    category: str
    sub_category: str
    holder_name: str
    shares: float
    percentage: float
    source_url: str
    source_file: str
    extracted_at: str


def load_manual_shp_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"quarter_end", "category", "sub_category", "holder_name", "shares", "percentage"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in SHP CSV: {missing}")
    return df


def save_template_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SHP_TEMPLATE).to_csv(path, index=False)


def parse_supplemental_csv(path: Path, source_url: str) -> List[ShareholdingRow]:
    df = load_manual_shp_csv(path)
    rows: List[ShareholdingRow] = []
    for _, r in df.iterrows():
        rows.append(
            ShareholdingRow(
                quarter_end=str(r["quarter_end"]),
                category=str(r["category"]),
                sub_category=str(r["sub_category"]),
                holder_name=str(r["holder_name"]),
                shares=float(r["shares"]),
                percentage=float(r["percentage"]),
                source_url=source_url,
                source_file=str(path),
                extracted_at=datetime.utcnow().isoformat(),
            )
        )
    return rows


def compare_quarters(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    if current is None or current.empty:
        return pd.DataFrame()
    current = current.copy()
    prev = previous.copy() if previous is not None else pd.DataFrame()
    key = ["category", "sub_category", "holder_name"]
    current["key"] = current[key].agg("|".join, axis=1)
    prev["key"] = prev[key].agg("|".join, axis=1) if not prev.empty else None
    merged = current.merge(prev[["key", "percentage", "shares"]] if not prev.empty else prev, on="key", how="left", suffixes=("", "_prev"))
    merged["delta_pct_points"] = merged["percentage"] - merged.get("percentage_prev")
    merged["delta_shares"] = merged["shares"] - merged.get("shares_prev")
    return merged.drop(columns=[c for c in ["key", "percentage_prev", "shares_prev"] if c in merged.columns])


def persist_shareholding(rows: List[ShareholdingRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [row.__dict__ for row in rows]
    output_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


__all__ = [
    "ShareholdingRow",
    "load_manual_shp_csv",
    "save_template_csv",
    "parse_supplemental_csv",
    "compare_quarters",
    "persist_shareholding",
]

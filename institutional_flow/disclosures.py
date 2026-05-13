from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List

DISCLOSURE_SCHEMA = [
    "event_date",
    "disclosure_type",
    "entity_name",
    "symbol",
    "acquisition_or_disposal",
    "shares",
    "percentage",
    "value_inr",
    "trigger_type",
    "source_url",
    "notes",
]


def load_disclosure_csv(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in DISCLOSURE_SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[df["symbol"].astype(str).str.upper() == symbol.upper()] if "symbol" in df.columns else df
    return df[DISCLOSURE_SCHEMA]


def classify_trigger(row: pd.Series, cfg_shares: int) -> str:
    try:
        pct = float(row.get("percentage") or 0)
        shares = float(row.get("shares") or 0)
    except Exception:  # noqa: BLE001
        pct = 0.0
        shares = 0.0
    if pct >= 5.0 or shares >= cfg_shares * 0.05:
        return "SAST_5_PERCENT"
    if pct >= 2.0 or shares >= cfg_shares * 0.02:
        return "SAST_2_PERCENT_CHANGE"
    try:
        val = float(row.get("value_inr") or 0)
    except Exception:  # noqa: BLE001
        val = 0.0
    if val >= 1_000_000:
        return "PIT_10_LAKH_QUARTER"
    return "OTHER"


__all__ = ["load_disclosure_csv", "classify_trigger", "DISCLOSURE_SCHEMA"]

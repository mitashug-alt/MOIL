from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List

from .config import TrackerConfig

MF_SCHEMA = ["month", "amc", "scheme_name", "shares", "market_value", "percentage_of_scheme", "source"]


def load_mf_manual_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    # best-effort filter rows containing MOIL
    mask = df.apply(lambda s: s.astype(str).str.contains("MOIL", case=False, na=False))
    if isinstance(mask, pd.DataFrame):
        mask_any = mask.any(axis=1)
    else:
        mask_any = mask
    filtered = df[mask_any].copy()
    if filtered.empty:
        return pd.DataFrame(columns=MF_SCHEMA)
    rename = {
        "Month": "month",
        "AMC": "amc",
        "Scheme": "scheme_name",
        "Shares": "shares",
        "Market Value": "market_value",
        "Pct of Scheme": "percentage_of_scheme",
    }
    filtered = filtered.rename(columns={k: v for k, v in rename.items() if k in filtered.columns})
    for col in MF_SCHEMA:
        if col not in filtered.columns:
            filtered[col] = None
    return filtered[MF_SCHEMA]


def mfdata_stub(symbol_variants: List[str]) -> pd.DataFrame:
    """Placeholder for mfdata.in API client; returns empty in offline mode."""
    return pd.DataFrame(columns=MF_SCHEMA)


__all__ = ["load_mf_manual_file", "mfdata_stub", "MF_SCHEMA"]

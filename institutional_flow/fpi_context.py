from __future__ import annotations

import pandas as pd
from pathlib import Path

FPI_SCHEMA = ["report_date", "category", "net_investment_inr_cr", "source"]


def load_fpi_manual(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FPI_SCHEMA)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in FPI_SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df[FPI_SCHEMA]


__all__ = ["load_fpi_manual", "FPI_SCHEMA"]

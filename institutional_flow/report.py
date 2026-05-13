from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import TrackerConfig


def build_summary(cfg: TrackerConfig, deal_df: pd.DataFrame, shp_df: pd.DataFrame, mf_df: pd.DataFrame, disc_df: pd.DataFrame) -> str:
    lines = ["# MOIL Institutional Flow Summary", ""]
    lines.append(f"Symbol: {cfg.symbol}")
    lines.append(f"Outstanding shares: {cfg.outstanding_shares:,}")
    lines.append("\n## Thresholds")
    lines.append(f"- Bulk threshold (0.5%): {cfg.bulk_threshold_shares:,} shares ({cfg.bulk_threshold_pct:.3f}%)")
    lines.append(f"- Named holder (1%): {cfg.named_holder_threshold_shares:,} shares")
    lines.append(f"- SAST 5%: {cfg.sast_five_percent_shares:,} shares")
    lines.append(f"- SAST 2% change: {cfg.sast_two_percent_shares:,} shares")
    lines.append("\n## Data availability")
    lines.append(f"- NSE deals rows: {len(deal_df) if deal_df is not None else 0}")
    lines.append(f"- Shareholding rows: {len(shp_df) if shp_df is not None else 0}")
    lines.append(f"- MF rows: {len(mf_df) if mf_df is not None else 0}")
    lines.append(f"- Disclosure rows: {len(disc_df) if disc_df is not None else 0}\n")
    return "\n".join(lines)


def write_report(md_text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_text, encoding="utf-8")


def write_features(df: pd.DataFrame, out_csv: Path, out_parquet: Path | None = None) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if out_parquet is not None:
        try:
            df.to_parquet(out_parquet, index=False)
        except Exception:
            # pyarrow optional; ignore if missing
            pass


__all__ = ["build_summary", "write_report", "write_features"]

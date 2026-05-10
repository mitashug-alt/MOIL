"""Cached backend-feed loader for MOIL Macro Radar.

Data Quality v2:
- Scheduled scrapers write raw/cache JSON files under data/cache.
- This module converts raw scraper output into canonical evidence rows.
- Evidence rows receive data-confidence scores before Groq or regime scoring.
- Streamlit should read cached JSON, not scrape live sites on page load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_layer.evidence_store import (
    CANONICAL_INDICATORS,
    evidence_rows_from_macro_cache,
    evidence_to_macro_rows,
    evidence_to_source_readiness,
)

CACHE_DIR = Path("data/cache")
LATEST_MACRO_PATH = CACHE_DIR / "latest_macro_data.json"
LATEST_INSTITUTIONAL_PATH = CACHE_DIR / "institutional_activity_latest.json"

CANONICAL_MACRO_INDICATORS = CANONICAL_INDICATORS


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"Cache file not found: {path}", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Could not read {path}: {exc}", "path": str(path)}


def load_latest_macro_cache() -> dict[str, Any]:
    """Return latest scheduled macro scrape result."""
    return _read_json(LATEST_MACRO_PATH)


def load_latest_institutional_cache() -> dict[str, Any]:
    """Return latest smart-money tracker scrape result."""
    return _read_json(LATEST_INSTITUTIONAL_PATH)


def macro_cache_to_evidence_rows(cache: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return canonical evidence rows from latest macro cache."""
    cache = cache if cache is not None else load_latest_macro_cache()
    return evidence_rows_from_macro_cache(cache)


def macro_cache_to_source_readiness(cache: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return Data Quality v2 source-readiness table."""
    evidence = macro_cache_to_evidence_rows(cache)
    return evidence_to_source_readiness(evidence)


def macro_cache_to_manual_rows(cache: dict[str, Any] | None = None) -> pd.DataFrame:
    """Convert scheduled scraper output into Groq-scoreable macro rows.

    Output keeps the legacy manual macro columns and adds confidence metadata:
    data_confidence, confidence_weight, scoring_allowed, source_tier, exact_data.
    """
    evidence = macro_cache_to_evidence_rows(cache)
    return evidence_to_macro_rows(evidence)


def institutional_cache_summary(cache: dict[str, Any]) -> dict[str, Any]:
    """Compact summary for app cards / Groq context."""
    if not isinstance(cache, dict) or not cache.get("ok", False):
        return {
            "ok": False,
            "message": cache.get("error", "No institutional cache available") if isinstance(cache, dict) else "No institutional cache available",
            "deals": 0,
            "shareholding_rows": 0,
            "sast_filings": 0,
            "smart_money_signals": 0,
        }
    return {
        "ok": True,
        "scraped_at": cache.get("scraped_at"),
        "deals": len(cache.get("institutional_deals", []) or []),
        "shareholding_rows": len(cache.get("shareholding_observations", []) or []),
        "sast_filings": len(cache.get("sast_filings", []) or []),
        "smart_money_signals": len(cache.get("smart_money_signals", []) or []),
        "source_status": cache.get("source_status", {}),
        "diagnostics": cache.get("diagnostics", []),
    }

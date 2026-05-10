"""Cached backend-feed loader for MOIL Macro Radar.

The scheduled scrapers write JSON files under data/cache. Streamlit should read
these cached files instead of scraping external sites on every page load.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_DIR = Path("data/cache")
LATEST_MACRO_PATH = CACHE_DIR / "latest_macro_data.json"
LATEST_INSTITUTIONAL_PATH = CACHE_DIR / "institutional_activity_latest.json"


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


def macro_cache_to_manual_rows(cache: dict[str, Any]) -> pd.DataFrame:
    """Convert scheduled scraper output into Groq-scoreable macro rows.

    The output schema matches data/manual_macro_template.csv:
    date, indicator, value, unit, status, score, commentary, source
    """
    columns = ["date", "indicator", "value", "unit", "status", "score", "commentary", "source"]
    rows: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not isinstance(cache, dict) or not cache:
        return pd.DataFrame(columns=columns)

    # MOIL PDF observations: exact company-linked price/circular facts.
    moil = cache.get("moil") or {}
    if moil:
        price_obs = moil.get("price_observations") or []
        revisions = moil.get("category_revisions") or []
        source_doc = moil.get("source_document") or {}
        if price_obs or revisions:
            facts = []
            for item in price_obs[:8]:
                facts.append(
                    f"{item.get('product_family','MOIL')} {item.get('ore_code','')}: "
                    f"{item.get('price_value')} {item.get('currency','INR')}/{item.get('unit','PMT')} "
                    f"basis={item.get('basis','')} effective={item.get('effective_date','')}"
                )
            for item in revisions[:8]:
                facts.append(
                    f"{item.get('category')}: {item.get('direction')} "
                    f"{item.get('change_pct')}% effective={item.get('effective_date','')}"
                )
            rows.append(
                {
                    "date": today,
                    "indicator": "MOIL Ore / EMD Price Circular",
                    "value": f"{len(price_obs)} price rows; {len(revisions)} category revisions",
                    "unit": "INR/PMT / % revision",
                    "status": "auto-cache",
                    "score": 0.0,
                    "commentary": " | ".join(facts)[:2500] or "MOIL cache present but no extracted values.",
                    "source": source_doc.get("source_url") or "MOIL scheduled PDF scraper",
                }
            )

    # HTML source observations: China steel/export/power and public proxies.
    for source_result in cache.get("html_sources", []) or []:
        source = source_result.get("source") or {}
        observations = source_result.get("observations") or []
        if not observations:
            continue
        indicator = source.get("indicator") or observations[0].get("indicator") or "Public macro table"
        facts = []
        for obs in observations[:8]:
            facts.append(
                f"period={obs.get('period')}; value={obs.get('value')}; "
                f"raw={obs.get('raw_text','')[:220]}"
            )
        rows.append(
            {
                "date": today,
                "indicator": indicator,
                "value": f"{len(observations)} cached public observation(s)",
                "unit": "public table/snippet",
                "status": "auto-cache",
                "score": 0.0,
                "commentary": " | ".join(facts)[:2500],
                "source": source.get("name") or source.get("url") or "scheduled HTML scraper",
            }
        )

    return pd.DataFrame(rows, columns=columns)


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

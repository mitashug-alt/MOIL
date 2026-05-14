"""Automatic data-feed helpers for MOIL Macro Radar.

These helpers fetch external data and create structured rows that the Groq AI
scorer can evaluate. Groq/Llama itself does not browse the web; the app must
supply facts/snippets first, then Groq scores them.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

# Provider-aware source map. Paid/vendor sources are only used when the user
# supplies an endpoint/API key in secrets or environment variables. Otherwise,
# the app falls back to source-targeted web search snippets and asks Groq to
# score only the facts/snippets supplied to it.
SOURCE_REGISTRY: Dict[str, Dict[str, object]] = {
    "Silico-Manganese Prices": {
        "priority_sources": ["Argus Media", "FerroAlloyNet", "BigMint/SteelMint manual backup"],
        "source_type": "paid/vendor commodity price assessment",
        "endpoint_secret": "ARGUS_SIMN_FEED_URL",
        "api_key_secret": "ARGUS_API_KEY",
        "fallback_query": "India silico manganese SiMn 60-14 price Raipur Durgapur latest Argus FerroAlloyNet BigMint SteelMint",
        "notes": "Exact SiMn 60-14 prices usually require a paid feed; public snippets may be incomplete.",
    },
    "India Crude Steel Production": {
        "priority_sources": ["Ministry of Steel / JPC", "World Steel Association"],
        "source_type": "official monthly steel production data",
        "endpoint_secret": "JPC_STEEL_FEED_URL",
        "api_key_secret": "JPC_API_KEY",
        "fallback_query": "site:steel.gov.in monthly economic report India crude steel production latest month YoY OR site:worldsteel.org India crude steel production latest month",
        "notes": "Use Ministry/JPC as primary India source; worldsteel for global comparison.",
    },
    "China Steel Exports": {
        "priority_sources": ["GACC", "Mysteel", "Argus Media"],
        "source_type": "official customs data plus market commentary",
        "endpoint_secret": "GACC_STEEL_EXPORTS_FEED_URL",
        "api_key_secret": "GACC_API_KEY",
        "fallback_query": "site:english.customs.gov.cn China Major Exports steel products latest month GACC OR China steel exports latest month Mysteel Argus",
        "notes": "GACC is official volume source; Mysteel/Argus are useful for interpretation.",
    },
    "China Power / Industrial Stress": {
        "priority_sources": ["NBS China", "Bloomberg", "S&P Global Commodity Insights"],
        "source_type": "industrial production/electricity stress proxy",
        "endpoint_secret": "CHINA_POWER_STRESS_FEED_URL",
        "api_key_secret": "CHINA_POWER_API_KEY",
        "fallback_query": "site:stats.gov.cn english China industrial production electricity generation latest steel cement power stress OR Bloomberg China power industrial stress OR S&P Global Commodity Insights China power demand manufacturing stress",
        "notes": "NBS gives official industrial/power proxies; Bloomberg/S&P provide faster market stress monitoring when licensed.",
    },
}

SCRAPER_TO_CANONICAL = {
    "Silico-Manganese 65/17 Price (Delivered Europe)": "Silico-Manganese Prices",
    "India Crude Steel Production (Total)": "India Crude Steel Production",
    "China Steel Exports (Current Month)": "China Steel Exports",
    "China Power Generation (Above-scale Industrial)": "China Power / Industrial Stress",
}

INDICATOR_UNIT_HINTS = {
    "Silico-Manganese Prices": "$US/MT",
    "India Crude Steel Production": "thousand tonnes",
    "China Steel Exports": "tonnes",
    "China Power / Industrial Stress": "亿千瓦时",
}

DEFAULT_SERPER_QUERIES: Dict[str, str] = {
    indicator: str(meta["fallback_query"])
    for indicator, meta in SOURCE_REGISTRY.items()
}

# HeadlineFeed fallback queries (headline-based extraction when SERPER/vendor/cache unavailable)
HEADLINEFEED_QUERIES: Dict[str, str] = {
    "Silico-Manganese Prices": "silico manganese price India",
    "India Crude Steel Production": "India crude steel production latest month",
    "China Steel Exports": "China steel exports latest month GACC",
    "China Power / Industrial Stress": "China electricity production industrial output NBS",
}

# Lightweight in-process caches to avoid burning daily quotas on repeated pulls
_SERPER_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_HEADLINE_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 3600  # 1h cache for search/headline fallbacks

NSE_ARCHIVE_CANDIDATES = [
    ("bulk", "https://nsearchives.nseindia.com/content/equities/bulk.csv"),
    ("block", "https://nsearchives.nseindia.com/content/equities/block.csv"),
]
_NSE_ARCHIVE_CACHE: Dict[str, Tuple[str, pd.DataFrame, str]] = {}
_NSE_SYMBOL_CACHE: Dict[str, Tuple[str, pd.DataFrame, List[str], str]] = {}
_NSE_CACHE_TTL_SECONDS = 4 * 3600  # 4h freshness window


def _get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment."""
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return str(os.getenv(name, default) or "").strip()


def _parse_numeric_and_date(text: str) -> Tuple[float | None, str | None]:
    """Extract first numeric value and date-ish token from text.

    - Numeric: first match of digits with commas/decimals.
    - Date: month name + year ("April 2026"), or YYYY-MM, or YYYY.
    """
    import re

    if not text:
        return None, None
    num_match = re.search(r"([0-9][0-9,\.]+)", text)
    val: float | None = None
    if num_match:
        try:
            val = float(num_match.group(1).replace(",", ""))
        except Exception:
            val = None

    date: str | None = None
    month_map = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", text, re.IGNORECASE)
    if m:
        mon = month_map.get(m.group(1).lower())
        yr = m.group(2)
        if mon:
            date = f"{yr}-{mon}-01"
    else:
        m = re.search(r"(\d{4})-(\d{1,2})", text)
        if m:
            yr, mon = m.group(1), m.group(2).zfill(2)
            date = f"{yr}-{mon}-01"
        else:
            m = re.search(r"\b(20\d{2})\b", text)
            if m:
                date = f"{m.group(1)}-01-01"
    return val, date


def _parse_indicator_specific(indicator: str, text: str) -> Tuple[float | None, str | None, str | None]:
    """Parse numeric/date/unit from snippet text with indicator-specific hints."""
    import re

    if not text:
        return None, None, None
    lo_indicator = indicator.lower()
    unit_hint = INDICATOR_UNIT_HINTS.get(indicator, "")

    if "silico-manganese" in lo_indicator or "simn" in text.lower():
        m = re.search(r"\$\s*([0-9][0-9,\.]+)\s*/?\s*(?:mt|tonne)", text, re.IGNORECASE)
        val = float(m.group(1).replace(",", "")) if m else None
        _, dt = _parse_numeric_and_date(text)
        return val, dt, "$US/MT"

    if "crude steel" in lo_indicator or "steel production" in text.lower():
        m = re.search(r"([0-9][0-9,\.]+)\s*(?:thousand\s+)?(?:tonnes|mt|million\s+tonnes)", text, re.IGNORECASE)
        val = None
        if m:
            raw = float(m.group(1).replace(",", ""))
            # If unit says million tonnes, convert to thousand
            if "million" in m.group(0).lower():
                raw = raw * 1000
            val = raw
        _, dt = _parse_numeric_and_date(text)
        return val, dt, "thousand tonnes"

    if "steel exports" in lo_indicator:
        m = re.search(r"([0-9][0-9,\.]+)\s*(million|\d+\s*0{4,}|tonnes|tons)", text, re.IGNORECASE)
        val = None
        if m:
            raw = float(m.group(1).replace(",", ""))
            if "million" in m.group(2).lower():
                raw = raw * 1_000_000
            elif re.search(r"0{4,}$", m.group(2)):
                # crude heuristic if written as e.g., 10000 (10k tonnes)
                raw = raw
            val = raw
        _, dt = _parse_numeric_and_date(text)
        return val, dt, "tonnes"

    if "power" in lo_indicator or "industrial stress" in lo_indicator:
        m = re.search(r"([0-9][0-9,\.]+)\s*(?:亿\s*千瓦时|twh|terawatt\s*hours)", text, re.IGNORECASE)
        val = None
        unit = "亿千瓦时"
        if m:
            raw = float(m.group(1).replace(",", ""))
            if "twh" in m.group(0).lower():
                # 1 TWh = 0.1 亿千瓦时
                raw = raw * 0.1
                unit = "亿千瓦时"
            val = raw
        _, dt = _parse_numeric_and_date(text)
        return val, dt, unit

    # fallback generic
    val, dt = _parse_numeric_and_date(text)
    return val, dt, unit_hint or ""


def is_serper_configured() -> bool:
    return bool(_get_secret("SERPER_API_KEY"))


def is_any_vendor_feed_configured() -> bool:
    """Return True if any source-specific endpoint is configured."""
    for meta in SOURCE_REGISTRY.values():
        endpoint_secret = str(meta.get("endpoint_secret", ""))
        if endpoint_secret and _get_secret(endpoint_secret):
            return True
    return False



def is_cached_macro_feed_available() -> bool:
    """Return True when scheduled backend cache has at least one macro row.

    This lets the UI treat data/cache/latest_macro_data.json as an automatic
    source, even when no paid vendor endpoint or SERPER_API_KEY is configured.
    """
    try:
        from backend_cache_loader import load_latest_macro_cache, macro_cache_to_manual_rows

        cache = load_latest_macro_cache()
        rows = macro_cache_to_manual_rows(cache)
        return rows is not None and not rows.empty
    except Exception:
        return False


def cached_macro_indicators() -> set[str]:
    """Return canonical indicator names currently available in backend cache."""
    try:
        from backend_cache_loader import load_latest_macro_cache, macro_cache_to_manual_rows

        cache = load_latest_macro_cache()
        rows = macro_cache_to_manual_rows(cache)
        if rows is None or rows.empty or "indicator" not in rows.columns:
            return set()
        return {str(x).strip() for x in rows["indicator"].dropna().tolist() if str(x).strip()}
    except Exception:
        return set()

def source_provider_status() -> pd.DataFrame:
    """Expose source readiness to the UI without displaying secrets.

    Data Quality v2 prefers the scheduled evidence cache. If no cache evidence
    exists for an indicator, the table still shows vendor/search/manual fallback
    readiness.
    """
    try:
        from backend_cache_loader import load_latest_macro_cache, macro_cache_to_source_readiness

        cache = load_latest_macro_cache()
        readiness = macro_cache_to_source_readiness(cache)
    except Exception:
        readiness = pd.DataFrame()

    cache_map = {}
    if readiness is not None and not readiness.empty and "indicator" in readiness.columns:
        cache_map = {str(row["indicator"]): row.to_dict() for _, row in readiness.iterrows()}

    rows = []
    serper_ready = is_serper_configured()

    for indicator, meta in SOURCE_REGISTRY.items():
        endpoint_secret = str(meta.get("endpoint_secret", ""))
        api_key_secret = str(meta.get("api_key_secret", ""))
        endpoint_configured = bool(endpoint_secret and _get_secret(endpoint_secret))
        api_key_configured = bool(api_key_secret and _get_secret(api_key_secret))
        cached = cache_map.get(indicator, {})
        cache_row_found = bool(cached) and int(cached.get("evidence_rows", 0) or 0) > 0

        if cache_row_found and bool(cached.get("scoring_allowed", False)):
            mode = "scheduled evidence cache"
            status = "configured/cache"
        elif cache_row_found:
            mode = "scheduled cache commentary-only"
            status = "low-confidence"
        elif endpoint_configured:
            mode = "vendor endpoint"
            status = "configured"
        elif serper_ready:
            mode = "targeted web search fallback"
            status = "fallback"
        else:
            mode = "manual only"
            status = "not configured"

        rows.append(
            {
                "indicator": indicator,
                "priority_sources": ", ".join(meta.get("priority_sources", [])),
                "best_source_found": cached.get("best_source_found", "None"),
                "source_tier": cached.get("source_tier", ""),
                "exact_data": cached.get("exact_data", False),
                "latest_period": cached.get("latest_period", ""),
                "evidence_rows": cached.get("evidence_rows", 0),
                "confidence": cached.get("confidence", 0),
                "confidence_label": cached.get("confidence_label", "missing"),
                "confidence_weight": cached.get("confidence_weight", 0.0),
                "scoring_allowed": cached.get("scoring_allowed", False),
                "source_type": meta.get("source_type", ""),
                "scheduled_cache_row": cache_row_found,
                "endpoint_secret": endpoint_secret,
                "endpoint_configured": endpoint_configured,
                "api_key_secret": api_key_secret,
                "api_key_configured": api_key_configured,
                "fallback_search": serper_ready,
                "mode": mode,
                "status": status,
                "notes": cached.get("notes") or meta.get("notes", ""),
            }
        )

    return pd.DataFrame(rows)


def _today_iso() -> str:
    # IST is UTC + 5:30
    ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    return ist_time.strftime("%Y-%m-%d")


def _safe_request_get(url: str, timeout: int = 20, retries: int = 3, backoff: float = 1.5) -> Tuple[bool, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }
    last_err = ""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code >= 400:
                last_err = f"HTTP {response.status_code} from {url}"
                continue
            text = response.text or ""
            if not text.strip():
                last_err = f"Empty response from {url}"
                continue
            return True, text, f"OK {url} rows/text={len(text)}"
        except Exception as exc:
            last_err = f"Request failed for {url}: {exc}"
        time.sleep(backoff)
    return False, "", last_err or f"Failed to fetch {url} after {retries} attempts"


def _safe_vendor_get(endpoint_secret: str, api_key_secret: str = "", timeout: int = 30) -> Tuple[bool, str, str]:
    """Fetch a user-configured vendor endpoint.

    This is deliberately generic because vendors such as Argus, FerroAlloyNet,
    Mysteel, Bloomberg and S&P use account-specific delivery mechanisms. The
    endpoint can be an internal proxy, a signed CSV export URL, or a lightweight
    API wrapper created by the user. API keys are never displayed.
    """
    url = _get_secret(endpoint_secret)
    if not url:
        return False, "", f"{endpoint_secret} not configured"
    headers = {
        "User-Agent": "MOIL-Macro-Radar/1.0",
        "Accept": "application/json,text/csv,text/plain,*/*",
    }
    api_key = _get_secret(api_key_secret) if api_key_secret else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            return False, "", f"{endpoint_secret}: HTTP {response.status_code}"
        text = response.text or ""
        if not text.strip():
            return False, "", f"{endpoint_secret}: empty response"
        return True, text, f"{endpoint_secret}: vendor endpoint OK, chars={len(text)}"
    except Exception as exc:
        return False, "", f"{endpoint_secret}: request failed: {exc}"


def _standardize_nse_deal_columns(df: pd.DataFrame, deal_type: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {
        "Date": "date",
        "Symbol": "symbol",
        "Security Name": "security_name",
        "Client Name": "client_name",
        "Buy/Sell": "buy_sell",
        "Quantity Traded": "quantity",
        "Trade Price / Wght. Avg. Price": "price",
        "Remarks": "remarks",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    for col in ["date", "symbol", "security_name", "client_name", "buy_sell", "quantity", "price", "remarks"]:
        if col not in out.columns:
            out[col] = ""
    out["deal_type"] = deal_type
    out["source"] = "NSE archive CSV"
    return out[["date", "symbol", "security_name", "client_name", "buy_sell", "quantity", "price", "remarks", "deal_type", "source"]]


def fetch_nse_deals_auto(symbol: str = "MOIL") -> Tuple[pd.DataFrame, List[str]]:
    """Fetch latest NSE bulk/block archive CSVs and filter to a symbol.

    Returns a tuple of (deals_dataframe, diagnostics). If the archive loads but
    the symbol has no rows, diagnostics makes that explicit so the UI does not
    incorrectly label it as an endpoint failure.
    """
    symbol = (symbol or "MOIL").strip().upper()
    today = _today_iso()

    # Symbol-level cache to avoid repeated filtering in a single day (respect TTL)
    sym_cache = _NSE_SYMBOL_CACHE.get(symbol)
    if sym_cache:
        cached_day, cached_df, cached_diag, cached_status = sym_cache
        age_secs = time.time() - datetime.strptime(cached_day, "%Y-%m-%d").timestamp()
        age_ok = cached_day == today or age_secs < _NSE_CACHE_TTL_SECONDS
        if age_ok:
            return cached_df.copy(), list(cached_diag)

    diagnostics: List[str] = []
    frames: List[pd.DataFrame] = []
    status_badge = "fresh"

    for deal_type, url in NSE_ARCHIVE_CANDIDATES:
        cache_entry = _NSE_ARCHIVE_CACHE.get(deal_type)
        std = None
        if cache_entry:
            cached_day, cached_df, cached_detail = cache_entry
            age_secs = time.time() - datetime.strptime(cached_day, "%Y-%m-%d").timestamp()
            age_ok = cached_day == today or age_secs < _NSE_CACHE_TTL_SECONDS
            if age_ok:
                std = cached_df
                diagnostics.append(f"{deal_type}: cache hit ({len(std)} rows)")
        if std is None:
            ok, text, detail = _safe_request_get(url)
            diagnostics.append(f"{deal_type}: {detail}")
            if not ok:
                continue
            try:
                df = pd.read_csv(io.StringIO(text))
                std = _standardize_nse_deal_columns(df, deal_type)
                # Drop rows missing critical fields
                std = std.dropna(subset=["symbol", "date", "quantity", "price"], how="any")
                # Date freshness check
                latest_date = None
                if not std.empty and "date" in std.columns:
                    dt = pd.to_datetime(std["date"], errors="coerce")
                    if dt.notna().any():
                        latest_date = dt.max().date().isoformat()
                if latest_date:
                    diagnostics.append(f"{deal_type}: latest date {latest_date}")
                    if latest_date < today:
                        status_badge = "stale"
                else:
                    diagnostics.append(f"{deal_type}: latest date unknown (parse failed)")
                _NSE_ARCHIVE_CACHE[deal_type] = (today, std, detail)
            except Exception as exc:
                diagnostics.append(f"{deal_type}: CSV parse failed: {exc}")
                continue
        if std is not None and not std.empty:
            frames.append(std)

    if not frames:
        return pd.DataFrame(), diagnostics

    all_deals = pd.concat(frames, ignore_index=True)
    # Dedup across bulk/block using stable fields
    dedup_cols = ["date", "symbol", "client_name", "buy_sell", "quantity", "price", "deal_type"]
    existing_cols = [c for c in dedup_cols if c in all_deals.columns]
    if existing_cols:
        all_deals = all_deals.drop_duplicates(subset=existing_cols)

    # Basic numeric validation
    if "quantity" in all_deals.columns:
        all_deals = all_deals[pd.to_numeric(all_deals["quantity"], errors="coerce") > 0]
    if "price" in all_deals.columns:
        all_deals = all_deals[pd.to_numeric(all_deals["price"], errors="coerce") > 0]

    # Secondary symbol filter by security_name contains symbol (if provided)
    matches = all_deals[all_deals["symbol"].astype(str).str.upper() == symbol].copy()
    if matches.empty and "security_name" in all_deals.columns:
        mask = all_deals["security_name"].astype(str).str.upper().str.contains(symbol, na=False)
        matches = all_deals[mask].copy()
    if matches.empty:
        diagnostics.append(f"NSE archive loaded successfully, but no {symbol} bulk/block rows were present in the latest archive.")
    else:
        diagnostics.append(f"Found {len(matches)} {symbol} NSE bulk/block row(s).")

    _NSE_SYMBOL_CACHE[symbol] = (today, matches.copy(), list(diagnostics), status_badge)
    return matches, diagnostics


def nse_deals_to_tracker_updates(deals: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "category", "item", "value", "impact", "confidence", "sentiment", "details", "source"]
    if deals is None or deals.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in deals.iterrows():
        side = str(row.get("buy_sell", "")).upper()
        impact = 1.0 if "BUY" in side else -1.0 if "SELL" in side else 0.0
        sentiment = "Bullish" if impact > 0 else "Bearish" if impact < 0 else "Neutral"
        details = (
            f"{row.get('deal_type','deal')} {row.get('symbol','')}: "
            f"{row.get('client_name','')} {side} qty={row.get('quantity','')} "
            f"price={row.get('price','')} remarks={row.get('remarks','')}"
        )
        rows.append(
            {
                "date": str(row.get("date", "")),
                "category": "Institutional",
                "item": str(row.get("client_name", "NSE deal")),
                "value": side or str(row.get("deal_type", "deal")),
                "impact": impact,
                "confidence": 0.75,
                "sentiment": sentiment,
                "details": details[:800],
                "source": "NSE archive CSV",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def fetch_source_aware_macro_feed(max_results_per_query: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch macro rows using cache, vendor endpoints, then search.

    Data flow:
    1. Read scheduled scraper cache from data/cache/latest_macro_data.json if present.
    2. Pull configured source-specific vendor endpoints.
    3. Pull targeted Serper search snippets for remaining indicators.
    4. Fall back to manual rows with diagnostics.

    The function returns rows in the same schema as the manual macro table so
    the existing Groq scorer can evaluate them.
    """
    columns = None  # preserve extended Data Quality v2 columns when present
    rows = []
    diagnostics: List[str] = []

    # First pass: scheduled backend cache. This is the preferred free/no-API
    # route because scraping happens once daily, outside user page loads.
    try:
        from backend_cache_loader import load_latest_macro_cache, macro_cache_to_manual_rows

        cached = load_latest_macro_cache()
        if isinstance(cached, dict):
            if cached.get("ok") is not False and (cached.get("moil") or cached.get("html_sources")):
                cached_rows = macro_cache_to_manual_rows(cached)
                if cached_rows is not None and not cached_rows.empty:
                    rows.extend(cached_rows.to_dict(orient="records"))
                    diagnostics.append(f"scheduled cache: loaded {len(cached_rows)} macro row(s) from data/cache/latest_macro_data.json")
            else:
                diagnostics.append(cached.get("error", "scheduled cache: no macro cache file found yet"))
        else:
            diagnostics.append("scheduled cache: simple list payload detected; using simple macro cache instead")
    except Exception as exc:
        diagnostics.append(f"scheduled cache read failed: {exc}")

    # Supplemental: load simple scraper cache (macro_data.json list of records)
    try:
        simple_cache_path = Path("data/cache/latest_macro_data.json")
        if simple_cache_path.exists():
            raw = json.loads(simple_cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                df_simple = pd.DataFrame(raw)
                # map scraper indicator names to canonical ones used by SOURCE_REGISTRY
                df_simple["indicator"] = df_simple["indicator"].apply(lambda x: SCRAPER_TO_CANONICAL.get(str(x).strip(), str(x).strip()))
                required = {"indicator", "value", "date"}
                if required.issubset(df_simple.columns):
                    df_simple = df_simple.fillna("")
                    # Map to manual macro schema with defaults
                    df_simple = df_simple.rename(columns={"source_name": "source"})
                    df_simple["status"] = df_simple.get("status", "cache").fillna("cache")
                    df_simple["unit"] = df_simple.get("unit", "")
                    df_simple["commentary"] = df_simple.get("notes", "")
                    df_simple["data_confidence"] = df_simple["status"].apply(lambda s: 90 if str(s).lower() in {"ok", "cache", "fallback"} else 60)
                    df_simple["confidence_weight"] = 1.0
                    df_simple["scoring_allowed"] = True
                    df_simple["confidence_label"] = df_simple["data_confidence"].apply(lambda c: "high" if c >= 85 else "medium")
                    df_simple["source_type"] = "cached-scrape"
                    df_simple["source_tier"] = 1
                    df_simple["exact_data"] = df_simple["status"].apply(lambda s: str(s).lower() in {"ok", "cache"})
                    df_simple["confidence_notes"] = "loaded from macro_data.json"
                    rows.extend(df_simple.to_dict(orient="records"))
                    diagnostics.append(f"simple macro cache: loaded {len(df_simple)} row(s) from data/cache/latest_macro_data.json")
    except Exception as exc:
        diagnostics.append(f"simple macro cache load failed: {exc}")

    # Second pass: configured vendor endpoints.
    missing_for_search: Dict[str, str] = {}
    already_present = {str(row.get("indicator", "")).strip() for row in rows}
    for indicator, meta in SOURCE_REGISTRY.items():
        endpoint_secret = str(meta.get("endpoint_secret", ""))
        api_key_secret = str(meta.get("api_key_secret", ""))
        if indicator in already_present:
            continue
        if endpoint_secret and _get_secret(endpoint_secret):
            ok, text, detail = _safe_vendor_get(endpoint_secret, api_key_secret)
            diagnostics.append(f"{indicator}: {detail}")
            if ok:
                rows.append(
                    {
                        "date": _today_iso(),
                        "indicator": indicator,
                        "value": f"Vendor feed pull via {endpoint_secret}",
                        "unit": str(meta.get("source_type", "vendor feed")),
                        "status": "vendor-feed",
                        "score": 0.0,
                        "commentary": text[:2200],
                        "source": ", ".join(meta.get("priority_sources", [])),
                        "data_confidence": 85,
                        "confidence_weight": 1.0,
                        "scoring_allowed": True,
                        "confidence_label": "high",
                        "source_type": str(meta.get("source_type", "vendor feed")),
                        "source_tier": 1,
                        "exact_data": True,
                        "confidence_notes": "Configured vendor endpoint; assumed licensed/controlled feed.",
                    }
                )
                continue
        missing_for_search[indicator] = str(meta.get("fallback_query", ""))

    # Third pass: source-targeted Serper fallback for feeds without endpoints/cache.
    if missing_for_search:
        if is_serper_configured():
            serper_rows, serper_diag = fetch_serper_macro_feed(missing_for_search, max_results_per_query=max_results_per_query)
            diagnostics.extend(serper_diag)
            if serper_rows is not None and not serper_rows.empty:
                rows.extend(serper_rows.to_dict(orient="records"))
            # If Serper returned nothing (or failed), attempt HeadlineFeed if available
            serper_indicators = set(serper_rows["indicator"].unique()) if serper_rows is not None and not serper_rows.empty else set()
            still_missing = {k: v for k, v in missing_for_search.items() if k not in serper_indicators}
            if still_missing:
                headline_rows, headline_diag = fetch_headlinefeed_macro_feed(still_missing)
                diagnostics.extend(headline_diag)
                if headline_rows is not None and not headline_rows.empty:
                    rows.extend(headline_rows.to_dict(orient="records"))
        else:
            # HeadlineFeed headlines fallback when Serper is not configured
            headline_rows, headline_diag = fetch_headlinefeed_macro_feed(missing_for_search)
            diagnostics.extend(headline_diag)
            if headline_rows is not None and not headline_rows.empty:
                rows.extend(headline_rows.to_dict(orient="records"))
            # If still missing, log manual guidance
            for indicator, meta in SOURCE_REGISTRY.items():
                if indicator in missing_for_search and (headline_rows is None or headline_rows.empty):
                    diagnostics.append(
                        f"{indicator}: no cache row, vendor endpoint, SERPER_API_KEY, or HEADLINEFEED_TOKEN; using manual fallback only. "
                        f"Preferred sources: {', '.join(meta.get('priority_sources', []))}"
                    )

    out = pd.DataFrame(rows)
    return out, diagnostics


def fetch_serper_macro_feed(queries: Dict[str, str] | None = None, max_results_per_query: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch fresh web-search snippets through Serper and convert to macro rows.

    Requires SERPER_API_KEY. The returned rows are intentionally unscored; Groq
    should score them from -2 to +2 with confidence and rationale.
    """
    api_key = _get_secret("SERPER_API_KEY")
    diagnostics: List[str] = []
    columns = ["date", "indicator", "value", "unit", "status", "score", "commentary", "source", "data_confidence", "confidence_weight", "scoring_allowed", "confidence_label", "source_type", "source_tier", "exact_data", "confidence_notes"]
    if not api_key:
        return pd.DataFrame(columns=columns), ["SERPER_API_KEY is not configured. Add it to Streamlit secrets or Codespaces secrets for automatic web pull."]

    queries = queries or DEFAULT_SERPER_QUERIES
    rows = []
    endpoint = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    now = time.time()
    for indicator, query in queries.items():
        cache_entry = _SERPER_CACHE.get(indicator)
        if cache_entry and (now - cache_entry[0] < _CACHE_TTL_SECONDS):
            cached_df = cache_entry[1]
            diagnostics.append(f"{indicator}: Serper cache hit (<=1h)")
            rows.append(cached_df)
            continue
        try:
            response = requests.post(endpoint, headers=headers, json={"q": query, "num": max_results_per_query}, timeout=30)
            if response.status_code >= 400:
                diagnostics.append(f"{indicator}: Serper HTTP {response.status_code}")
                continue
            payload = response.json()
            organic = payload.get("organic", []) if isinstance(payload, dict) else []
            if not organic:
                diagnostics.append(f"{indicator}: no search results")
                continue
            snippets = []
            sources = []
            parsed_value = None
            parsed_date = None
            parsed_unit = None
            for item in organic[:max_results_per_query]:
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                link = str(item.get("link", "")).strip()
                text = f"{title}: {snippet}".strip(": ")
                if text:
                    snippets.append(text)
                    if parsed_value is None:
                        val, dt, unit = _parse_indicator_specific(indicator, text)
                        parsed_value = val if parsed_value is None else parsed_value
                        parsed_date = dt if parsed_date is None else parsed_date
                        parsed_unit = unit if parsed_unit is None else parsed_unit
                if link:
                    sources.append(link)

            if parsed_value is not None and parsed_date is not None:
                row = pd.DataFrame(
                    [
                        {
                            "date": parsed_date,
                            "indicator": indicator,
                            "value": parsed_value,
                            "unit": parsed_unit or INDICATOR_UNIT_HINTS.get(indicator, ""),
                            "status": "auto-feed",
                            "score": 0.0,
                            "commentary": " | ".join(snippets)[:1800],
                            "source": " | ".join(sources[:3]) if sources else "Serper web search",
                            "data_confidence": 75,
                            "confidence_weight": 0.55,
                            "scoring_allowed": True,
                            "confidence_label": "medium",
                            "source_type": "targeted_search_snippet",
                            "source_tier": 4,
                            "exact_data": True,
                            "confidence_notes": "Numeric/date parsed from search snippet; verify source; unit/date inferred.",
                        }
                    ],
                    columns=columns,
                )
            else:
                row = pd.DataFrame(
                    [
                        {
                            "date": _today_iso(),
                            "indicator": indicator,
                            "value": f"Auto web pull: {len(snippets)} result(s)",
                            "unit": "news/search",
                            "status": "auto-feed",
                            "score": 0.0,
                            "commentary": " | ".join(snippets)[:1800],
                            "source": " | ".join(sources[:3]) if sources else "Serper web search",
                            "data_confidence": 25,
                            "confidence_weight": 0.0,
                            "scoring_allowed": False,
                            "confidence_label": "commentary-only",
                            "source_type": "targeted_search_snippet",
                            "source_tier": 5,
                            "exact_data": False,
                            "confidence_notes": "Search snippets are context only; no regime score impact unless verified.",
                        }
                    ],
                    columns=columns,
                )
            rows.append(row)
            _SERPER_CACHE[indicator] = (now, row)
            diagnostics.append(f"{indicator}: pulled {len(snippets)} search result(s)")
        except Exception as exc:
            diagnostics.append(f"{indicator}: Serper request failed: {exc}")

    if not rows:
        return pd.DataFrame(columns=columns), diagnostics
    return pd.concat(rows, ignore_index=True), diagnostics


def fetch_headlinefeed_macro_feed(queries: Dict[str, str] | None = None, max_results: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch headlines via HeadlineFeed API as a fallback when Serper is unavailable.

    Requires HEADLINEFEED_TOKEN. Returns rows shaped like the manual macro table;
    commentary-first (no score impact) so Groq can extract/score.
    """
    token = _get_secret("HEADLINEFEED_TOKEN")
    diagnostics: List[str] = []
    columns = [
        "date",
        "indicator",
        "value",
        "unit",
        "status",
        "score",
        "commentary",
        "source",
        "data_confidence",
        "confidence_weight",
        "scoring_allowed",
        "confidence_label",
        "source_type",
        "source_tier",
        "exact_data",
        "confidence_notes",
    ]
    if not token:
        return pd.DataFrame(columns=columns), ["HEADLINEFEED_TOKEN is not configured; set it in secrets to enable headline fallback."]

    queries = queries or HEADLINEFEED_QUERIES
    endpoint = "https://api.headlinefeed.dev/api/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    rows = []
    now = time.time()
    for indicator, query in queries.items():
        cache_entry = _HEADLINE_CACHE.get(indicator)
        if cache_entry and (now - cache_entry[0] < _CACHE_TTL_SECONDS):
            cached_df = cache_entry[1]
            diagnostics.append(f"{indicator}: HeadlineFeed cache hit (<=1h)")
            rows.append(cached_df)
            continue
        try:
            response = None
            for attempt in range(2):
                try:
                    response = requests.post(endpoint, headers=headers, json={"query": query, "limit": max_results}, timeout=30)
                    if response.status_code < 500:
                        break
                    time.sleep(1.5 * (attempt + 1))
                except Exception:
                    if attempt == 1:
                        raise
            if response is None:
                diagnostics.append(f"{indicator}: HeadlineFeed request returned no response")
                continue
            if response.status_code >= 400:
                diagnostics.append(f"{indicator}: HeadlineFeed HTTP {response.status_code}")
                continue
            data = response.json() if response is not None else {}
            articles = data.get("articles", []) if isinstance(data, dict) else []
            if not articles:
                diagnostics.append(f"{indicator}: no HeadlineFeed results")
                continue
            snippets = []
            sources = []
            parsed_value = None
            parsed_date = None
            parsed_unit = None
            for art in articles[:max_results]:
                title = str(art.get("title", "")).strip()
                desc = str(art.get("description", "")).strip()
                url = str(art.get("url", "")).strip()
                text = f"{title}: {desc}".strip(": ")
                if text:
                    snippets.append(text)
                    if parsed_value is None:
                        val, dt, unit = _parse_indicator_specific(indicator, text)
                        parsed_value = val if parsed_value is None else parsed_value
                        parsed_date = dt if parsed_date is None else parsed_date
                        parsed_unit = unit if parsed_unit is None else parsed_unit
                if url:
                    sources.append(url)

            if parsed_value is not None and parsed_date is not None:
                row = pd.DataFrame(
                    [
                        {
                            "date": parsed_date,
                            "indicator": indicator,
                            "value": parsed_value,
                            "unit": parsed_unit or INDICATOR_UNIT_HINTS.get(indicator, ""),
                            "status": "auto-feed",
                            "score": 0.0,
                            "commentary": " | ".join(snippets)[:1800],
                            "source": " | ".join(sources[:3]) if sources else "HeadlineFeed",
                            "data_confidence": 70,
                            "confidence_weight": 0.5,
                            "scoring_allowed": True,
                            "confidence_label": "medium",
                            "source_type": "headline_snippet",
                            "source_tier": 4,
                            "exact_data": True,
                            "confidence_notes": "Numeric/date parsed from headline snippet; verify source; unit/date inferred.",
                        }
                    ],
                    columns=columns,
                )
            else:
                row = pd.DataFrame(
                    [
                        {
                            "date": _today_iso(),
                            "indicator": indicator,
                            "value": f"HeadlineFeed: {len(snippets)} article(s)",
                            "unit": "news/headlines",
                            "status": "auto-feed",
                            "score": 0.0,
                            "commentary": " | ".join(snippets)[:1800],
                            "source": " | ".join(sources[:3]) if sources else "HeadlineFeed",
                            "data_confidence": 25,
                            "confidence_weight": 0.0,
                            "scoring_allowed": False,
                            "confidence_label": "commentary-only",
                            "source_type": "headline_snippet",
                            "source_tier": 5,
                            "exact_data": False,
                            "confidence_notes": "Headline snippets are context only; no regime score impact unless verified.",
                        }
                    ],
                    columns=columns,
                )
            rows.append(row)
            _HEADLINE_CACHE[indicator] = (now, row)
            diagnostics.append(f"{indicator}: pulled {len(snippets)} HeadlineFeed article(s)")
        except Exception as exc:
            diagnostics.append(f"{indicator}: HeadlineFeed request failed: {exc}")

    if not rows:
        return pd.DataFrame(columns=columns), diagnostics
    return pd.concat(rows, ignore_index=True), diagnostics

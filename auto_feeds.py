"""Automatic data-feed helpers for MOIL Macro Radar.

These helpers fetch external data and create structured rows that the Groq AI
scorer can evaluate. Groq/Llama itself does not browse the web; the app must
supply facts/snippets first, then Groq scores them.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
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

DEFAULT_SERPER_QUERIES: Dict[str, str] = {
    indicator: str(meta["fallback_query"])
    for indicator, meta in SOURCE_REGISTRY.items()
}

NSE_ARCHIVE_CANDIDATES = [
    ("bulk", "https://nsearchives.nseindia.com/content/equities/bulk.csv"),
    ("block", "https://nsearchives.nseindia.com/content/equities/block.csv"),
]


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


def is_serper_configured() -> bool:
    return bool(_get_secret("SERPER_API_KEY"))


def is_any_vendor_feed_configured() -> bool:
    """Return True if any source-specific endpoint is configured."""
    for meta in SOURCE_REGISTRY.values():
        endpoint_secret = str(meta.get("endpoint_secret", ""))
        if endpoint_secret and _get_secret(endpoint_secret):
            return True
    return False


def source_provider_status() -> pd.DataFrame:
    """Expose data-source readiness to the UI without displaying secrets."""
    rows = []
    for indicator, meta in SOURCE_REGISTRY.items():
        endpoint_secret = str(meta.get("endpoint_secret", ""))
        api_key_secret = str(meta.get("api_key_secret", ""))
        endpoint_configured = bool(endpoint_secret and _get_secret(endpoint_secret))
        api_key_configured = bool(api_key_secret and _get_secret(api_key_secret))
        if endpoint_configured:
            mode = "vendor endpoint"
            status = "configured"
        elif is_serper_configured():
            mode = "targeted web search fallback"
            status = "fallback"
        else:
            mode = "manual only"
            status = "not configured"
        rows.append(
            {
                "indicator": indicator,
                "priority_sources": ", ".join(meta.get("priority_sources", [])),
                "source_type": meta.get("source_type", ""),
                "endpoint_secret": endpoint_secret,
                "endpoint_configured": endpoint_configured,
                "api_key_secret": api_key_secret,
                "api_key_configured": api_key_configured,
                "fallback_search": is_serper_configured(),
                "mode": mode,
                "status": status,
                "notes": meta.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _safe_request_get(url: str, timeout: int = 20) -> Tuple[bool, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            return False, "", f"HTTP {response.status_code} from {url}"
        text = response.text or ""
        if not text.strip():
            return False, "", f"Empty response from {url}"
        return True, text, f"OK {url} rows/text={len(text)}"
    except Exception as exc:
        return False, "", f"Request failed for {url}: {exc}"


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
    diagnostics: List[str] = []
    frames: List[pd.DataFrame] = []

    for deal_type, url in NSE_ARCHIVE_CANDIDATES:
        ok, text, detail = _safe_request_get(url)
        diagnostics.append(f"{deal_type}: {detail}")
        if not ok:
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
            std = _standardize_nse_deal_columns(df, deal_type)
            if not std.empty:
                frames.append(std)
        except Exception as exc:
            diagnostics.append(f"{deal_type}: CSV parse failed: {exc}")

    if not frames:
        return pd.DataFrame(), diagnostics

    all_deals = pd.concat(frames, ignore_index=True)
    matches = all_deals[all_deals["symbol"].astype(str).str.upper() == symbol].copy()
    if matches.empty:
        diagnostics.append(f"NSE archive loaded successfully, but no {symbol} bulk/block rows were present in the latest archive.")
    else:
        diagnostics.append(f"Found {len(matches)} {symbol} NSE bulk/block row(s).")
    return matches, diagnostics


def nse_deals_to_tracker_updates(deals: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "category", "item", "value", "impact", "confidence", "details", "source"]
    if deals is None or deals.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in deals.iterrows():
        side = str(row.get("buy_sell", "")).upper()
        impact = 1.0 if "BUY" in side else -1.0 if "SELL" in side else 0.0
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
    columns = ["date", "indicator", "value", "unit", "status", "score", "commentary", "source"]
    rows = []
    diagnostics: List[str] = []

    # First pass: scheduled backend cache. This is the preferred free/no-API
    # route because scraping happens once daily, outside user page loads.
    try:
        from backend_cache_loader import load_latest_macro_cache, macro_cache_to_manual_rows

        cached = load_latest_macro_cache()
        if cached.get("ok") is not False and (cached.get("moil") or cached.get("html_sources")):
            cached_rows = macro_cache_to_manual_rows(cached)
            if cached_rows is not None and not cached_rows.empty:
                rows.extend(cached_rows.to_dict(orient="records"))
                diagnostics.append(f"scheduled cache: loaded {len(cached_rows)} macro row(s) from data/cache/latest_macro_data.json")
        else:
            diagnostics.append(cached.get("error", "scheduled cache: no macro cache file found yet"))
    except Exception as exc:
        diagnostics.append(f"scheduled cache read failed: {exc}")

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
        else:
            for indicator, meta in SOURCE_REGISTRY.items():
                if indicator in missing_for_search:
                    diagnostics.append(
                        f"{indicator}: no cache row, vendor endpoint, or SERPER_API_KEY; using manual fallback only. "
                        f"Preferred sources: {', '.join(meta.get('priority_sources', []))}"
                    )

    return pd.DataFrame(rows, columns=columns), diagnostics


def fetch_serper_macro_feed(queries: Dict[str, str] | None = None, max_results_per_query: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch fresh web-search snippets through Serper and convert to macro rows.

    Requires SERPER_API_KEY. The returned rows are intentionally unscored; Groq
    should score them from -2 to +2 with confidence and rationale.
    """
    api_key = _get_secret("SERPER_API_KEY")
    diagnostics: List[str] = []
    columns = ["date", "indicator", "value", "unit", "status", "score", "commentary", "source"]
    if not api_key:
        return pd.DataFrame(columns=columns), ["SERPER_API_KEY is not configured. Add it to Streamlit secrets or Codespaces secrets for automatic web pull."]

    queries = queries or DEFAULT_SERPER_QUERIES
    rows = []
    endpoint = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    for indicator, query in queries.items():
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
            for item in organic[:max_results_per_query]:
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                link = str(item.get("link", "")).strip()
                if title or snippet:
                    snippets.append(f"{title}: {snippet}".strip(": "))
                if link:
                    sources.append(link)
            rows.append(
                {
                    "date": _today_iso(),
                    "indicator": indicator,
                    "value": f"Auto web pull: {len(snippets)} result(s)",
                    "unit": "news/search",
                    "status": "auto-feed",
                    "score": 0.0,
                    "commentary": " | ".join(snippets)[:1800],
                    "source": " | ".join(sources[:3]) if sources else "Serper web search",
                }
            )
            diagnostics.append(f"{indicator}: pulled {len(snippets)} search result(s)")
        except Exception as exc:
            diagnostics.append(f"{indicator}: Serper request failed: {exc}")

    return pd.DataFrame(rows, columns=columns), diagnostics

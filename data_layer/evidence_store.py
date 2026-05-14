from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .confidence_engine import (
    compute_data_confidence,
    confidence_weight,
    quality_label,
    scoring_allowed,
    source_tier_from_name,
)
from .macro_quality_rules import assess_macro_evidence_quality

CANONICAL_INDICATORS = [
    "Silico-Manganese Prices",
    "India Crude Steel Production",
    "China Steel Exports",
    "China Power / Industrial Stress",
    "MOIL Ore / EMD Price Circular",
]

BASE_MACRO_COLUMNS = ["date", "indicator", "value", "unit", "status", "score", "commentary", "source"]
EVIDENCE_COLUMNS = [
    "evidence_id",
    "indicator",
    "period",
    "value",
    "unit",
    "region",
    "source_name",
    "source_url",
    "source_type",
    "source_tier",
    "exact_data",
    "publication_date",
    "scraped_at",
    "parser_method",
    "parser_confidence",
    "cross_source_confirmed",
    "data_confidence",
    "confidence_weight",
    "scoring_allowed",
    "confidence_label",
    "confidence_notes",
    "raw_text",
]


def _now_iso() -> str:
    # IST is UTC + 5:30
    ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    return ist_time.isoformat()


def _today() -> str:
    # IST is UTC + 5:30
    ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    return ist_time.strftime("%Y-%m-%d")


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _stable_id(*parts: Any) -> str:
    import hashlib

    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def canonicalize_indicator(indicator: str = "", source_name: str = "", raw_text: str = "") -> str:
    text = f"{indicator} {source_name} {raw_text}".lower()
    if any(token in text for token in ["silico manganese", "silmn", "simn", "si mn", "ferro alloy", "ferroalloy"]):
        return "Silico-Manganese Prices"
    if "india" in text and "crude steel" in text:
        return "India Crude Steel Production"
    if "worldsteel" in text and "india" in text and "crude steel" in text:
        return "India Crude Steel Production"
    if "china" in text and "steel" in text and any(token in text for token in ["export", "customs", "trade"]):
        return "China Steel Exports"
    if any(token in text for token in ["electricity", "power generation", "industrial production", "cement", "power stress", "mining"]):
        return "China Power / Industrial Stress"
    if any(token in text for token in ["moil", "emd", "basic price", "manganese ore", "smgr"]):
        return "MOIL Ore / EMD Price Circular"
    return indicator or "Public macro table"


def _infer_exact_data(obs: Dict[str, Any]) -> bool:
    raw = _safe_str(obs.get("raw_text"))
    value = obs.get("value")
    method = _safe_str(obs.get("extraction_method"))
    source = _safe_str(obs.get("source_name") or obs.get("source"))
    if value is not None and _safe_str(value).strip() not in {"", "None", "nan"}:
        return True
    if any(x in method.lower() for x in ["pdfplumber_table", "official_csv"]):
        return True
    if re.search(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", raw) and any(u in raw.lower() for u in ["mt", "tonne", "ton", "inr", "rs", "%"]):
        return True
    return False


def _make_evidence_row(
    *,
    indicator: str,
    period: str | None = None,
    value: Any = None,
    unit: str | None = None,
    region: str | None = None,
    source_name: str = "",
    source_url: str = "",
    source_type: str = "public_html",
    source_tier: int | None = None,
    exact_data: bool = False,
    publication_date: Any = None,
    scraped_at: Any = None,
    parser_method: str = "",
    parser_confidence: float | None = None,
    cross_source_confirmed: bool = False,
    raw_text: str = "",
) -> Dict[str, Any]:
    canonical = canonicalize_indicator(indicator, source_name, raw_text)
    tier = source_tier or source_tier_from_name(source_name, source_type)
    scraped = _safe_str(scraped_at) or _now_iso()

    quality = assess_macro_evidence_quality(
        indicator=canonical,
        source_name=source_name,
        source_type=source_type,
        value=value,
        unit=unit,
        period=period,
        raw_text=raw_text,
        exact_data=exact_data,
        parser_method=parser_method,
    )
    unit = quality.get("normalized_unit") or unit
    exact_data = bool(quality.get("valid_exact_data", False))

    confidence = compute_data_confidence(
        source_tier=tier,
        source_name=source_name,
        source_type=source_type,
        exact_data=exact_data,
        value=value,
        unit=unit,
        period=period,
        publication_date=publication_date or period,
        scraped_at=scraped,
        parser_confidence=parser_confidence,
        extraction_method=parser_method,
        cross_source_confirmed=cross_source_confirmed,
    )
    data_conf = int(min(confidence["data_confidence"], quality.get("confidence_cap", 100)))
    row_scoring_allowed = bool(quality.get("quality_scoring_allowed", False)) and scoring_allowed(data_conf)
    notes = "; ".join([confidence["confidence_notes"], str(quality.get("quality_notes", ""))]).strip("; ")
    return {
        "evidence_id": _stable_id(canonical, period, value, source_name, source_url, raw_text[:120]),
        "indicator": canonical,
        "period": period,
        "value": value,
        "unit": unit,
        "region": region,
        "source_name": source_name,
        "source_url": source_url,
        "source_type": source_type,
        "source_tier": confidence["source_tier"],
        "exact_data": bool(exact_data),
        "publication_date": _safe_str(publication_date or period),
        "scraped_at": scraped,
        "parser_method": parser_method,
        "parser_confidence": parser_confidence if parser_confidence is not None else 0.5,
        "cross_source_confirmed": bool(cross_source_confirmed),
        "data_confidence": data_conf,
        "confidence_weight": confidence_weight(data_conf) if row_scoring_allowed else 0.0,
        "scoring_allowed": row_scoring_allowed,
        "confidence_label": quality_label(data_conf) if row_scoring_allowed else "commentary-only",
        "confidence_notes": notes,
        "raw_text": raw_text[:2500],
    }


def evidence_rows_from_macro_cache(cache: Dict[str, Any]) -> pd.DataFrame:
    """Convert scheduled scraper JSON into canonical evidence rows."""
    rows: List[Dict[str, Any]] = []
    # Simple list-based cache (macro_data.json list of records)
    if isinstance(cache, list):
        df = pd.DataFrame(cache)
        required = {"indicator", "value", "date"}
        if not required.issubset(df.columns):
            return pd.DataFrame(columns=EVIDENCE_COLUMNS)
        df = df.fillna("")
        for _, r in df.iterrows():
            indicator = str(r.get("indicator", "")).strip()
            if not indicator:
                continue
            rows.append(
                _make_evidence_row(
                    indicator=indicator,
                    period=r.get("date"),
                    value=r.get("value"),
                    unit=r.get("unit", ""),
                    region=r.get("geography", ""),
                    source_name=r.get("source_name") or r.get("source") or "cached scraper",
                    source_url=r.get("source_url", ""),
                    source_type="cached-scrape",
                    source_tier=1,
                    exact_data=str(r.get("status", "")).lower() in {"ok", "cache", "fallback"},
                    publication_date=r.get("date"),
                    scraped_at=r.get("retrieved_at_utc"),
                    parser_method="simple_cache",
                    parser_confidence=0.9,
                    cross_source_confirmed=False,
                    raw_text=r.get("notes", ""),
                )
            )
        if not rows:
            return pd.DataFrame(columns=EVIDENCE_COLUMNS)
        out = pd.DataFrame(rows)
        for col in EVIDENCE_COLUMNS:
            if col not in out.columns:
                out[col] = None
        return out[EVIDENCE_COLUMNS]

    if not isinstance(cache, dict) or not cache:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    # MOIL PDF circulars / filings.
    moil = cache.get("moil") or {}
    if isinstance(moil, dict):
        source_doc = moil.get("source_document") or {}
        source_url = _safe_str(source_doc.get("source_url") or moil.get("selected_pdf"))
        source_name = _safe_str(source_doc.get("source_name") or "MOIL PDF circular")
        doc_date = source_doc.get("document_date")
        for item in moil.get("price_observations") or []:
            raw = item.get("raw_text", "")
            rows.append(
                _make_evidence_row(
                    indicator="MOIL Ore / EMD Price Circular",
                    period=item.get("effective_date") or doc_date,
                    value=item.get("price_value"),
                    unit=item.get("unit") or "PMT",
                    region="India",
                    source_name=source_name,
                    source_url=item.get("source_url") or source_url,
                    source_type="official_pdf",
                    source_tier=1,
                    exact_data=item.get("price_value") is not None,
                    publication_date=item.get("effective_date") or doc_date,
                    scraped_at=item.get("observation_date") or cache.get("finished_at"),
                    parser_method=item.get("extraction_method", "pdfplumber"),
                    parser_confidence=item.get("confidence", 0.85),
                    cross_source_confirmed=False,
                    raw_text=raw,
                )
            )
        for item in moil.get("category_revisions") or []:
            rows.append(
                _make_evidence_row(
                    indicator="MOIL Ore / EMD Price Circular",
                    period=item.get("effective_date") or doc_date,
                    value=item.get("change_pct"),
                    unit="% revision",
                    region="India",
                    source_name=source_name,
                    source_url=item.get("source_url") or source_url,
                    source_type="official_pdf",
                    source_tier=1,
                    exact_data=item.get("change_pct") is not None,
                    publication_date=item.get("effective_date") or doc_date,
                    scraped_at=cache.get("finished_at"),
                    parser_method="pdfplumber_text_regex",
                    parser_confidence=item.get("confidence", 0.75),
                    raw_text=item.get("raw_text", ""),
                )
            )

    # HTML/public sources.
    for source_result in cache.get("html_sources", []) or []:
        source = source_result.get("source") or {}
        source_name = _safe_str(source.get("name") or source.get("url") or "scheduled public source")
        source_url = _safe_str(source.get("url"))
        raw_indicator = _safe_str(source.get("indicator") or "Public macro table")
        for obs in source_result.get("observations") or []:
            raw = _safe_str(obs.get("raw_text"))
            indicator = canonicalize_indicator(raw_indicator or obs.get("indicator", ""), source_name, raw)
            exact = _infer_exact_data(obs)
            source_type = "official_html" if any(x in source_name.lower() for x in ["nbs", "gacc", "customs", "worldsteel", "ministry", "jpc"]) else "industry_public_html"
            rows.append(
                _make_evidence_row(
                    indicator=indicator,
                    period=obs.get("period"),
                    value=obs.get("value"),
                    unit=obs.get("unit") or "public table/snippet",
                    region=obs.get("country") or source.get("country"),
                    source_name=source_name,
                    source_url=obs.get("source_url") or source_url,
                    source_type=source_type,
                    exact_data=exact,
                    publication_date=obs.get("period"),
                    scraped_at=obs.get("scrape_time") or cache.get("finished_at"),
                    parser_method=obs.get("extraction_method", "html_scraper"),
                    parser_confidence=obs.get("confidence", 0.5),
                    cross_source_confirmed=False,
                    raw_text=raw,
                )
            )

    if not rows:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    out = pd.DataFrame(rows)
    for col in EVIDENCE_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[EVIDENCE_COLUMNS]


def manual_rows_to_evidence(manual_rows: pd.DataFrame) -> pd.DataFrame:
    """Convert current manual/source rows into evidence rows.

    Manual rows can be high confidence only if they carry source metadata. Plain
    manual opinion rows remain medium/low confidence and are confidence-gated.
    """
    if manual_rows is None or manual_rows.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    rows = []
    for _, row in manual_rows.iterrows():
        indicator = _safe_str(row.get("indicator"))
        source = _safe_str(row.get("source", "Manual"))
        data_conf = _safe_float(row.get("data_confidence"), None)
        if data_conf is not None:
            # Preserve existing confidence metadata by constructing an evidence row then overriding.
            evidence = _make_evidence_row(
                indicator=indicator,
                period=row.get("date"),
                value=row.get("value"),
                unit=row.get("unit"),
                region=row.get("region"),
                source_name=source,
                source_url=row.get("source_url", ""),
                source_type=row.get("source_type", "manual_verified" if "manual" not in source.lower() else "manual"),
                source_tier=int(_safe_float(row.get("source_tier"), source_tier_from_name(source, "manual")) or 4),
                exact_data=str(row.get("exact_data", "false")).lower() in {"true", "1", "yes"},
                publication_date=row.get("date"),
                scraped_at=_now_iso(),
                parser_method=row.get("parser_method", "manual_row"),
                parser_confidence=row.get("parser_confidence", 0.6),
                raw_text=row.get("commentary", ""),
            )
            evidence["data_confidence"] = int(max(0, min(100, data_conf)))
            evidence["confidence_weight"] = confidence_weight(evidence["data_confidence"])
            evidence["scoring_allowed"] = scoring_allowed(evidence["data_confidence"])
            evidence["confidence_label"] = quality_label(evidence["data_confidence"])
            evidence["confidence_notes"] = row.get("confidence_notes", evidence["confidence_notes"])
            rows.append(evidence)
            continue
        rows.append(
            _make_evidence_row(
                indicator=indicator,
                period=row.get("date"),
                value=row.get("value"),
                unit=row.get("unit"),
                source_name=source,
                source_type="manual_verified" if any(x in source.lower() for x in ["argus", "jpc", "gacc", "nbs", "moil", "worldsteel", "steelmint", "bigmint"]) else "manual",
                exact_data=str(row.get("value", "")).strip() not in {"", "Quote pending", "Manual"},
                publication_date=row.get("date"),
                parser_method="manual_row",
                parser_confidence=0.65,
                raw_text=row.get("commentary", ""),
            )
        )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def evidence_to_source_readiness(evidence_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evidence_df = evidence_df.copy() if evidence_df is not None else pd.DataFrame(columns=EVIDENCE_COLUMNS)
    for indicator in CANONICAL_INDICATORS:
        subset = evidence_df[evidence_df["indicator"] == indicator] if not evidence_df.empty and "indicator" in evidence_df.columns else pd.DataFrame()
        if subset.empty:
            rows.append(
                {
                    "indicator": indicator,
                    "best_source_found": "None",
                    "source_tier": None,
                    "exact_data": False,
                    "latest_period": "",
                    "evidence_rows": 0,
                    "confidence": 0,
                    "confidence_label": "missing",
                    "confidence_weight": 0.0,
                    "scoring_allowed": False,
                    "mode": "manual only / missing",
                    "notes": "No qualified evidence row found in scheduled cache or upload.",
                }
            )
            continue
        subset = subset.sort_values(["data_confidence", "scraped_at"], ascending=[False, False])
        best = subset.iloc[0]
        rows.append(
            {
                "indicator": indicator,
                "best_source_found": best.get("source_name", ""),
                "source_tier": best.get("source_tier"),
                "exact_data": bool(best.get("exact_data", False)),
                "latest_period": best.get("period", ""),
                "evidence_rows": int(len(subset)),
                "confidence": int(best.get("data_confidence", 0)),
                "confidence_label": best.get("confidence_label", ""),
                "confidence_weight": float(best.get("confidence_weight", 0.0)),
                "scoring_allowed": bool(best.get("scoring_allowed", False)),
                "mode": "scheduled cache / verified evidence" if bool(best.get("scoring_allowed", False)) else "commentary only",
                "notes": best.get("confidence_notes", ""),
            }
        )
    return pd.DataFrame(rows)


def evidence_to_macro_rows(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Convert evidence rows into Groq-scoreable macro rows.

    Output extends the manual macro CSV schema with source quality metadata.
    """
    extended_cols = BASE_MACRO_COLUMNS + [
        "best_source",
        "source_tier",
        "source_type",
        "exact_data",
        "latest_period",
        "data_confidence",
        "confidence_weight",
        "scoring_allowed",
        "confidence_label",
        "confidence_notes",
        "evidence_count",
        "raw_evidence",
    ]
    if evidence_df is None or evidence_df.empty:
        return pd.DataFrame(columns=extended_cols)

    rows = []
    for indicator, subset in evidence_df.groupby("indicator"):
        subset = subset.sort_values(["data_confidence", "scraped_at"], ascending=[False, False])
        best = subset.iloc[0]
        facts = []
        for _, obs in subset.head(8).iterrows():
            facts.append(
                f"source={obs.get('source_name')}; period={obs.get('period')}; value={obs.get('value')} {obs.get('unit')}; confidence={obs.get('data_confidence')}; raw={_safe_str(obs.get('raw_text'))[:220]}"
            )
        rows.append(
            {
                "date": _today(),
                "indicator": indicator,
                "value": f"{len(subset)} evidence row(s); best confidence {int(best.get('data_confidence', 0))}/100",
                "unit": best.get("unit") or "evidence pack",
                "status": "evidence-qualified" if best.get("scoring_allowed") else "commentary-only",
                "score": 0.0,
                "commentary": " | ".join(facts)[:3500],
                "source": best.get("source_name") or "scheduled evidence cache",
                "best_source": best.get("source_name"),
                "source_tier": best.get("source_tier"),
                "source_type": best.get("source_type"),
                "exact_data": bool(best.get("exact_data", False)),
                "latest_period": best.get("period"),
                "data_confidence": int(best.get("data_confidence", 0)),
                "confidence_weight": float(best.get("confidence_weight", 0.0)),
                "scoring_allowed": bool(best.get("scoring_allowed", False)),
                "confidence_label": best.get("confidence_label"),
                "confidence_notes": best.get("confidence_notes"),
                "evidence_count": int(len(subset)),
                "raw_evidence": "\n---\n".join(facts)[:6000],
            }
        )
    return pd.DataFrame(rows, columns=extended_cols)


def load_evidence_from_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "evidence" in data:
            df = pd.DataFrame(data["evidence"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()
    for col in EVIDENCE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[EVIDENCE_COLUMNS]


def evidence_schema_columns() -> list[str]:
    return EVIDENCE_COLUMNS.copy()

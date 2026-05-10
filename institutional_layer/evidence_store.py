from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .quality_engine import compute_institutional_confidence, utc_now

CACHE_PATH = Path("data/cache/institutional_activity_latest.json")


def _as_float(value: Any) -> float | None:
    try:
        if value in [None, ""]:
            return None
        return float(value)
    except Exception:
        return None


def _hashable_key(*parts: Any) -> str:
    import hashlib
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8", errors="ignore")).hexdigest()[:24]


def load_institutional_cache(path: str | Path = CACHE_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"Institutional cache missing: {p}"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"Could not read institutional cache {p}: {exc}"}


def _record_group_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        key = str(r.get("trade_date") or r.get("period") or r.get("filing_date") or "") + "|" + str(r.get("client_name") or r.get("holder_category") or r.get("acquirer_name") or "") + "|" + str(r.get("quantity") or r.get("holding_pct") or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_institutional_evidence_rows(cache: dict[str, Any] | None = None, outstanding_shares: float | None = None) -> pd.DataFrame:
    """Return canonical institutional evidence rows with strict quality scores.

    Evidence can be displayed even when it is commentary-only. Only rows that
    are directional, exact enough, material and sufficiently fresh are allowed
    to affect the Smart Money Score.
    """
    cache = cache if cache is not None else load_institutional_cache()
    if outstanding_shares is None:
        try:
            outstanding_shares = float(os.getenv("MOIL_OUTSTANDING_SHARES", "") or 0) or None
        except Exception:
            outstanding_shares = None
    if not isinstance(cache, dict) or not cache.get("ok", False):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for deal in cache.get("institutional_deals", []) or []:
        row = dict(deal)
        row["evidence_type"] = "bulk_block_deal"
        row["event_date"] = row.get("trade_date")
        row["institution"] = row.get("client_name")
        row["holder_category"] = None
        row["delta_pct_points"] = None
        row["raw_impact"] = _raw_deal_impact(row)
        rows.append(row)

    for obs in cache.get("shareholding_observations", []) or []:
        row = dict(obs)
        row["evidence_type"] = "shareholding_delta"
        # Do not use scraped_at as the quarter/event date. If the source did not
        # expose a quarter, this row is evidence but not scoring-grade.
        row["event_date"] = row.get("period") or row.get("latest_period")
        row["institution"] = row.get("holder_category")
        row["transaction_type"] = "HOLDING_CHANGE"
        row["quantity"] = None
        row["price"] = None
        row["value_inr"] = None
        row["client_name"] = row.get("holder_category")
        row["raw_impact"] = _raw_holding_impact(row)
        rows.append(row)

    for filing in cache.get("sast_filings", []) or []:
        row = dict(filing)
        row["evidence_type"] = "sast_or_regulation_filing"
        row["event_date"] = row.get("filing_date")
        row["institution"] = row.get("acquirer_name")
        row["client_name"] = row.get("acquirer_name")
        row["transaction_type"] = "SAST_FILING"
        row["raw_impact"] = 0.0
        rows.append(row)

    counts = _record_group_counts(rows)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("trade_date") or row.get("period") or row.get("filing_date") or "") + "|" + str(row.get("client_name") or row.get("holder_category") or row.get("acquirer_name") or "") + "|" + str(row.get("quantity") or row.get("holding_pct") or "")
        quality = compute_institutional_confidence(row, outstanding_shares=outstanding_shares, duplicate_count=counts.get(key, 1))
        raw_impact = _as_float(row.get("raw_impact")) or 0.0
        effective_impact = raw_impact * float(quality.get("confidence_weight", 0.0)) if quality.get("scoring_allowed") else 0.0
        row.update(quality)
        row["effective_score"] = round(effective_impact, 4)
        row["effective_impact"] = round(effective_impact, 4)
        row["quality_timestamp"] = utc_now()
        row["evidence_id"] = row.get("observation_id") or _hashable_key(row.get("source_name"), row.get("event_date"), row.get("institution"), row.get("raw_text"))
        enriched.append(row)

    df = pd.DataFrame(enriched)
    if not df.empty:
        desired = [
            "evidence_id", "evidence_type", "source_name", "source_tier", "event_date", "institution",
            "entity_type", "transaction_type", "quantity", "price", "value_inr", "pct_equity", "holder_category",
            "holding_pct", "previous_holding_pct", "delta_pct_points", "raw_impact", "data_confidence",
            "confidence_weight", "effective_score", "effective_impact", "scoring_allowed", "confidence_bucket", "confidence_notes",
            "source_url", "raw_text",
        ]
        for col in desired:
            if col not in df.columns:
                df[col] = None
        df = df[desired + [c for c in df.columns if c not in desired]]
    return df


def _raw_deal_impact(row: dict[str, Any]) -> float:
    side = str(row.get("transaction_type") or "").upper()
    qty = _as_float(row.get("quantity"))
    value = _as_float(row.get("value_inr"))
    impact = 0.0
    if side == "BUY":
        impact = 0.5
    elif side == "SELL":
        impact = -0.5
    if qty and qty >= 500000:
        impact *= 1.5
    if value and value >= 100_000_000:
        impact *= 1.5
    return max(-2.0, min(2.0, impact))


def _raw_holding_impact(row: dict[str, Any]) -> float:
    delta = _as_float(row.get("delta_pct_points"))
    if delta is None:
        return 0.0
    if delta >= 2.0:
        return 2.0
    if delta >= 1.0:
        return 1.0
    if delta >= 0.25:
        return 0.5
    if delta <= -2.0:
        return -2.0
    if delta <= -1.0:
        return -1.0
    if delta <= -0.25:
        return -0.5
    return 0.0


def build_institutional_source_readiness(cache: dict[str, Any] | None = None) -> pd.DataFrame:
    cache = cache if cache is not None else load_institutional_cache()
    evidence = build_institutional_evidence_rows(cache)
    rows = []
    source_targets = [
        ("NSE Bulk/Block Archive", "official exchange CSV", "bulk/block deals"),
        ("Trendlyne MOIL Bulk/Block Deals", "public aggregator", "historical bulk/block deals"),
        ("MOIL Official Shareholding", "company official", "quarterly shareholding"),
        ("Screener MOIL", "public aggregator", "shareholding deltas"),
        ("NSE Corporate Filings Regulation 29 / SAST", "official exchange filings", "SAST/Regulation 29"),
    ]
    for source_name, source_type, coverage in source_targets:
        sub = evidence[evidence["source_name"].astype(str).str.lower().str.contains(source_name.lower().split()[0], na=False)] if not evidence.empty else pd.DataFrame()
        rows.append({
            "source_name": source_name,
            "source_type": source_type,
            "coverage": coverage,
            "rows_found": int(len(sub)),
            "avg_confidence": round(float(sub["data_confidence"].mean()), 1) if not sub.empty and "data_confidence" in sub else 0.0,
            "scoring_rows": int(sub["scoring_allowed"].sum()) if not sub.empty and "scoring_allowed" in sub else 0,
            "status": "active" if not sub.empty else "no rows / blocked / no current event",
            "notes": "Use official sources first; aggregator rows receive confidence haircuts.",
        })
    return pd.DataFrame(rows)


def build_smart_money_scorecard(cache: dict[str, Any] | None = None) -> pd.DataFrame:
    evidence = build_institutional_evidence_rows(cache)
    if evidence.empty:
        return pd.DataFrame(columns=["signal", "severity", "institution", "raw_impact", "data_confidence", "effective_impact", "description"])
    rows = []
    for _, row in evidence.iterrows():
        raw = _as_float(row.get("raw_impact")) or 0.0
        eff = _as_float(row.get("effective_impact")) or 0.0
        if raw == 0 or not bool(row.get("scoring_allowed")) or eff == 0:
            continue
        severity = "Confirmation" if eff >= 0.7 else "Watch" if eff > 0 else "High-Stress" if eff <= -1.2 else "Risk Warning" if eff < -0.4 else "Watch"
        action = str(row.get("transaction_type") or row.get("evidence_type") or "signal")
        inst = str(row.get("institution") or row.get("client_name") or row.get("holder_category") or "Unknown")
        rows.append({
            "signal": str(row.get("evidence_type", "institutional_evidence")),
            "severity": severity,
            "institution": inst,
            "raw_impact": round(raw, 2),
            "data_confidence": row.get("data_confidence"),
            "confidence_weight": row.get("confidence_weight"),
            "effective_score": round(eff, 2),
            "effective_impact": round(eff, 2),
            "description": f"{inst}: {action}; confidence {row.get('data_confidence')} / 100.",
            "source_name": row.get("source_name"),
            "event_date": row.get("event_date"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["effective_impact", "data_confidence"], ascending=[True, False])
    return df


def institutional_quality_summary(cache: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence = build_institutional_evidence_rows(cache)
    scorecard = build_smart_money_scorecard(cache)
    if evidence.empty:
        return {
            "evidence_rows": 0,
            "scoring_rows": 0,
            "average_confidence": 0.0,
            "net_effective_impact": 0.0,
            "institutional_tone": "No institutional evidence",
        }
    net = float(scorecard["effective_impact"].sum()) if not scorecard.empty else 0.0
    if net >= 1.0:
        tone = "Institutional confirmation"
    elif net <= -1.0:
        tone = "Institutional stress"
    elif net > 0.2:
        tone = "Mild accumulation / watch"
    elif net < -0.2:
        tone = "Mild distribution / watch"
    else:
        tone = "Neutral / no material institutional signal"
    return {
        "evidence_rows": int(len(evidence)),
        "scoring_rows": int(evidence["scoring_allowed"].sum()) if "scoring_allowed" in evidence else 0,
        "average_confidence": round(float(evidence["data_confidence"].mean()), 1) if "data_confidence" in evidence else 0.0,
        "net_effective_impact": round(net, 2),
        "institutional_tone": tone,
    }

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


SOURCE_TIER_RULES = {
    "nse bulk/block archive": {"tier": 1, "authority": 30, "parser": 14, "notes": "Official exchange archive CSV"},
    "nse corporate filings": {"tier": 1, "authority": 30, "parser": 12, "notes": "Official exchange disclosure page"},
    "nse sast": {"tier": 1, "authority": 30, "parser": 12, "notes": "Official exchange SAST/Regulation filing"},
    "bse": {"tier": 1, "authority": 28, "parser": 12, "notes": "Official exchange disclosure page"},
    "moil official shareholding": {"tier": 1, "authority": 30, "parser": 13, "notes": "Company official shareholding page"},
    "moil official": {"tier": 1, "authority": 30, "parser": 13, "notes": "Company official source"},
    "screener": {"tier": 3, "authority": 18, "parser": 10, "notes": "Public aggregator; useful but not primary"},
    "trendlyne": {"tier": 3, "authority": 18, "parser": 10, "notes": "Public aggregator; useful historical deal view"},
    "manual official upload": {"tier": 1, "authority": 28, "parser": 14, "notes": "Human-uploaded official file"},
    "manual aggregator upload": {"tier": 3, "authority": 16, "parser": 10, "notes": "Human-uploaded aggregator file"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def source_profile(source_name: str) -> dict[str, Any]:
    lower = _norm_text(source_name)
    for key, profile in SOURCE_TIER_RULES.items():
        if key in lower:
            return profile.copy()
    return {"tier": 4, "authority": 10, "parser": 7, "notes": "Unclassified public/inferred source"}


def parse_date(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            return None
        if getattr(ts, "tzinfo", None) is None:
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return ts.to_pydatetime().astimezone(timezone.utc)
    except Exception:
        return None


def freshness_score(event_date: Any = None, scraped_at: Any = None, evidence_type: str = "") -> tuple[int, str]:
    """Return freshness points out of 15 and explanatory note.

    Bulk/block/SAST events decay quickly; quarterly shareholding gets a longer
    acceptable window.
    """
    event_dt = parse_date(event_date) or parse_date(scraped_at)
    if not event_dt:
        return 6, "No parsable event date; partial freshness credit only."

    age_days = max(0, (datetime.now(timezone.utc) - event_dt).days)
    typ = _norm_text(evidence_type)

    if "shareholding" in typ:
        if age_days <= 100:
            return 15, f"Quarterly shareholding age {age_days}d."
        if age_days <= 190:
            return 10, f"Shareholding age {age_days}d; somewhat stale."
        if age_days <= 370:
            return 5, f"Shareholding age {age_days}d; stale."
        return 1, f"Shareholding age {age_days}d; very stale."

    if age_days <= 3:
        return 15, f"Event age {age_days}d."
    if age_days <= 15:
        return 11, f"Event age {age_days}d."
    if age_days <= 45:
        return 7, f"Event age {age_days}d; watch staleness."
    if age_days <= 120:
        return 3, f"Event age {age_days}d; stale."
    return 1, f"Event age {age_days}d; very stale."


def exactness_score(record: dict[str, Any]) -> tuple[int, str]:
    """Return exactness points out of 20."""
    typ = _norm_text(record.get("evidence_type"))

    if "bulk" in typ or "block" in typ or "deal" in typ:
        fields = [record.get("transaction_type"), record.get("quantity"), record.get("price"), record.get("trade_date"), record.get("client_name")]
        filled = sum(x not in [None, "", "UNKNOWN", "Unknown"] for x in fields)
        pts = int(round(20 * filled / len(fields)))
        return pts, f"Deal exact fields {filled}/{len(fields)}."

    if "shareholding" in typ:
        fields = [record.get("holder_category"), record.get("holding_pct"), record.get("previous_holding_pct"), record.get("delta_pct_points")]
        filled = sum(x not in [None, "", "UNKNOWN", "Unknown"] for x in fields)
        pts = int(round(20 * filled / len(fields)))
        return pts, f"Shareholding exact fields {filled}/{len(fields)}."

    if "sast" in typ or "regulation" in typ:
        fields = [record.get("acquirer_name") or record.get("client_name"), record.get("filing_date") or record.get("trade_date"), record.get("regulation"), record.get("raw_text")]
        filled = sum(x not in [None, "", "UNKNOWN", "Unknown"] for x in fields)
        pts = int(round(20 * filled / len(fields)))
        return pts, f"Filing exact fields {filled}/{len(fields)}."

    return 8, "Unclassified evidence exactness."


def materiality_score(record: dict[str, Any], outstanding_shares: float | None = None) -> tuple[int, float | None, str]:
    """Return materiality points out of 10 and pct_equity."""
    pct_equity = record.get("pct_equity")
    try:
        pct_equity = float(pct_equity) if pct_equity not in [None, ""] else None
    except Exception:
        pct_equity = None

    if pct_equity is None:
        qty = record.get("quantity")
        if outstanding_shares and qty:
            try:
                pct_equity = float(qty) / float(outstanding_shares) * 100
            except Exception:
                pct_equity = None

    delta = record.get("delta_pct_points")
    try:
        delta_abs = abs(float(delta)) if delta not in [None, ""] else None
    except Exception:
        delta_abs = None

    if pct_equity is not None:
        if pct_equity >= 1.0:
            return 10, pct_equity, f"Large deal materiality {pct_equity:.2f}% equity."
        if pct_equity >= 0.5:
            return 8, pct_equity, f"Bulk-threshold materiality {pct_equity:.2f}% equity."
        if pct_equity >= 0.1:
            return 5, pct_equity, f"Moderate deal materiality {pct_equity:.2f}% equity."
        return 2, pct_equity, f"Low deal materiality {pct_equity:.2f}% equity."

    if delta_abs is not None:
        if delta_abs >= 2.0:
            return 10, None, f"Large holding delta {delta_abs:.2f} ppt."
        if delta_abs >= 1.0:
            return 8, None, f"Material holding delta {delta_abs:.2f} ppt."
        if delta_abs >= 0.25:
            return 5, None, f"Moderate holding delta {delta_abs:.2f} ppt."
        return 2, None, f"Small holding delta {delta_abs:.2f} ppt."

    return 2, pct_equity, "Materiality unknown; limited credit."


def cross_source_score(record: dict[str, Any], duplicate_count: int = 1) -> tuple[int, str]:
    if duplicate_count >= 2:
        return 10, f"Cross-source confirmation count {duplicate_count}."
    if bool(record.get("cross_source_confirmed")):
        return 10, "Cross-source confirmed."
    return 0, "No independent cross-source confirmation."


def confidence_weight(score: float | int | None) -> float:
    try:
        score_f = float(score)
    except Exception:
        return 0.0
    if score_f >= 80:
        return 1.0
    if score_f >= 60:
        return 0.70
    if score_f >= 40:
        return 0.40
    return 0.0


def confidence_bucket(score: float | int | None) -> str:
    try:
        score_f = float(score)
    except Exception:
        return "unknown"
    if score_f >= 80:
        return "high"
    if score_f >= 60:
        return "medium-high"
    if score_f >= 40:
        return "medium-low"
    if score_f >= 20:
        return "low"
    return "ignore"


def compute_institutional_confidence(
    record: dict[str, Any],
    outstanding_shares: float | None = None,
    duplicate_count: int = 1,
) -> dict[str, Any]:
    """Compute confidence and scoring gate for institutional evidence.

    Score dimensions:
    - source authority: 0-30
    - exactness: 0-20
    - freshness: 0-15
    - parser reliability: 0-15
    - cross-source confirmation: 0-10
    - materiality: 0-10
    """
    profile = source_profile(str(record.get("source_name", "")))
    exact_pts, exact_note = exactness_score(record)
    fresh_pts, fresh_note = freshness_score(record.get("trade_date") or record.get("filing_date") or record.get("period"), record.get("scraped_at"), str(record.get("evidence_type", "")))
    material_pts, pct_equity, material_note = materiality_score(record, outstanding_shares=outstanding_shares)
    cross_pts, cross_note = cross_source_score(record, duplicate_count=duplicate_count)

    parser_conf = record.get("confidence", record.get("parser_confidence", 0.55))
    try:
        parser_pts = min(15, max(0, int(round(float(parser_conf) * 15))))
    except Exception:
        parser_pts = int(profile.get("parser", 7))

    total = int(max(0, min(100, profile["authority"] + exact_pts + fresh_pts + parser_pts + cross_pts + material_pts)))
    weight = confidence_weight(total)
    scoring_allowed = bool(weight > 0 and material_pts >= 2)

    return {
        "source_tier": profile["tier"],
        "source_authority_points": profile["authority"],
        "exactness_points": exact_pts,
        "freshness_points": fresh_pts,
        "parser_points": parser_pts,
        "cross_source_points": cross_pts,
        "materiality_points": material_pts,
        "pct_equity": pct_equity,
        "data_confidence": total,
        "confidence_bucket": confidence_bucket(total),
        "confidence_weight": weight,
        "scoring_allowed": scoring_allowed,
        "confidence_notes": "; ".join([profile.get("notes", ""), exact_note, fresh_note, cross_note, material_note]).strip("; "),
    }

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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

HFT_PROP_KEYWORDS = [
    "graviton",
    "research capital",
    "arbitrage",
    "quantitative",
    "prop",
    "proprietary",
    "market maker",
    "stock brokers",
    "securities pvt",
]



def utc_now() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).isoformat()



def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()



def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "nan", "nat", "null", "unknown"}



def source_profile(source_name: str) -> dict[str, Any]:
    lower = _norm_text(source_name)
    for key, profile in SOURCE_TIER_RULES.items():
        if key in lower:
            return profile.copy()
    return {"tier": 4, "authority": 10, "parser": 7, "notes": "Unclassified public/inferred source"}



def classify_entity_type(name: Any) -> str:
    text = _norm_text(name)
    if not text:
        return "Unknown"
    if any(x in text for x in ["government", "ministry", "president of india"]):
        return "Promoter / Government"
    if "lic" in text or "life insurance" in text or "insurance" in text:
        return "Insurance"
    if "mutual fund" in text or "asset management" in text or "amc" in text:
        return "Mutual Fund"
    if any(x in text for x in ["blackrock", "ishares", "vanguard", "fpi", "fii", "foreign"]):
        return "FII / FPI / ETF"
    if any(x in text for x in HFT_PROP_KEYWORDS):
        return "HFT / Prop / Market Structure"
    if any(x in text for x in ["securities", "brokers", "capital services"]):
        return "Broker / Prop"
    if any(x in text for x in ["fund", "capital", "investments"]):
        return "Fund / Investment Vehicle"
    if any(x in text for x in ["public", "retail"]):
        return "Public"
    if any(x in text for x in ["promoter"]):
        return "Promoter / Government"
    return "Unknown"



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



def freshness_score(event_date: Any = None, scraped_at: Any = None, evidence_type: str = "") -> tuple[int, str, int | None]:
    """Return freshness points out of 15, note and age_days.

    Bulk/block/SAST events decay quickly; quarterly shareholding gets a longer
    acceptable window. Scrape time is not a substitute for the event/quarter date.
    """
    event_dt = parse_date(event_date)
    if not event_dt:
        return 2, "No parsable event/quarter date; scrape timestamp is not evidence date.", None

    age_days = max(0, ((datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)) - event_dt).days)
    typ = _norm_text(evidence_type)

    if "shareholding" in typ:
        if age_days <= 120:
            return 15, f"Quarterly shareholding age {age_days}d.", age_days
        if age_days <= 220:
            return 9, f"Shareholding age {age_days}d; somewhat stale.", age_days
        if age_days <= 400:
            return 4, f"Shareholding age {age_days}d; stale.", age_days
        return 1, f"Shareholding age {age_days}d; very stale.", age_days

    if age_days <= 3:
        return 15, f"Event age {age_days}d.", age_days
    if age_days <= 15:
        return 11, f"Event age {age_days}d.", age_days
    if age_days <= 45:
        return 7, f"Event age {age_days}d; watch staleness.", age_days
    if age_days <= 120:
        return 3, f"Event age {age_days}d; stale.", age_days
    return 1, f"Event age {age_days}d; very stale.", age_days



def exactness_score(record: dict[str, Any]) -> tuple[int, str, bool]:
    """Return exactness points out of 20 and a boolean exactness gate."""
    typ = _norm_text(record.get("evidence_type"))

    if "bulk" in typ or "block" in typ or "deal" in typ:
        side = str(record.get("transaction_type") or "").upper()
        fields = [record.get("quantity"), record.get("price"), record.get("trade_date"), record.get("client_name")]
        filled = sum(not is_missing(x) for x in fields)
        side_ok = side in {"BUY", "SELL"}
        pts = int(round(18 * filled / len(fields))) + (2 if side_ok else 0)
        exact_gate = bool(side_ok and filled >= 4)
        note = f"Deal exact fields {filled}/4; side {'known' if side_ok else 'unknown'}."
        return pts, note, exact_gate

    if "shareholding" in typ:
        fields = [record.get("holder_category"), record.get("holding_pct"), record.get("previous_holding_pct"), record.get("delta_pct_points"), record.get("period")]
        filled = sum(not is_missing(x) for x in fields)
        exact_gate = bool(filled >= 4 and not is_missing(record.get("delta_pct_points")) and not is_missing(record.get("period")))
        pts = int(round(20 * filled / len(fields)))
        return pts, f"Shareholding exact fields {filled}/{len(fields)}.", exact_gate

    if "sast" in typ or "regulation" in typ:
        fields = [record.get("acquirer_name") or record.get("client_name"), record.get("filing_date") or record.get("trade_date"), record.get("regulation"), record.get("raw_text")]
        filled = sum(not is_missing(x) for x in fields)
        exact_gate = bool(filled >= 3)
        pts = int(round(20 * filled / len(fields)))
        return pts, f"Filing exact fields {filled}/{len(fields)}.", exact_gate

    return 6, "Unclassified evidence exactness.", False



def materiality_score(record: dict[str, Any], outstanding_shares: float | None = None) -> tuple[int, float | None, str, bool]:
    """Return materiality points out of 10, pct_equity, note, materiality gate."""
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
            return 10, pct_equity, f"Large deal materiality {pct_equity:.2f}% equity.", True
        if pct_equity >= 0.5:
            return 8, pct_equity, f"Bulk-threshold materiality {pct_equity:.2f}% equity.", True
        if pct_equity >= 0.1:
            return 5, pct_equity, f"Moderate deal materiality {pct_equity:.2f}% equity.", True
        return 2, pct_equity, f"Low deal materiality {pct_equity:.2f}% equity.", False

    if delta_abs is not None:
        if delta_abs >= 2.0:
            return 10, None, f"Large holding delta {delta_abs:.2f} ppt.", True
        if delta_abs >= 1.0:
            return 8, None, f"Material holding delta {delta_abs:.2f} ppt.", True
        if delta_abs >= 0.25:
            return 5, None, f"Moderate holding delta {delta_abs:.2f} ppt.", True
        return 2, None, f"Small holding delta {delta_abs:.2f} ppt.", False

    return 0, pct_equity, "Materiality unknown; commentary-only.", False



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
    if score_f >= 85:
        return 1.0
    if score_f >= 70:
        return 0.70
    if score_f >= 50:
        return 0.40
    return 0.0



def confidence_bucket(score: float | int | None) -> str:
    try:
        score_f = float(score)
    except Exception:
        return "unknown"
    if score_f >= 85:
        return "high"
    if score_f >= 70:
        return "medium-high"
    if score_f >= 50:
        return "medium-low"
    if score_f >= 35:
        return "commentary-only"
    return "ignore"



def compute_institutional_confidence(
    record: dict[str, Any],
    outstanding_shares: float | None = None,
    duplicate_count: int = 1,
) -> dict[str, Any]:
    """Compute evidence quality and stricter scoring gate for institutional rows.

    Rows can be displayed as evidence even when commentary-only, but they only
    move the Smart Money Score when the event is directional, material and
    sufficiently exact.
    """
    profile = source_profile(str(record.get("source_name", "")))
    exact_pts, exact_note, exact_gate = exactness_score(record)
    fresh_pts, fresh_note, age_days = freshness_score(
        record.get("trade_date") or record.get("filing_date") or record.get("period") or record.get("event_date"),
        record.get("scraped_at"),
        str(record.get("evidence_type", "")),
    )
    material_pts, pct_equity, material_note, material_gate = materiality_score(record, outstanding_shares=outstanding_shares)
    cross_pts, cross_note = cross_source_score(record, duplicate_count=duplicate_count)

    parser_conf = record.get("confidence", record.get("parser_confidence", 0.55))
    try:
        parser_pts = min(15, max(0, int(round(float(parser_conf) * 15))))
    except Exception:
        parser_pts = int(profile.get("parser", 7))

    total = int(max(0, min(100, profile["authority"] + exact_pts + fresh_pts + parser_pts + cross_pts + material_pts)))

    typ = _norm_text(record.get("evidence_type"))
    side = str(record.get("transaction_type") or "").upper()
    entity_type = classify_entity_type(record.get("client_name") or record.get("holder_category") or record.get("institution") or record.get("acquirer_name"))
    raw_impact = record.get("raw_impact")
    try:
        raw_impact_f = float(raw_impact)
    except Exception:
        raw_impact_f = 0.0

    # Hard gates. These deliberately stop stale/ambiguous aggregator rows from
    # inflating scoring_rows or regime impact.
    gate_notes: list[str] = []
    hard_gate = True
    if raw_impact_f == 0.0:
        hard_gate = False
        gate_notes.append("non-directional raw impact")
    if not exact_gate:
        hard_gate = False
        gate_notes.append("insufficient exact fields")
    if not material_gate:
        hard_gate = False
        gate_notes.append("below materiality threshold or unknown materiality")
    if "bulk" in typ or "block" in typ or "deal" in typ:
        if side not in {"BUY", "SELL"}:
            hard_gate = False
            gate_notes.append("bulk/block side unknown")
        if age_days is not None and age_days > 120:
            hard_gate = False
            gate_notes.append("bulk/block event too stale for scoring")
        if entity_type == "HFT / Prop / Market Structure" and side not in {"BUY", "SELL"}:
            hard_gate = False
            gate_notes.append("HFT/prop activity without direction is commentary-only")
    if "shareholding" in typ:
        if is_missing(record.get("period")):
            hard_gate = False
            gate_notes.append("shareholding period missing")
        if is_missing(record.get("delta_pct_points")):
            hard_gate = False
            gate_notes.append("shareholding delta missing")
    if total < 50:
        hard_gate = False
        gate_notes.append("confidence below scoring threshold")

    weight = confidence_weight(total) if hard_gate else 0.0
    scoring_allowed = bool(weight > 0)

    notes = [
        profile.get("notes", ""),
        exact_note,
        fresh_note,
        cross_note,
        material_note,
    ] + gate_notes

    return {
        "source_tier": profile["tier"],
        "source_authority_points": profile["authority"],
        "exactness_points": exact_pts,
        "freshness_points": fresh_pts,
        "parser_points": parser_pts,
        "cross_source_points": cross_pts,
        "materiality_points": material_pts,
        "pct_equity": pct_equity,
        "entity_type": entity_type,
        "data_confidence": total,
        "confidence_bucket": confidence_bucket(total),
        "confidence_weight": weight,
        "scoring_allowed": scoring_allowed,
        "confidence_notes": "; ".join([x for x in notes if x]).strip("; "),
    }

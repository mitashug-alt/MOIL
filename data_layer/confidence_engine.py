from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd


SOURCE_TIER_AUTHORITY_POINTS = {
    1: 30,  # official / licensed institutional feed
    2: 24,  # high-quality public industry source with visible data
    3: 16,  # public industry page / supplier update
    4: 8,   # supplier asking-price / generic public source
    5: 5,   # search snippet / weak evidence
}


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def freshness_points(publication_date: Any = None, scraped_at: Any = None, today: pd.Timestamp | None = None) -> tuple[int, str]:
    """Return 0-20 freshness points and a readable note."""
    today = today or (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=5, minutes=30)).normalize()
    parsed = _parse_date(publication_date) or _parse_date(scraped_at)
    if parsed is None:
        return 8, "freshness unknown"
    age_days = int(max((today - parsed.normalize()).days, 0))
    if age_days <= 7:
        return 20, f"fresh: {age_days}d old"
    if age_days <= 30:
        return 16, f"recent: {age_days}d old"
    if age_days <= 90:
        return 10, f"stale-watch: {age_days}d old"
    return 3, f"stale: {age_days}d old"


def source_tier_from_name(source_name: str = "", source_type: str = "") -> int:
    text = f"{source_name} {source_type}".lower()
    if any(x in text for x in ["ministry of steel", "jpc", "worldsteel", "world steel", "gacc", "customs", "nbs", "national bureau", "moil pdf", "moil ltd", "official"]):
        return 1
    if any(x in text for x in ["argus", "steelmint", "bigmint", "ferroalloynet", "mysteel", "s&p", "bloomberg"]):
        return 2
    if any(x in text for x in ["ofbusiness", "china data portal", "ceic", "macromicro"]):
        return 3
    if any(x in text for x in ["indiamart", "supplier", "asking"]):
        return 4
    if any(x in text for x in ["serper", "search", "snippet"]):
        return 5
    return 4


def exactness_points(exact_data: bool, value: Any = None, unit: str | None = None, period: str | None = None) -> tuple[int, str]:
    """Return 0-25 exactness points."""
    has_value = value is not None and str(value).strip() not in {"", "None", "nan"}
    has_unit = bool(str(unit or "").strip())
    has_period = bool(str(period or "").strip())
    if exact_data and has_value and has_unit and has_period:
        return 25, "exact value, unit and period"
    if exact_data and has_value and has_unit:
        return 21, "exact value and unit"
    if has_value and has_unit:
        return 16, "numeric value visible but incomplete metadata"
    if has_value:
        return 12, "numeric value visible but unit/period incomplete"
    return 5, "no exact numeric value extracted"


def parser_points(parser_confidence: float | int | None = None, extraction_method: str = "") -> tuple[int, str]:
    """Return 0-15 parser reliability points."""
    try:
        conf = float(parser_confidence)
        if conf > 1:
            conf = conf / 100
    except Exception:
        conf = 0.5
    method = str(extraction_method or "").lower()
    base = int(round(max(0, min(1, conf)) * 15))
    if any(x in method for x in ["pdfplumber_table", "official_csv", "pandas_read_html"]):
        base = max(base, 12)
    elif any(x in method for x in ["visible_text", "snippet", "search"]):
        base = min(base, 8)
    return int(max(0, min(15, base))), f"parser confidence {conf:.2f} via {extraction_method or 'unknown'}"


def compute_data_confidence(
    *,
    source_tier: int | None = None,
    source_name: str = "",
    source_type: str = "",
    exact_data: bool = False,
    value: Any = None,
    unit: str | None = None,
    period: str | None = None,
    publication_date: Any = None,
    scraped_at: Any = None,
    parser_confidence: float | int | None = None,
    extraction_method: str = "",
    cross_source_confirmed: bool = False,
) -> Dict[str, Any]:
    """Compute institutional data confidence from 0-100.

    Components:
    - source authority: 0-30
    - exactness: 0-25
    - freshness: 0-20
    - parser reliability: 0-15
    - cross-source check: 0-10
    """
    tier = int(source_tier or source_tier_from_name(source_name, source_type))
    tier = max(1, min(5, tier))
    authority = SOURCE_TIER_AUTHORITY_POINTS.get(tier, 8)
    exact, exact_note = exactness_points(exact_data, value=value, unit=unit, period=period)
    fresh, fresh_note = freshness_points(publication_date, scraped_at)
    parser, parser_note = parser_points(parser_confidence, extraction_method)
    cross = 10 if cross_source_confirmed else 0
    total = int(max(0, min(100, authority + exact + fresh + parser + cross)))
    return {
        "data_confidence": total,
        "source_tier": tier,
        "source_authority_points": authority,
        "data_exactness_points": exact,
        "freshness_points": fresh,
        "parser_reliability_points": parser,
        "cross_source_points": cross,
        "confidence_notes": "; ".join([exact_note, fresh_note, parser_note, "cross-source confirmed" if cross else "no cross-source check"]),
    }


def confidence_weight(data_confidence: float | int | None) -> float:
    try:
        c = float(data_confidence)
    except Exception:
        c = 0.0
    if c >= 80:
        return 1.0
    if c >= 60:
        return 0.70
    if c >= 40:
        return 0.40
    # 20-39 is commentary-only by policy; <20 ignored.
    return 0.0


def scoring_allowed(data_confidence: float | int | None) -> bool:
    return confidence_weight(data_confidence) > 0


def quality_label(data_confidence: float | int | None) -> str:
    try:
        c = float(data_confidence)
    except Exception:
        c = 0
    if c >= 80:
        return "high"
    if c >= 60:
        return "medium-high"
    if c >= 40:
        return "medium-low"
    if c >= 20:
        return "commentary-only"
    return "ignore"

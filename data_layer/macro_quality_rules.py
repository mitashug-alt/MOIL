from __future__ import annotations

import math
import re
from typing import Any, Dict

import pandas as pd


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip().lower()
    return text in {"", "none", "nan", "nat", "null"}


def _to_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        v = float(str(value).replace(",", ""))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _contains_any(text: str, tokens: list[str]) -> bool:
    lower = (text or "").lower()
    return any(t.lower() in lower for t in tokens)


def _has_period(period: Any, raw_text: str = "") -> bool:
    text = f"{period or ''} {raw_text or ''}"
    if str(period or "").strip():
        return True
    return bool(
        re.search(r"\b20\d{2}[-/](0?[1-9]|1[0-2])\b", text)
        or re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+20\d{2}\b", text, re.I)
    )


def _source_is_paid_or_assessed(source_name: str, source_type: str = "") -> bool:
    text = f"{source_name} {source_type}".lower()
    return any(x in text for x in ["argus", "steelmint", "bigmint", "ferroalloynet", "vendor", "paid", "manual_verified"])


def _source_is_official(source_name: str, source_type: str = "") -> bool:
    text = f"{source_name} {source_type}".lower()
    return any(x in text for x in ["official", "ministry", "jpc", "worldsteel", "world steel", "gacc", "customs", "nbs", "national bureau", "moil pdf", "moil ltd"])


def _cap_for_source(source_name: str, source_type: str, default: int) -> int:
    text = f"{source_name} {source_type}".lower()
    if any(x in text for x in ["argus", "steelmint", "ferroalloynet", "vendor", "paid", "manual_verified"]):
        return min(default, 95)
    if "bigmint" in text:
        return min(default, 72)
    if any(x in text for x in ["worldsteel", "ministry", "jpc", "gacc", "customs", "nbs", "moil pdf", "official"]):
        return min(default, 92)
    if any(x in text for x in ["china data portal"]):
        return min(default, 68)
    if any(x in text for x in ["ofbusiness", "indiamart", "supplier", "asking"]):
        return min(default, 55)
    return min(default, 60)


def assess_macro_evidence_quality(
    *,
    indicator: str,
    source_name: str = "",
    source_type: str = "",
    value: Any = None,
    unit: str | None = None,
    period: Any = None,
    raw_text: str = "",
    exact_data: bool = False,
    parser_method: str = "",
) -> Dict[str, Any]:
    """Institutional-quality validation for macro evidence rows.

    This function is intentionally stricter than generic scraping. A page can be
    useful as context while still being too weak to affect the regime score.
    """
    ind = (indicator or "").strip()
    src = source_name or ""
    stype = source_type or ""
    raw = raw_text or ""
    raw_l = raw.lower()
    v = _to_float(value)
    notes: list[str] = []
    normalized_unit = unit
    valid_exact = bool(exact_data and v is not None)
    confidence_cap: int | None = None
    scoring_ok = True

    if v is None:
        valid_exact = False
        scoring_ok = False
        confidence_cap = 35
        notes.append("no validated numeric value; commentary-only")
    if not _has_period(period, raw):
        # Period-less data can be fresh because of scraped_at, but it is not exact enough for strong scoring.
        confidence_cap = min(confidence_cap or 100, 55)
        notes.append("no explicit period/date in evidence")

    if ind == "Silico-Manganese Prices":
        has_product = _contains_any(raw_l, ["silico manganese", "silmn", "simn", "si mn", "60-14", "hc 60"])
        has_region_or_grade = _contains_any(raw_l, ["raipur", "durgapur", "vizag", "raigarh", "60-14", "65-17", "exw", "ex-works"])
        price_like = False
        if v is not None:
            if 25000 <= v <= 150000:
                price_like = True
                normalized_unit = normalized_unit or "INR/t"
            elif 25 <= v <= 150:
                price_like = True
                normalized_unit = normalized_unit or "INR/kg"
            else:
                notes.append(f"numeric value {v:g} is outside plausible SiMn price ranges")
        if not (has_product and price_like):
            valid_exact = False
            scoring_ok = False
            confidence_cap = min(confidence_cap or 100, 39)
            notes.append("SiMn evidence lacks validated product/price range")
        elif not has_region_or_grade:
            confidence_cap = min(confidence_cap or 100, 58)
            notes.append("SiMn price lacks region/grade confirmation")
        if not _source_is_paid_or_assessed(src, stype):
            # OfBusiness/IndiaMART-style supplier pages can be context but should not move the regime alone.
            confidence_cap = min(confidence_cap or 100, 55)
            if any(x in f"{src} {stype}".lower() for x in ["ofbusiness", "indiamart", "supplier", "asking"]):
                scoring_ok = False
                notes.append("supplier/asking-price proxy; require assessed price source for scoring")

    elif ind == "India Crude Steel Production":
        has_india = "india" in raw_l or "india" in src.lower()
        has_crude = "crude steel" in raw_l or "crude steel" in src.lower()
        plausible = v is not None and ((5 <= v <= 20) or (-20 <= v <= 25))
        if not (has_india and has_crude and plausible):
            valid_exact = False
            scoring_ok = False
            confidence_cap = min(confidence_cap or 100, 39)
            notes.append("India crude steel evidence lacks official exact production/YoY value")
        if not _source_is_official(src, stype):
            confidence_cap = min(confidence_cap or 100, 65)
            notes.append("India steel source is not official/JPC/Worldsteel")

    elif ind == "China Steel Exports":
        has_china = "china" in raw_l or "china" in src.lower()
        has_steel_export = _contains_any(raw_l + " " + src.lower(), ["steel export", "steel products", "customs", "exports"])
        plausible = v is not None and ((1 <= v <= 25) or (1000 <= v <= 25000) or (-50 <= v <= 80))
        has_unit_or_period = _has_period(period, raw) and _contains_any(raw_l, ["mt", "ton", "tonne", "%", "yoy", "monthly", "customs", "export"])
        if not (has_china and has_steel_export and plausible and has_unit_or_period):
            valid_exact = False
            scoring_ok = False
            confidence_cap = min(confidence_cap or 100, 39)
            notes.append("China steel export evidence lacks validated volume/unit/period")
        if "china data portal" in src.lower():
            # China Data Portal is useful context, but generic scraped rows often
            # contain navigation/statistic numbers that are not a steel-export
            # volume. Treat it as commentary-only unless the row itself clearly
            # carries an exact steel export volume with a period.
            volume_exact = _contains_any(raw_l, ["million tonnes", "million tons", "mt", "tonnes", "tons"]) and _has_period(period, raw)
            confidence_cap = min(confidence_cap or 100, 58)
            if not volume_exact:
                scoring_ok = False
                notes.append("China Data Portal row lacks exact volume/period; commentary-only")
            else:
                notes.append("China Data Portal public proxy; prefer GACC official rows")
        elif not _source_is_official(src, stype):
            confidence_cap = min(confidence_cap or 100, 60)

    elif ind == "China Power / Industrial Stress":
        has_proxy = _contains_any(raw_l + " " + src.lower(), ["electricity", "power generation", "industrial production", "cement", "mining", "nbs"])
        plausible = v is not None and (-30 <= v <= 50 or 100 <= v <= 10000)
        if not (has_proxy and plausible):
            valid_exact = False
            scoring_ok = False
            confidence_cap = min(confidence_cap or 100, 39)
            notes.append("China power/industrial proxy lacks validated NBS-style value")
        if not _source_is_official(src, stype):
            confidence_cap = min(confidence_cap or 100, 60)
        else:
            confidence_cap = min(confidence_cap or 100, 82)
            notes.append("proxy indicator; not direct real-time power stress")

    elif ind == "MOIL Ore / EMD Price Circular":
        if v is None:
            valid_exact = False
            scoring_ok = False
            confidence_cap = min(confidence_cap or 100, 39)
            notes.append("MOIL circular evidence has no extracted numeric price/revision")
        if not _source_is_official(src, stype):
            confidence_cap = min(confidence_cap or 100, 60)
            notes.append("MOIL price evidence is not from official PDF/disclosure")

    # Generic invalid rows with visible_text and no number should never score.
    if "visible_text" in (parser_method or "").lower() and not valid_exact:
        confidence_cap = min(confidence_cap or 100, 35)
        scoring_ok = False

    if not valid_exact:
        scoring_ok = False

    if confidence_cap is None:
        confidence_cap = _cap_for_source(src, stype, 100)

    return {
        "valid_exact_data": bool(valid_exact),
        "normalized_unit": normalized_unit,
        "confidence_cap": int(confidence_cap),
        "quality_scoring_allowed": bool(scoring_ok),
        "quality_notes": "; ".join(dict.fromkeys(notes)) or "macro evidence passed strict validation",
    }

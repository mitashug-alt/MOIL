from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

from .confidence_engine import confidence_weight, scoring_allowed


def validate_evidence_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate evidence rows and return warnings, never raise for UI usage."""
    warnings: List[str] = []
    if df is None or df.empty:
        return pd.DataFrame(), ["No evidence rows available."]
    out = df.copy()
    required = ["indicator", "source_name", "data_confidence", "raw_text"]
    for col in required:
        if col not in out.columns:
            out[col] = "" if col != "data_confidence" else 0
            warnings.append(f"Missing evidence column added with default: {col}")
    out["data_confidence"] = pd.to_numeric(out["data_confidence"], errors="coerce").fillna(0).clip(0, 100)
    existing_allowed = out.get("scoring_allowed")
    out["confidence_weight"] = out["data_confidence"].apply(confidence_weight)
    out["scoring_allowed"] = out["data_confidence"].apply(scoring_allowed)
    if existing_allowed is not None:
        out["scoring_allowed"] = out["scoring_allowed"] & existing_allowed.astype(str).str.lower().isin(["true", "1", "yes"])
        out.loc[~out["scoring_allowed"], "confidence_weight"] = 0.0
    weak = out[out["data_confidence"] < 40]
    if not weak.empty:
        warnings.append(f"{len(weak)} evidence row(s) below scoring threshold; commentary-only.")
    if "exact_data" in out.columns:
        no_exact = out[out["exact_data"].astype(str).str.lower().isin(["false", "0", "", "none", "nan"])]
        if not no_exact.empty:
            warnings.append(f"{len(no_exact)} evidence row(s) lack exact values; confidence haircut applied.")
    return out, warnings


def validate_groq_macro_score(item: Dict[str, Any]) -> Dict[str, Any]:
    indicator = str(item.get("indicator", "")).strip()
    signal_score = item.get("signal_score", item.get("score", 0))
    try:
        signal_score = float(signal_score)
    except Exception:
        signal_score = 0.0
    signal_score = max(-2.0, min(2.0, signal_score))
    try:
        data_confidence = float(item.get("data_confidence", item.get("confidence", 0.5) * 100))
    except Exception:
        data_confidence = 50.0
    data_confidence = max(0.0, min(100.0, data_confidence))
    declared_allowed = item.get("scoring_allowed", True)
    if isinstance(declared_allowed, str):
        declared_allowed = declared_allowed.lower() in {"true", "1", "yes"}
    weight = confidence_weight(data_confidence) if bool(declared_allowed) else 0.0
    try:
        effective_score = float(item.get("effective_score", signal_score * weight))
    except Exception:
        effective_score = signal_score * weight
    effective_score = max(-2.0, min(2.0, effective_score))
    return {
        "indicator": indicator,
        "score": signal_score,
        "signal_score": signal_score,
        "data_confidence": data_confidence,
        "confidence_weight": weight,
        "effective_score": effective_score,
        "status": str(item.get("status", "neutral")).strip() or "neutral",
        "rationale": str(item.get("rationale", "")).strip(),
        "confidence": max(0.0, min(1.0, data_confidence / 100.0)),
        "data_quality": str(item.get("data_quality", "medium")).strip() or "medium",
        "scoring_allowed": bool(weight > 0),
    }

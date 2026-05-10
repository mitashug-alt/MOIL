"""Data-quality v2 layer for MOIL Macro Radar.

This package separates source evidence, confidence scoring and regime scoring.
The dashboard should score only confidence-qualified evidence rows.
"""

from .confidence_engine import confidence_weight, compute_data_confidence, scoring_allowed
from .evidence_store import (
    CANONICAL_INDICATORS,
    canonicalize_indicator,
    evidence_rows_from_macro_cache,
    evidence_to_macro_rows,
    evidence_to_source_readiness,
    manual_rows_to_evidence,
)

__all__ = [
    "CANONICAL_INDICATORS",
    "canonicalize_indicator",
    "confidence_weight",
    "compute_data_confidence",
    "scoring_allowed",
    "evidence_rows_from_macro_cache",
    "evidence_to_macro_rows",
    "evidence_to_source_readiness",
    "manual_rows_to_evidence",
]

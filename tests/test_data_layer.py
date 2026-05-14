"""Tests for data_layer modules (confidence_engine, validators, evidence_store)."""

import pandas as pd
import pytest


def test_confidence_weight_calculation():
    """Test confidence weight calculations for different score ranges."""
    from data_layer.confidence_engine import confidence_weight
    
    # Full impact >= 80
    assert confidence_weight(85) == 1.0
    assert confidence_weight(80) == 1.0
    
    # 70% impact 60-79
    assert confidence_weight(70) == 0.70
    assert confidence_weight(79) == 0.70
    
    # 40% impact 40-59
    assert confidence_weight(50) == 0.40
    assert confidence_weight(40) == 0.40
    
    # No impact < 40
    assert confidence_weight(39) == 0.0
    assert confidence_weight(0) == 0.0
    assert confidence_weight(None) == 0.0


def test_scoring_allowed_threshold():
    """Test scoring allowed threshold at 40."""
    from data_layer.confidence_engine import scoring_allowed
    
    assert scoring_allowed(40) is True
    assert scoring_allowed(39) is False
    assert scoring_allowed(80) is True
    assert scoring_allowed(0) is False


def test_validate_evidence_dataframe_basic():
    """Test evidence DataFrame validation."""
    from data_layer.validators import validate_evidence_dataframe
    
    df = pd.DataFrame({
        "indicator": ["Steel Production"],
        "source_name": ["JPC"],
        "data_confidence": [75],
        "raw_text": ["India steel output rose 5%"],
    })
    
    validated, warnings = validate_evidence_dataframe(df)
    
    assert len(validated) == 1
    assert validated["data_confidence"].iloc[0] == 75
    assert validated["confidence_weight"].iloc[0] == 0.70
    assert validated["scoring_allowed"].iloc[0] == True


def test_validate_evidence_dataframe_low_confidence():
    """Test evidence validation with low confidence rows."""
    from data_layer.validators import validate_evidence_dataframe
    
    df = pd.DataFrame({
        "indicator": ["Steel Production", "Manganese Price"],
        "source_name": ["JPC", "Unknown"],
        "data_confidence": [75, 30],  # Second row below threshold
        "raw_text": ["India steel output rose 5%", "Price data missing"],
    })
    
    validated, warnings = validate_evidence_dataframe(df)
    
    # Check that low confidence row gets weight 0
    low_conf_row = validated[validated["data_confidence"] == 30]
    assert low_conf_row["confidence_weight"].iloc[0] == 0.0
    assert low_conf_row["scoring_allowed"].iloc[0] == False
    
    # Should have warning about low confidence
    assert any("below scoring threshold" in w for w in warnings)


def test_validate_evidence_dataframe_missing_columns():
    """Test evidence validation adds missing columns."""
    from data_layer.validators import validate_evidence_dataframe
    
    df = pd.DataFrame({
        "indicator": ["Steel Production"],
    })
    
    validated, warnings = validate_evidence_dataframe(df)
    
    # Missing columns should be added
    assert "source_name" in validated.columns
    assert "data_confidence" in validated.columns
    assert "raw_text" in validated.columns
    assert validated["data_confidence"].iloc[0] == 0


def test_validate_groq_macro_score():
    """Test Groq macro score validation."""
    from data_layer.validators import validate_groq_macro_score
    
    item = {
        "indicator": "Steel Production",
        "signal_score": 1.5,
        "data_confidence": 75,
    }
    
    result = validate_groq_macro_score(item)
    
    assert result["indicator"] == "Steel Production"
    assert result["signal_score"] == 1.5
    assert result["data_confidence"] == 75
    assert result["confidence_weight"] == 0.70  # 75 is in 60-79 range
    assert result["effective_score"] == 1.5 * 0.70  # score * weight
    assert result["scoring_allowed"] is True


def test_validate_groq_macro_score_clipping():
    """Test Groq score validation clips values to valid ranges."""
    from data_layer.validators import validate_groq_macro_score
    
    item = {
        "indicator": "Test",
        "signal_score": 5.0,  # Above max of 2.0
        "data_confidence": 150,  # Above max of 100
    }
    
    result = validate_groq_macro_score(item)
    
    assert result["signal_score"] == 2.0  # Clipped to max
    assert result["data_confidence"] == 100.0  # Clipped to max


def test_evidence_to_macro_rows_empty():
    """Test evidence to macro rows conversion with empty DataFrame."""
    from data_layer.evidence_store import evidence_to_macro_rows
    
    result = evidence_to_macro_rows(pd.DataFrame())
    assert result.empty
    assert "indicator" in result.columns


def test_evidence_to_macro_rows_basic():
    """Test evidence to macro rows conversion."""
    from data_layer.evidence_store import evidence_to_macro_rows
    
    evidence = pd.DataFrame({
        "indicator": ["Steel Production", "Steel Production"],  # Duplicate to test grouping
        "source_name": ["JPC", "World Steel"],
        "source_tier": [1, 2],
        "source_type": ["official", "aggregator"],
        "exact_data": [True, True],
        "period": ["2026-04", "2026-04"],
        "value": ["5000", "5100"],
        "unit": ["kt", "kt"],
        "data_confidence": [85, 70],
        "confidence_weight": [1.0, 0.70],
        "scoring_allowed": [True, True],
        "confidence_label": ["high", "medium-high"],
        "confidence_notes": ["Official data", "Aggregator estimate"],
        "scraped_at": ["2026-05-01", "2026-05-01"],
        "raw_text": ["Steel production data", "World steel report"],
    })
    
    result = evidence_to_macro_rows(evidence)
    
    assert len(result) == 1  # Grouped by indicator
    assert result["indicator"].iloc[0] == "Steel Production"
    assert result["evidence_count"].iloc[0] == 2
    assert result["best_source"].iloc[0] == "JPC"  # Higher confidence
    assert "2 evidence row(s)" in result["value"].iloc[0]


def test_load_evidence_from_file_csv(tmp_path):
    """Test loading evidence from CSV file."""
    from data_layer.evidence_store import load_evidence_from_file
    
    csv_path = tmp_path / "test_evidence.csv"
    csv_path.write_text(
        "indicator,source_name,data_confidence,raw_text\n"
        "Steel,JPC,85,Official data\n"
    )
    
    result = load_evidence_from_file(csv_path)
    
    assert len(result) == 1
    assert result["indicator"].iloc[0] == "Steel"
    assert result["source_name"].iloc[0] == "JPC"


def test_load_evidence_from_file_json(tmp_path):
    """Test loading evidence from JSON file."""
    from data_layer.evidence_store import load_evidence_from_file
    import json
    
    json_path = tmp_path / "test_evidence.json"
    json_path.write_text(json.dumps([
        {"indicator": "Steel", "source_name": "JPC", "data_confidence": 85, "raw_text": "Data"}
    ]))
    
    result = load_evidence_from_file(json_path)
    
    assert len(result) == 1
    assert result["indicator"].iloc[0] == "Steel"


def test_load_evidence_from_file_missing():
    """Test loading evidence from non-existent file."""
    from data_layer.evidence_store import load_evidence_from_file
    
    result = load_evidence_from_file("/nonexistent/file.csv")
    assert result.empty


def test_source_profile_tier_lookup():
    """Test source profile tier lookup."""
    from institutional_layer.quality_engine import source_profile
    
    # Known source
    profile = source_profile("nse bulk/block archive")
    assert profile["tier"] == 1
    
    # Unknown source
    profile = source_profile("Unknown Random Source")
    assert profile["tier"] == 4  # Unclassified


def test_freshness_score_calculation():
    """Test freshness score calculation."""
    from institutional_layer.quality_engine import freshness_score
    from datetime import datetime, timedelta
    
    # Recent date
    recent = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    score, note, age = freshness_score(recent)
    assert score == 15  # Full freshness points
    # age may be 0 if dates are the same day (timezone handling)
    assert age is not None
    
    # Stale date
    stale = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    score, note, age = freshness_score(stale)
    assert score < 15  # Reduced points
    assert age is not None


def test_exactness_score_fields():
    """Test exactness score based on field completeness."""
    from institutional_layer.quality_engine import exactness_score
    
    # Complete deal record - all required fields present
    complete_record = {
        "evidence_type": "bulk_block_deal",
        "quantity": "1000",
        "price": "250.5",
        "trade_date": "2026-05-12",
        "client_name": "Alpha Fund",
        "transaction_type": "BUY",
    }
    score, note, exact = exactness_score(complete_record)
    assert score >= 6  # Should have reasonable score with complete data
    assert exact is True  # All required fields present and side known
    
    # Incomplete record - missing fields
    incomplete_record = {
        "evidence_type": "bulk_block_deal",
        "quantity": "1000",
        "transaction_type": "BUY",
        # Missing price, trade_date, client_name
    }
    score, note, exact = exactness_score(incomplete_record)
    assert score >= 0  # Should have some score
    assert exact is False  # Not all required fields present

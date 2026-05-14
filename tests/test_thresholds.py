import pandas as pd

from institutional_flow.config import TrackerConfig, tracker_config


def test_thresholds_math():
    cfg = tracker_config()
    assert cfg.bulk_threshold_shares == 1_017_427
    assert cfg.named_holder_threshold_shares == 2_034_853
    assert cfg.sast_five_percent_shares == 10_174_261
    assert cfg.sast_two_percent_shares == 4_069_705
    # percent helpers
    assert round(cfg.bulk_threshold_pct, 4) == round(0.5, 4)
    assert round(cfg.named_holder_threshold_pct, 4) == round(1.0, 4)
    assert round(cfg.sast_five_percent_pct, 4) == round(5.0, 4)
    assert round(cfg.sast_two_percent_pct, 4) == round(2.0, 4)


def test_config_override():
    cfg = tracker_config({"symbol": "TEST", "outstanding_shares": 100})
    assert cfg.symbol == "TEST"
    assert cfg.outstanding_shares == 100


def test_zero_outstanding_shares_no_division_by_zero():
    """Regression test: zero outstanding_shares should not cause ZeroDivisionError."""
    cfg = tracker_config({"outstanding_shares": 0})
    # All percentage properties should return 0.0, not raise
    assert cfg.bulk_threshold_pct == 0.0
    assert cfg.named_holder_threshold_pct == 0.0
    assert cfg.sast_five_percent_pct == 0.0
    assert cfg.sast_two_percent_pct == 0.0

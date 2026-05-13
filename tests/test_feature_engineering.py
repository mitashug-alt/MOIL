import pandas as pd

from institutional_flow.config import tracker_config
from institutional_flow.features import combine_features, deal_features, shareholding_features, mf_features, disclosures_features


def sample_deals():
    return pd.DataFrame(
        {
            "date": ["2026-05-12", "2026-05-12"],
            "symbol": ["MOIL", "MOIL"],
            "security_name": ["MOIL LTD", "MOIL LTD"],
            "client_name": ["Fund A", "Fund B"],
            "buy_sell": ["BUY", "SELL"],
            "quantity": [1_100_000, 200_000],
            "price": [320.5, 321.0],
            "remarks": ["", ""],
            "deal_type": ["bulk", "bulk"],
            "source": ["NSE", "NSE"],
        }
    )


def sample_shp():
    return pd.DataFrame(
        {
            "quarter_end": ["2026-03-31", "2026-03-31"],
            "category": ["Promoter", "Public"],
            "sub_category": ["Promoter", "Mutual Funds"],
            "holder_name": ["Government", "MF total"],
            "shares": [131_624_088, 4_960_456],
            "percentage": [64.68, 2.44],
        }
    )


def sample_mf():
    return pd.DataFrame(
        {
            "month": ["2026-04-30", "2026-04-30"],
            "amc": ["AMC1", "AMC2"],
            "scheme_name": ["Fund1", "Fund2"],
            "shares": [200_000, 100_000],
            "market_value": [0, 0],
            "percentage_of_scheme": [0.1, 0.05],
            "source": ["manual", "manual"],
        }
    )


def sample_disclosures():
    return pd.DataFrame(
        {
            "event_date": ["2026-05-10"],
            "disclosure_type": ["SAST"],
            "entity_name": ["Investor X"],
            "symbol": ["MOIL"],
            "acquisition_or_disposal": ["BUY"],
            "shares": [5_000_000],
            "percentage": [2.46],
            "value_inr": [100_000_000],
            "trigger_type": ["SAST_2_PERCENT_CHANGE"],
            "source_url": [""],
            "notes": [""],
        }
    )


def test_deal_features_bulk_threshold():
    cfg = tracker_config()
    feats = deal_features(sample_deals(), cfg)
    assert bool(feats.loc[0, "moil_bulk_threshold_hit"]) is True
    assert feats.loc[0, "bulk_net_qty"] == 900_000


def test_shareholding_features_named_holder():
    cfg = tracker_config()
    feats = shareholding_features(sample_shp(), cfg)
    assert feats.loc[0, "named_1pct_holder_count"] == 1


def test_mf_features():
    cfg = tracker_config()
    feats = mf_features(sample_mf(), cfg)
    assert feats.loc[0, "mf_net_shares"] == 300_000
    assert feats.loc[0, "top_mf_holder_name"] == "Fund1"


def test_disclosure_features():
    cfg = tracker_config()
    feats = disclosures_features(sample_disclosures(), cfg)
    assert bool(feats.loc[0, "sast_event_flag"]) is True
    assert feats.loc[0, "value_disclosed_in_events"] == 100_000_000


def test_combine_features():
    cfg = tracker_config()
    combined = combine_features(sample_deals(), sample_shp(), sample_mf(), sample_disclosures(), cfg)
    assert "bulk_net_qty" in combined.columns
    assert "mf_net_shares" in combined.columns
    assert combined["symbol"].iloc[0] == "MOIL"

import pandas as pd

from institutional_flow.nse_client import NSEClient
from institutional_flow.config import tracker_config


def test_nse_standardize_columns():
    cfg = tracker_config({"enable_network": False})
    client = NSEClient.create(cfg)
    raw = pd.DataFrame(
        {
            "Date": ["2026-05-12"],
            "Symbol": ["MOIL"],
            "Security Name": ["MOIL LTD"],
            "Client Name": ["Big Fund"],
            "Buy/Sell": ["BUY"],
            "Quantity Traded": [1_200_000],
            "Trade Price / Wght. Avg. Price": [320.5],
            "Remarks": ["NSE"],
        }
    )
    std = client._standardize_deals(raw, "bulk_deals")
    assert set(["date", "symbol", "security_name", "client_name", "buy_sell", "quantity", "price", "remarks", "deal_type", "source"]).issubset(std.columns)
    assert std.loc[0, "quantity"] == 1_200_000
    assert std.loc[0, "price"] == 320.5
    assert std.loc[0, "deal_type"] == "bulk_deals"


def test_filter_symbol_matches_security_name():
    cfg = tracker_config({"enable_network": False})
    client = NSEClient.create(cfg)
    df = pd.DataFrame(
        {
            "date": ["2026-05-12"],
            "symbol": [""],
            "security_name": ["MOIL LTD"],
        }
    )
    filtered = client.filter_symbol(df, "MOIL")
    assert len(filtered) == 1

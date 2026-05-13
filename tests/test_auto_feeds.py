import io
import time

import pandas as pd

import auto_feeds as af


def _sample_csv(text: str) -> str:
    return text.strip() + "\n"


def test_nse_deals_cache_dedup_and_validation(monkeypatch):
    # clear caches
    af._NSE_ARCHIVE_CACHE.clear()
    af._NSE_SYMBOL_CACHE.clear()

    bulk_csv = _sample_csv(
        """
Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks
2025-01-15,MOIL,MOIL LTD,Alpha,BUY,1000,250,bulk1
2025-01-15,MOIL,MOIL LTD,Alpha,BUY,1000,250,bulk1
2025-01-10,XYZ,XYZ LTD,Beta,SELL,0,10,bad_qty
"""
    )
    block_csv = _sample_csv(
        """
Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks
2025-01-15,MOIL,MOIL LTD,Gamma,SELL,500,240,block1
"""
    )

    calls = {"bulk": 0, "block": 0}

    def fake_get(url: str, timeout: int = 20, retries: int = 3, backoff: float = 1.5):
        if "bulk" in url:
            calls["bulk"] += 1
            return True, bulk_csv, "OK bulk"
        if "block" in url:
            calls["block"] += 1
            return True, block_csv, "OK block"
        return False, "", "bad url"

    monkeypatch.setattr(af, "_safe_request_get", fake_get)

    deals, diag = af.fetch_nse_deals_auto("MOIL")
    # Dedup should remove duplicate bulk row; bad_qty row dropped; total 2 rows remain
    assert len(deals) == 2
    assert any("latest date" in d for d in diag)
    assert calls["bulk"] == 1 and calls["block"] == 1

    # Second call should hit symbol-level cache and not re-fetch
    deals2, diag2 = af.fetch_nse_deals_auto("MOIL")
    assert len(deals2) == 2
    assert calls["bulk"] == 1 and calls["block"] == 1  # no additional fetches


def test_nse_deals_security_name_fallback(monkeypatch):
    af._NSE_ARCHIVE_CACHE.clear()
    af._NSE_SYMBOL_CACHE.clear()

    bulk_csv = _sample_csv(
        """
Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks
2025-01-15,MOILL,MOIL LTD,Alpha,BUY,1000,250,bulk1
"""
    )

    def fake_get(url: str, timeout: int = 20, retries: int = 3, backoff: float = 1.5):
        return True, bulk_csv, "OK bulk"

    monkeypatch.setattr(af, "_safe_request_get", fake_get)

    deals, diag = af.fetch_nse_deals_auto("MOIL")
    # Symbol doesn't match, but security_name contains MOIL -> return rows from archives
    assert len(deals) >= 1
    assert deals.iloc[0]["security_name"].upper().startswith("MOIL")

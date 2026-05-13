import os
from pathlib import Path

import pandas as pd
import pytest

import macro_scraper.macro_scraper as ms


def test_load_config_expands_env(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    monkeypatch.setenv("FOO_KEY", "bar-value")
    cfg_path.write_text("indicators: []\nfoo: ${FOO_KEY}\n", encoding="utf-8")
    cfg = ms.load_config(cfg_path)
    assert cfg["foo"] == "bar-value"


def test_csv_aggregate_latest(tmp_path):
    csv_path = tmp_path / "steel.csv"
    csv_path.write_text(
        """CAL_YEAR,MONTH_NO,MONTH,DATA_TYPE,QTY
2024,12,DEC,CRUDE_STEEL_PROD,100
2025,1,JAN,CRUDE_STEEL_PROD,200
2025,1,JAN,CRUDE_STEEL_PROD,300
""",
        encoding="utf-8",
    )
    spec = {
        "source_url": str(csv_path),
        "method": "csv",
        "value_field": "QTY",
        "date_field": {"year": "CAL_YEAR", "month_number": "MONTH_NO"},
        "row_filter": "DATA_TYPE == 'CRUDE_STEEL_PROD'",
        "row_selector": "last_after_sort(CAL_YEAR, MONTH_NO)",
        "sort_by": ["CAL_YEAR", "MONTH_NO"],
        "aggregate_latest_by": ["CAL_YEAR", "MONTH_NO"],
    }
    # session is unused in fetch_csv
    value, date, notes = ms.fetch_csv(None, spec)
    assert value == 500  # sum of latest period (2025-01)
    assert date == "2025-01-01"
    assert notes == ""


def test_run_uses_fallback_when_primary_fails(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    cfg_path.write_text(
        """
rate_limit_sleep_seconds: 0
indicators:
  - name: "China Steel Exports"
    geography: "China"
    unit: "tonnes"
    source_name: "GACC"
    source_url: "file://./does_not_exist.html"
    method: "html"
    css_selector: "body"
""",
        encoding="utf-8",
    )

    fallback_row = {
        "indicator": "China Steel Exports",
        "geography": "China",
        "date": "2026-04-01",
        "value": 12345,
        "unit": "tonnes",
        "source": "fallback_test",
    }

    def fake_fallback():
        return {"China Steel Exports": fallback_row}, ["fallback used"]

    monkeypatch.setattr(ms, "load_fallback_rows", fake_fallback)

    ms.run(cfg_path, out_dir, ["csv", "json"])

    df = pd.read_csv(out_dir / "macro_data.csv")
    row = df.iloc[0]
    assert row["indicator"] == "China Steel Exports"
    assert row["status"] == "fallback"
    assert row["value"] == 12345
    assert row["date"] == "2026-04-01"
    assert row["notes"] == "filled from auto_feeds fallback"


def test_html_regex_parse(tmp_path):
    html_path = tmp_path / "page.html"
    html_path.write_text(
        "<html><body>Steel products 10,000 Tons 123.45 more text April 2026</body></html>",
        encoding="utf-8",
    )
    spec = {
        "source_url": f"file://{html_path}",
        "method": "html",
        "css_selector": "body",
        "value_regex": r"Steel products 10,000 Tons ([0-9,.]+)",
        "date_regex": r"(April 2026)",
        "value_group": 1,
        "value_multiplier": 10000,
    }
    session = ms.make_session()
    value, date, notes = ms.fetch_html(session, spec)
    assert value == 123.45 * 10000
    assert date.startswith("2026-04")  # day component may be inferred as mid-month
    assert notes == ""


def test_api_json_path(monkeypatch):
    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": {"val": 42, "dt": "2025-02-01"}}

    class FakeSession:
        request_timeout = 5

        def request(self, method, url, json=None, params=None, headers=None, timeout=None):
            return FakeResp()

    spec = {
        "source_url": "https://api.example.com/x",
        "http_method": "GET",
        "json_path": ["data"],
        "value_field": "val",
        "date_field": "dt",
    }
    value, date, notes = ms.fetch_api(FakeSession(), spec)
    assert value == 42
    assert date == "2025-02-01"
    assert notes == ""


def test_enforce_output_schema_missing_columns_raises():
    df = pd.DataFrame({"indicator": ["x"], "status": ["ok"], "value": [1], "date": ["2025-01-01"]})
    with pytest.raises(ValueError):
        ms.enforce_output_schema(df)


def test_plausibility_marks_partial():
    row = {
        "name": "Silico-Manganese 65/17 Price (Delivered Europe)",
        "geography": "Europe",
        "unit": "$US/MT",
        "source_name": "test",
        "source_url": "file://x",
        "method": "api",
    }

    class FakeSession:
        request_timeout = 1

        def request(self, method, url, json=None, params=None, headers=None, timeout=None):
            class R:
                status_code = 200

                def json(self):
                    return {"val": 999999, "dt": "2025-01-01"}

            return R()

    spec = {**row, "value_field": "val", "date_field": "dt"}
    out = ms.fetch_indicator(FakeSession(), spec)
    assert out["status"] == "partial"
    assert "plausibility" in out["notes"]

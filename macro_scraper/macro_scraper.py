#!/usr/bin/env python
"""
Config-driven macro scraper.

CLI:
  python macro_scraper.py --config config.yaml --out ./out --format csv,json

Output schema (CSV & JSON):
indicator, geography, date, value, unit, source_name, source_url, retrieved_at_utc, status, notes
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from dateutil import parser as dateparser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # Only used if configured for HTML scraping


LOG = logging.getLogger("macro_scraper")

PLAUSIBILITY_RANGES = {
    "Silico-Manganese 65/17 Price (Delivered Europe)": (0, 5000),  # $/MT
    "India Crude Steel Production (Total)": (0, 50000),  # thousand tonnes
    "China Steel Exports (Current Month)": (0, 1e8),  # tonnes
    "China Power Generation (Above-scale Industrial)": (0, 1e6),  # 亿千瓦时
}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    expanded = os.path.expandvars(text)
    return yaml.safe_load(expanded)


def resolve_local_path(url: str) -> Path | None:
    if url.startswith("file://"):
        return Path(url[7:])
    p = Path(url)
    if p.exists():
        return p
    return None


def load_fallback_rows() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Load fallback macro rows via auto_feeds.fetch_source_aware_macro_feed.

    Returns a mapping of indicator -> row dict and diagnostics list.
    """
    try:
        from auto_feeds import fetch_source_aware_macro_feed  # type: ignore

        df, diagnostics = fetch_source_aware_macro_feed()
        mapping: Dict[str, Dict[str, Any]] = {}
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                ind = str(r.get("indicator", "")).strip()
                if ind:
                    mapping[ind] = r.to_dict()
        return mapping, diagnostics
    except Exception as exc:  # pragma: no cover
        return {}, [f"fallback pipeline failed: {exc}"]


def enforce_output_schema(df: pd.DataFrame) -> None:
    required_cols = [
        "indicator",
        "geography",
        "date",
        "value",
        "unit",
        "source_name",
        "source_url",
        "retrieved_at_utc",
        "status",
        "notes",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Output missing required columns: {missing}")

    # For rows that are ok/fallback/partial ensure value/date present and value numeric
    must_have = df[df["status"].isin(["ok", "fallback", "partial"])]
    if not must_have.empty:
        if must_have["date"].isna().any() or (must_have["date"].astype(str).str.len() == 0).any():
            raise ValueError("Output validation failed: date missing for ok/fallback/partial rows")
        val_num = pd.to_numeric(must_have["value"], errors="coerce")
        if val_num.isna().any():
            raise ValueError("Output validation failed: non-numeric value in ok/fallback/partial rows")


def now_utc_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%dT%H:%M:%S IST")


def make_session(timeout: int = 20, retries: int = 3, backoff: float = 1.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def parse_numeric(val: Any) -> Tuple[float | None, str]:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None, "value is NaN"
        s = str(val).strip().replace(",", "")
        if s == "":
            return None, "value is empty"
        num = float(s)
        if pd.isna(num):
            return None, "value is NaN"
        return num, ""
    except Exception as exc:  # pragma: no cover
        return None, f"non-numeric value: {exc}"


def parse_date(val: Any) -> Tuple[str | None, str]:
    try:
        dt = dateparser.parse(str(val)).date()
        return dt.isoformat(), ""
    except Exception as exc:  # pragma: no cover
        return None, f"unparsable date: {exc}"


def parse_date_from_row(row: Any, date_field_spec: Any) -> Tuple[str | None, str]:
    if isinstance(date_field_spec, dict):
        try:
            def to_int(val: Any) -> int:
                return int(float(val))

            year_raw = row.get(date_field_spec.get("year"))
            month_raw = row.get(date_field_spec.get("month_number", date_field_spec.get("month", 1)))
            day_key = date_field_spec.get("day")
            day_raw = row.get(day_key) if day_key else 1
            if pd.isna(year_raw) or pd.isna(month_raw) or pd.isna(day_raw):
                raise ValueError("missing date parts")
            year = to_int(year_raw)
            month = to_int(month_raw)
            day = to_int(day_raw)
            dt = datetime(year=year, month=month, day=day).date()
            return dt.isoformat(), ""
        except Exception as exc:
            return None, f"date parts missing: {exc}"
    return parse_date(row.get(date_field_spec))


def fetch_api(session: requests.Session, spec: Dict[str, Any]) -> Tuple[float | None, str | None, str]:
    url = spec["source_url"]
    method = spec.get("http_method", "GET").upper()
    payload = spec.get("payload")
    headers = spec.get("headers", {})
    params = spec.get("params", {})
    resp = session.request(method, url, json=payload, params=params, headers=headers, timeout=session.request_timeout)
    if resp.status_code >= 400:
        return None, None, f"HTTP {resp.status_code}"
    data = resp.json()
    json_path = spec.get("json_path", [])
    try:
        for key in json_path:
            data = data[key]
    except Exception as exc:
        return None, None, f"json_path failed: {exc}"
    value_raw = data.get(spec.get("value_field")) if isinstance(data, dict) else data
    date_raw = data.get(spec.get("date_field")) if isinstance(data, dict) else None
    value, verr = parse_numeric(value_raw)
    date, derr = parse_date(date_raw) if date_raw is not None else (None, "missing date")
    notes = "; ".join([x for x in [verr, derr] if x])
    return value, date, notes or ""


def fetch_csv(session: requests.Session, spec: Dict[str, Any]) -> Tuple[float | None, str | None, str]:
    url = spec["source_url"]
    local_path = resolve_local_path(url)
    try:
        if local_path:
            df = pd.read_csv(local_path)
        else:
            df = pd.read_csv(url)
    except Exception as exc:
        return None, None, f"csv load failed: {exc}"
    df = df.dropna(how="all")
    filter_expr = spec.get("row_filter")
    if filter_expr:
        try:
            df = df.query(filter_expr)
        except Exception as exc:
            return None, None, f"row_filter failed: {exc}"
    if df.empty:
        return None, None, "no rows after filter"

    sort_fields: List[str] = []
    row_selector = spec.get("row_selector")
    if isinstance(row_selector, str) and row_selector.startswith("last_after_sort("):
        inside = row_selector[len("last_after_sort(") : -1]
        sort_fields = [x.strip() for x in inside.split(",") if x.strip()]
    if not sort_fields:
        sort_fields = spec.get("sort_by", [])
    if sort_fields:
        try:
            df = df.sort_values(by=sort_fields)
        except Exception as exc:
            return None, None, f"sort failed: {exc}"

    value_multiplier = float(spec.get("value_multiplier", 1.0))

    if spec.get("aggregate_latest_by"):
        latest_fields = spec["aggregate_latest_by"]
        try:
            # drop rows missing date parts if date_field is dict
            date_field_spec = spec.get("date_field")
            df_use = df
            if isinstance(date_field_spec, dict):
                drop_fields = [f for f in [date_field_spec.get("year"), date_field_spec.get("month_number"), date_field_spec.get("month"), date_field_spec.get("day")] if f]
                if drop_fields:
                    df_use = df.dropna(subset=drop_fields)
                    if df_use.empty:
                        return None, None, "no rows after dropping missing date parts"
            last_row = df_use.iloc[-1] if not sort_fields else df_use.sort_values(by=sort_fields).iloc[-1]
            mask = pd.Series(True, index=df_use.index)
            for field in latest_fields:
                mask &= df_use[field] == last_row[field]
            subset = df_use[mask]
            raw_value = subset[spec.get("value_field")].sum()
            date, derr = parse_date_from_row(last_row, spec.get("date_field"))
            value, verr = parse_numeric(raw_value)
        except Exception as exc:
            return None, None, f"aggregate_latest_by failed: {exc}"
        if value is not None:
            value *= value_multiplier
        notes = "; ".join([x for x in [verr, derr] if x])
        return value, date, notes or ""

    try:
        if row_selector == "first":
            row = df.iloc[0]
        elif row_selector == "last" or row_selector == "last_after_sort":
            row = df.iloc[-1]
        elif isinstance(row_selector, int):
            row = df.iloc[row_selector]
        else:
            row = df.iloc[-1]
    except Exception as exc:
        return None, None, f"row selection failed: {exc}"

    value, verr = parse_numeric(row.get(spec.get("value_field")))
    if value is not None:
        value *= value_multiplier
    date, derr = parse_date_from_row(row, spec.get("date_field"))
    notes = "; ".join([x for x in [verr, derr] if x])
    return value, date, notes or ""


def fetch_html(session: requests.Session, spec: Dict[str, Any]) -> Tuple[float | None, str | None, str]:
    if BeautifulSoup is None:
        return None, None, "bs4 not available"
    url = spec["source_url"]
    local_path = resolve_local_path(url)
    try:
        if local_path:
            html_text = local_path.read_text(encoding="utf-8", errors="ignore")
        else:
            resp = session.get(url, timeout=session.request_timeout, headers=spec.get("headers", {}))
            if resp.status_code >= 400:
                return None, None, f"HTTP {resp.status_code}"
            html_text = resp.text
        soup = BeautifulSoup(html_text, "html.parser")
        selector = spec.get("css_selector")
        node = soup.select_one(selector) if selector else None
        if not node:
            return None, None, "css_selector not found"
        text = node.get_text(" ", strip=True)
    except Exception as exc:
        return None, None, f"html parse failed: {exc}"
    value = None
    date = None
    notes: List[str] = []
    flags = re.IGNORECASE if spec.get("regex_ignore_case", True) else 0

    if spec.get("row_regex"):
        try:
            matches = re.findall(spec["row_regex"], text, flags)
            if not matches:
                notes.append("row_regex no match")
            else:
                match_index = int(spec.get("row_match_index", 0))
                m = matches[match_index]
                if not isinstance(m, tuple):
                    m = (m,)
                if spec.get("row_value_group"):
                    raw_val = m[int(spec["row_value_group"]) - 1]
                    value, verr = parse_numeric(raw_val)
                    if verr:
                        notes.append(verr)
                if spec.get("row_date_group"):
                    raw_date = m[int(spec["row_date_group"]) - 1]
                    date, derr = parse_date(raw_date)
                    if derr:
                        notes.append(derr)
        except Exception as exc:
            return None, None, f"row_regex failed: {exc}"

    if value is None and spec.get("value_regex"):
        try:
            matches = re.findall(spec["value_regex"], text, flags)
            if not matches:
                notes.append("value_regex no match")
            else:
                match_index = int(spec.get("value_match_index", 0))
                m = matches[match_index]
                if isinstance(m, tuple):
                    group_index = int(spec.get("value_group", 1)) - 1
                    raw_val = m[group_index]
                else:
                    raw_val = m
                value, verr = parse_numeric(raw_val)
                if verr:
                    notes.append(verr)
        except Exception as exc:
            return None, None, f"value_regex failed: {exc}"

    if date is None and spec.get("date_regex"):
        try:
            matches = re.findall(spec["date_regex"], text, flags)
            if not matches:
                notes.append("date_regex no match")
            else:
                match_index = int(spec.get("date_match_index", 0))
                m = matches[match_index]
                if isinstance(m, tuple):
                    group_index = int(spec.get("date_group", 1)) - 1
                    raw_date = m[group_index]
                else:
                    raw_date = m
                date, derr = parse_date(raw_date)
                if derr:
                    notes.append(derr)
        except Exception as exc:
            return None, None, f"date_regex failed: {exc}"

    if value is None:
        value, verr = parse_numeric(text)
        if verr:
            notes.append(verr)

    if date is None:
        date, derr = parse_date(spec.get("assumed_date", ""))
        if derr:
            notes.append(derr)

    value_multiplier = float(spec.get("value_multiplier", 1.0))
    if value is not None:
        value *= value_multiplier

    notes_str = "; ".join([n for n in notes if n])
    return value, date, notes_str


def fetch_indicator(session: requests.Session, ind: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "indicator": ind.get("name", ""),
        "geography": ind.get("geography", ""),
        "unit": ind.get("unit", ""),
        "source_name": ind.get("source_name", ""),
        "source_url": ind.get("source_url", ""),
        "retrieved_at_utc": now_utc_iso(),
    }
    method = ind.get("method", "api")
    value = None
    date = None
    notes = ""
    status = "failed"
    try:
        if method == "api":
            value, date, notes = fetch_api(session, ind)
        elif method == "csv":
            value, date, notes = fetch_csv(session, ind)
        elif method == "html":
            value, date, notes = fetch_html(session, ind)
        elif method == "none":
            notes = "method none; fallback"
        else:
            notes = f"unsupported method {method}"
        if value is not None and date is not None and not notes:
            status = "ok"
        elif value is not None and date is not None:
            status = "partial"

        # Plausibility check
        pr = PLAUSIBILITY_RANGES.get(ind.get("name", ""))
        if value is not None and pr is not None:
            lo, hi = pr
            if value < lo or value > hi:
                status = "partial" if status != "failed" else status
                note_val = f"plausibility: {value} outside [{lo}, {hi}]"
                notes = note_val if not notes else f"{notes}; {note_val}"
    except Exception as exc:  # pragma: no cover
        notes = f"exception: {exc}"
    return {
        **base,
        "date": date,
        "value": value,
        "status": status,
        "notes": notes,
    }


def run(config_path: Path, out_dir: Path, out_formats: List[str]) -> None:
    cfg = load_config(config_path)
    indicators = cfg.get("indicators", [])
    session = make_session(timeout=20, retries=3, backoff=1.5)
    fallback_map, fallback_diag = load_fallback_rows()
    rows: List[Dict[str, Any]] = []
    for ind in indicators:
        time.sleep(cfg.get("rate_limit_sleep_seconds", 1.0))
        row = fetch_indicator(session, ind)
        # Fallback: if failed, attempt to fill from auto_feeds mapping
        if row["status"] == "failed":
            fb = fallback_map.get(row["indicator"].strip()) if row.get("indicator") else None
            if fb:
                row = {
                    "indicator": row.get("indicator", fb.get("indicator", "")),
                    "geography": row.get("geography", fb.get("geography", "")),
                    "date": fb.get("date", None),
                    "value": fb.get("value", None),
                    "unit": fb.get("unit", ""),
                    "source_name": fb.get("source", "fallback"),
                    "source_url": ind.get("source_url", ""),
                    "retrieved_at_utc": now_utc_iso(),
                    "status": "fallback",
                    "notes": "filled from auto_feeds fallback",
                }
        LOG.info("indicator=%s status=%s notes=%s", row["indicator"], row["status"], row["notes"])
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "indicator",
        "geography",
        "date",
        "value",
        "unit",
        "source_name",
        "source_url",
        "retrieved_at_utc",
        "status",
        "notes",
    ])
    enforce_output_schema(df)

    if "csv" in out_formats:
        df.to_csv(out_dir / "macro_data.csv", index=False)
    if "json" in out_formats:
        (out_dir / "macro_data.json").write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro indicator scraper (config-driven)")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--format", type=str, default="csv,json", help="Comma-separated: csv,json")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    formats = [x.strip() for x in args.format.split(",") if x.strip() in {"csv", "json"}]
    if not formats:
        LOG.error("No valid output formats specified; use csv,json")
        sys.exit(1)

    run(args.config, args.out, formats)


if __name__ == "__main__":
    main()

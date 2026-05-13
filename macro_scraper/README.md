# Macro Scraper

Config‑driven scraper for four macro indicators. No hardcoded sources: all endpoints/selectors live in `config.yaml`. Safe defaults: retries, timeouts, basic validation, no paywalls.

## Requirements
- Python 3.13.13
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```

## Configure
Edit `config.yaml` with real, non-paywalled sources. Supported knobs:
- `method`: api | csv | html
- For `api`: `source_url`, `http_method`, `params`, `headers`, `json_path`, `value_field`, `date_field`
- For `csv`: `source_url`, `value_field`, `date_field` (string or `{year, month_number, day}`), optional `row_filter` (pandas query), `sort_by`, `row_selector` (`first`, `last`, or `last_after_sort(col1,col2)`), `aggregate_latest_by` (fields to group the latest period and sum), `value_multiplier`
- For `html`: `source_url`, `css_selector` (use `body` + regex if classes are unstable), optional regex keys: `row_regex` with `row_match_index`, `row_value_group`, `row_date_group`; or `value_regex`/`date_regex` with `value_group`/`date_group` and match indexes; `assumed_date` fallback; `value_multiplier`

Offline/local cache: you can point `source_url` to local files (absolute paths or `file://...`). A helper in code resolves local paths so you can download HTML/CSV once into `macro_scraper/local_cache/` and run offline.

Current example sources wired in:
- JPC crude steel via data.gov.in (requires `DATA_GOV_IN_KEY` env)
- ScrapMonster Silico-Manganese 65/17 (historical table regex)
- China Steel Exports (GACC HTML placeholder; will fall back if local file missing)
- China Power Generation (NBS HTML placeholder; will fall back if local file missing)
If a scrape fails, the scraper now attempts to fill the row from `auto_feeds.fetch_source_aware_macro_feed` and marks `status="fallback"`.

## Run
```bash
python macro_scraper.py --config config.yaml --out ./out --format csv,json
```

## Test
```bash
pip install pytest
pytest tests
```

Outputs:
- `out/macro_data.csv`
- `out/macro_data.json`

Schema:
```
indicator, geography, date, value, unit, source_name, source_url, retrieved_at_utc, status, notes
```

Status rules:
- `ok`: value and date parsed, no validation notes
- `partial`: value/date parsed but with notes
- `failed`: nothing parsed; see `notes`

## Notes
- Retries: 3 with exponential backoff; timeout: 20s.
- Basic numeric/date validation; if parsing fails, status is `failed` with reason.
- HTML parsing uses BeautifulSoup only when configured.
- Respect robots.txt / paywalls; replace example URLs with official/public endpoints.

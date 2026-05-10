# MOIL Macro Radar - Groq Regime Intelligence Upgrade

Streamlit dashboard for MOIL Ltd and the Indian manganese/steel cycle.

## New in this patch

- Groq/Llama 3.3 remains the primary AI scorer.
- Optional automatic macro feed pull using `SERPER_API_KEY` before Groq scoring.
- Groq no longer depends only on stale manual rows when a search API is configured.
- NSE institutional sync now tries the official NSE archive CSV first and distinguishes:
  - endpoint blocked/unavailable
  - feed loaded but no MOIL bulk/block rows today
- CSV upload fallback for NSE bulk/block files when Codespaces blocks exchange endpoints.
- Telegram alerts remain enabled.

## Important architecture note

Groq/Llama does not browse the web by itself in this app. The dashboard must first fetch data from APIs/web feeds/search snippets, then Groq scores the supplied facts. Automatic physical macro scoring therefore requires either:

1. a paid/authorized commodity feed such as SteelMint/BigMint/Reuters, or
2. a web-search API such as Serper, or
3. NSE/public archive CSVs and uploaded files.

## Secrets

Create `.streamlit/secrets.toml` locally or use GitHub Codespaces secrets.

```toml
GROQ_API_KEY = ""
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_KEY_NAME = "MOIL Macro Radar Groq Key"
GROQ_PROJECT = "MOIL Macro Radar"

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# Optional, but required for automatic macro web pull
SERPER_API_KEY = ""
```

Never commit `.streamlit/secrets.toml`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

## Recommended workflow

1. Open Market radar and Data quality.
2. Go to Groq AI desk.
3. If `SERPER_API_KEY` is configured, click **Pull auto macro feed**.
4. Click **Run Groq macro scoring**.
5. Click **Apply Groq scores to regime**.
6. Go to Institutional sync and click **NSE Sync: Bulk/Block Deals**.
7. If NSE is blocked, upload NSE CSV through the fallback uploader.
8. Generate Groq commentary and send Telegram alert.


## Source-aware automatic macro feeds

Groq/Llama scores facts supplied by the app; it does not independently browse paid commodity sources. The app therefore uses a feed ladder:

1. Licensed/vendor endpoints configured in Streamlit secrets
2. SERPER_API_KEY source-targeted web-search fallback
3. Manual macro table fallback

Preferred source map:

| Indicator | Primary source | Secondary source | Secret endpoint |
|---|---|---|---|
| Silico-Manganese Prices | Argus Media | FerroAlloyNet / BigMint manual | ARGUS_SIMN_FEED_URL |
| India Crude Steel Production | Ministry of Steel / JPC | World Steel Association | JPC_STEEL_FEED_URL |
| China Steel Exports | GACC | Mysteel / Argus | GACC_STEEL_EXPORTS_FEED_URL |
| China Power / Industrial Stress | NBS China | Bloomberg / S&P Global Commodity Insights | CHINA_POWER_STRESS_FEED_URL |

Add optional secrets:

```toml
SERPER_API_KEY = ""
ARGUS_API_KEY = ""
ARGUS_SIMN_FEED_URL = ""
JPC_STEEL_FEED_URL = ""
GACC_STEEL_EXPORTS_FEED_URL = ""
CHINA_POWER_STRESS_FEED_URL = ""
```

These endpoint URLs can be vendor APIs, internal CSV exports, or lightweight proxy services. The app never displays keys.

## Backend scraper + smart money upgrade

This repository can run low-frequency scheduled jobs that write cached JSON for the dashboard:

```bash
python -m scrapers.run_daily_scrape
python institutional_tracker.py
```

Cached outputs:

```text
data/cache/latest_macro_data.json
data/cache/institutional_activity_latest.json
```

The Streamlit app reads cached macro rows through `backend_cache_loader.py` and `auto_feeds.py`, so Groq scores supplied facts instead of relying only on stale manual inputs.

GitHub Actions workflows:

```text
.github/workflows/daily_macro_scrape.yml
.github/workflows/daily_institutional_tracker.yml
.github/workflows/manual_scraper_test.yml
```

Use official downloads and permitted public pages first. Do not scrape on every dashboard page load.

## Data Quality v2 Upgrade

This build adds a source-governed evidence pipeline. Scheduled scrapers and verified uploads are converted into canonical evidence rows with a 0-100 data confidence score. Groq/Llama 3.3 now returns both a raw `signal_score` and a confidence-adjusted `effective_score`; only the effective score moves the regime model.

Confidence gates:

```text
>=80     full score impact
60-79    70% score impact
40-59    40% score impact
<40      commentary-only / no regime impact
```

New files:

```text
data_layer/source_registry.yaml
data_layer/confidence_engine.py
data_layer/evidence_store.py
data_layer/validators.py
schemas/evidence_row.schema.json
DATA_QUALITY_V2.md
```

New app surfaces:

```text
Groq AI desk -> Source readiness v2
Manual macro tracker -> Verified evidence CSV/JSON upload
Data quality -> Data Quality Command Center + Evidence Viewer
```

## Institutional Quality v2

The smart-money tracker now converts NSE/Trendlyne/Screener/MOIL shareholding outputs into canonical institutional evidence rows. Each row receives a confidence score and a confidence-adjusted smart-money impact. Low-confidence or low-materiality rows are used for commentary only and do not move the regime score.

See `INSTITUTIONAL_QUALITY_V2.md` for details.

# MOIL Macro Radar Backend Scraper + Smart Money Patch

This patch upgrades the current Groq-powered MOIL Macro Radar into a cached backend intelligence system.

## What it adds

- Scheduled MOIL PDF circular scraper using `requests` + `pdfplumber`
- Generic public HTML table scraper using `BeautifulSoup` + `pandas.read_html`
- Smart Money Tracker for NSE archive CSV, Trendlyne public rows, Screener/MOIL shareholding pages
- Cached JSON outputs under `data/cache/`
- JSONL history under `data/history/`
- GitHub Actions daily jobs
- Source confidence and diagnostics
- Manual upload fallback stays in the app

## Data architecture

```text
Official download / permitted public page
        -> scheduled scraper
        -> cached JSON
        -> Groq scoring
        -> regime score
        -> Telegram alert
```

Do not scrape during every Streamlit page load.

## Key files

```text
scrapers/moil_pdf_scraper.py
scrapers/html_table_scraper.py
scrapers/run_daily_scrape.py
institutional_tracker.py
backend_cache_loader.py
auto_feeds.py
.github/workflows/daily_macro_scrape.yml
.github/workflows/daily_institutional_tracker.yml
schemas/
```

## Local commands

```bash
pip install -r requirements.txt
python -m scrapers.run_daily_scrape
python institutional_tracker.py
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

## Actions manual run

Go to GitHub -> Actions -> Daily MOIL Macro Scrape -> Run workflow.
Then run Daily MOIL Smart Money Tracker.

## Secrets

Required for AI/Telegram:

```text
GROQ_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Optional:

```text
MOIL_OUTSTANDING_SHARES
SERPER_API_KEY
ARGUS_SIMN_FEED_URL
JPC_STEEL_FEED_URL
GACC_STEEL_EXPORTS_FEED_URL
CHINA_POWER_STRESS_FEED_URL
```

## Compliance guardrail

The scrapers do not bypass login walls, CAPTCHA, fingerprinting, paywalls, password-protected PDFs, or bot-detection systems. If a source blocks cloud access, the system records diagnostics and uses cached/manual fallback.

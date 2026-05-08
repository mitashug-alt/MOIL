# MOIL Macro Radar

Institutional-grade commodity intelligence dashboard focused on MOIL, Indian steel-cycle proxies, manganese-linked demand signals, China supply pressure, energy stress and broader commodity risk appetite.

## What this version includes

- Streamlit dashboard with interactive Plotly charts
- Live yfinance market proxies for MOIL, NIFTY Metal, steel peers, Brent, USDINR and Baltic Dry proxy
- Robust ticker map that can be edited in the sidebar
- Price normalization and MOIL candlestick view
- Correlation engine with heatmap, MOIL correlation ranking and rolling correlation chart
- Composite macro regime scoring with transparent component-level audit trail
- Manual physical macro tracker for silico-manganese prices, India crude steel production, China steel exports and China power/industrial stress
- Return anomaly monitor using rolling z-scores
- Telegram alert preview and push button
- Rule-based macro commentary by default, with optional Google Gemini commentary when credentials are supplied
- Downloadable price panel, manual macro CSV and regime scorecard
- Feed placeholder structure for Reuters / SteelMint / internal data connectors

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Credentials

Copy the example secrets file and fill values locally:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Supported optional secrets:

```toml
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
API_KEY = ""
GOOGLE_API_KEY = ""
```

Never commit real credentials.

## Manual physical macro workflow

The dashboard loads `data/manual_macro_template.csv` by default. Update this file weekly/monthly for indicators that do not have reliable free live feeds:

- Silico-manganese prices
- India crude steel production
- China steel exports
- China power / industrial stress

Each row has a `score` from `-2` to `+2`:

- `+2`: strongly supportive for MOIL-linked manganese/steel cycle
- `+1`: mildly supportive
- `0`: neutral or unknown
- `-1`: mildly adverse
- `-2`: strongly adverse

You can also upload or edit this CSV directly inside the Streamlit app and download the updated file.

## Regime score interpretation

The score converts component points into a 0-100 regime scale:

- 70-100: Bullish industrial expansion
- 55-70: Constructive / accumulating
- 45-55: Neutral / watch
- 30-45: Defensive industrial regime
- 0-30: Stress / de-rating risk

The model is intentionally transparent. Treat it as a desk radar, not as an automatic trading recommendation.

## Telegram alerts

Use the dashboard button under **Alerts & commentary**, or run a test from the command line after setting environment variables:

```bash
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHAT_ID="your-chat-id"
python telegram_alert.py
```

## Future extensions

Suggested next modules:

- Reuters China industrial stress connector
- SteelMint silico-manganese feed connector
- India steel production feed
- China steel export monitor
- More advanced anomaly detection and lead-lag correlation engine
- Macro scoring model with persisted history and alert thresholds
- Scheduled jobs that write normalized data into `data/`

## Notes

Yahoo Finance coverage can change and some proxies may be unavailable in certain regions. Use the sidebar ticker map to replace any failing symbol with your preferred proxy. Paid feeds should be integrated only through licensed APIs or approved internal exports.

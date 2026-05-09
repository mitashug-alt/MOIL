# MOIL Macro Radar

Institutional regime-intelligence dashboard for MOIL Ltd and the Indian manganese/steel ecosystem.

This version is positioned as a **predictive regime analysis platform**, not a high-frequency trading bot or automatic buy/sell system.

## What this version adds

- Groq / Llama 3.3 primary AI scoring desk
- Manual macro scoring fallback
- Composite 0-100 regime score
- Signal confidence score
- Data quality score
- Volume confirmation layer using relative volume and volume Z-score
- Correlation engine and rolling MOIL correlation monitor
- Return and volume anomaly detection
- Telegram alerting with technical snapshots
- Institutional/news tracker
- NSE bulk/block sync placeholder with safe failure handling
- Validation tab for real-world testing discipline
- BDRY as default dry-bulk freight proxy

## Files

```text
app.py                     Streamlit dashboard
a macro_radar.py           Core analytics and scoring engine
groq_macro_scorer.py       Groq/Llama 3.3 scoring and commentary
telegram_alert.py          Telegram sender
requirements.txt           Dependencies
README.md                  Setup and operating guide
data/manual_macro_template.csv
data/news_tracker.csv
.streamlit/secrets.toml.example
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub Codespaces setup

Run:

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open the forwarded 8501 port.

## Secrets

Copy the example file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Fill locally only:

```toml
GROQ_API_KEY = ""
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_KEY_NAME = "MOIL Macro Radar Groq Key"
GROQ_PROJECT = "MOIL Macro Radar"

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
```

Never commit `.streamlit/secrets.toml`.

For Codespaces, use GitHub Codespaces secrets:

```text
GROQ_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

## Scoring framework

The dashboard calculates:

```text
Regime Score:        0-100
Confidence Score:    0-100
Data Quality Score:  0-100
```

Regime interpretation:

| Score | Regime |
|---:|---|
| 70-100 | Bullish industrial reflation |
| 55-69 | Constructive / watch confirmation |
| 45-54 | Neutral / mixed cycle |
| 0-44 | Risk-off commodity regime |

Manual macro scale:

| Score | Meaning |
|---:|---|
| +2.0 | Strongly bullish |
| +1.0 | Bullish / constructive |
| +0.5 | Mildly constructive |
| 0.0 | Neutral / mixed / unavailable |
| -0.5 | Mildly negative |
| -1.0 | Bearish |
| -2.0 | Severe stress |

## Groq AI scoring

The Groq AI desk reads:

- Manual macro table
- Market snapshot
- Regime context
- News/institutional tracker

It returns structured JSON:

```json
{
  "macro_scores": [
    {
      "indicator": "Silico-Manganese Prices",
      "score": -1.0,
      "status": "bearish",
      "rationale": "...",
      "confidence": 0.72,
      "data_quality": "medium"
    }
  ],
  "overall_manual_macro_score": 0.25,
  "regime_commentary": "...",
  "watch_items": ["..."],
  "risks": ["..."]
}
```

If Groq is missing or fails, the dashboard continues using manual macro scores and rule-based commentary.

## Telegram

Telegram is optional. The alert includes:

- Regime score
- Confidence and data quality
- MOIL technical levels
- Support/resistance
- 50DMA / 200DMA
- Top positives and pressures
- Manual macro scores
- Anomaly flags
- Optional Groq desk commentary

## Model-risk controls

This version directly addresses investor feedback:

- Not positioned for high-frequency or intraday scalping
- Adds volume confirmation
- Adds data quality scoring
- Adds confidence scoring
- Adds validation tab
- Keeps model transparent and auditable
- Makes stock-specific modeling a feature, not a weakness

## Notes

- This is a research dashboard, not investment advice.
- yfinance symbols can occasionally fail or be delayed.
- Paid data connectors such as Reuters, SteelMint, NSE licensed feeds or broker feeds should be connected only through approved/licensed APIs.

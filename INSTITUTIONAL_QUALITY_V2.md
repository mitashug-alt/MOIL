# Institutional Quality v2

This upgrade applies the same evidence-confidence discipline used in Data Quality v2 to MOIL smart-money tracking.

## Principle

Do not let a scraped institutional row move the regime score unless the data is material, exact, fresh and sourced from a high-authority source.

The new flow is:

```text
Institutional source -> raw cache -> canonical evidence row -> confidence score -> confidence-adjusted impact -> Groq commentary / regime context
```

## Confidence formula

Each institutional evidence row receives a 0-100 confidence score:

| Dimension | Points |
|---|---:|
| Source authority | 0-30 |
| Exactness | 0-20 |
| Freshness | 0-15 |
| Parser reliability | 0-15 |
| Cross-source confirmation | 0-10 |
| Materiality | 0-10 |

## Confidence gates

| Data confidence | Treatment |
|---:|---|
| 80-100 | Full scoring impact |
| 60-79 | 70% scoring impact |
| 40-59 | 40% scoring impact |
| <40 | Commentary only |

## Source hierarchy

| Source | Tier | Notes |
|---|---:|---|
| NSE official bulk/block archive | 1 | Official exchange data |
| NSE/BSE SAST and corporate filings | 1 | Official exchange disclosure data |
| MOIL official shareholding pattern | 1 | Company source |
| Screener/Trendlyne shareholding/deals | 3 | Useful aggregator data; confidence haircut |
| Manual upload of official exchange CSV/PDF | 1 | High quality if source metadata is provided |
| Manual upload of aggregator data | 3 | Use with haircut |

## Outputs

The institutional tracker now writes these fields into:

```text
data/cache/institutional_activity_latest.json
```

New fields:

```text
institutional_evidence_rows
institutional_source_readiness
smart_money_scorecard_v2
institutional_quality_summary
```

## Dashboard additions

The Institutional sync tab now shows:

```text
Institutional Quality v2
Institutional source readiness
Confidence-adjusted smart-money scorecard
Institutional evidence viewer
```

The Data quality tab also includes Institutional Quality v2 alongside macro evidence quality.

## Guardrails

This layer does not bypass CAPTCHA, logins, paywalls or bot-detection systems. It uses official CSV/downloads where possible, public pages where permitted, cached JSON, manual upload fallback and confidence scoring.

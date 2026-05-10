# Data Quality v2 Patch

This patch changes MOIL Macro Radar from broad scraping to a source-governed evidence pipeline.

## Principle

No source confidence means no regime impact. Groq scores only supplied evidence packs; it does not collect data by itself.

## Flow

```text
Scheduled scraper / verified upload
        -> canonical evidence row
        -> data confidence score
        -> Groq signal score
        -> confidence-adjusted effective score
        -> composite regime score
```

## Confidence formula

```text
source authority      0-30
data exactness        0-25
freshness             0-20
parser reliability    0-15
cross-source check    0-10
--------------------------
total confidence      0-100
```

## Regime gates

```text
>=80     full score impact
60-79    70% score impact
40-59    40% score impact
20-39    commentary-only, no regime impact
<20      ignored for scoring
```

## New app surfaces

- Source readiness v2 in Groq AI desk
- Data Quality Command Center
- Evidence Viewer
- Verified evidence CSV/JSON upload
- Confidence-adjusted Groq scoring
- Commentary-only treatment for weak sources

## Deployment check

After applying the patch:

```bash
pip install -r requirements.txt
python -m scrapers.run_daily_scrape
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Then use:

```text
Groq AI desk -> Pull auto macro feed -> Run Groq macro scoring -> Apply Groq scores
Data quality -> Evidence viewer
```

## Evidence upload

Use a CSV with columns from `schemas/evidence_row.schema.json`. Minimum useful columns:

```text
indicator,period,value,unit,source_name,source_type,source_tier,exact_data,publication_date,parser_method,parser_confidence,raw_text,data_confidence
```

If `data_confidence` is supplied, the app preserves it. Otherwise the confidence engine computes one.

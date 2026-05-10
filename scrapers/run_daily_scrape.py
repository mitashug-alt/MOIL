from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .html_table_scraper import DEFAULT_HTML_SOURCES, HtmlTableScraper
    from .india_steel_scraper import IndiaSteelScraper
    from .moil_pdf_scraper import MoilPdfScraper
except ImportError as exc:  # pragma: no cover
    message = str(exc)
    if "attempted relative import" in message or "no known parent package" in message:
        from html_table_scraper import DEFAULT_HTML_SOURCES, HtmlTableScraper
        from india_steel_scraper import IndiaSteelScraper
        from moil_pdf_scraper import MoilPdfScraper
    else:
        raise


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    Path("data/history").mkdir(parents=True, exist_ok=True)
    run_id = f"moil_macro_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    result = {"ok": True, "scrape_run_id": run_id, "started_at": now(), "moil": None, "html_sources": [], "errors": []}

    try:
        result["moil"] = MoilPdfScraper().scrape_latest()
    except Exception as exc:  # noqa: BLE001
        result["errors"].append({"source": "MOIL PDF", "error": str(exc)})

    try:
        result["html_sources"].append(IndiaSteelScraper().scrape_latest())
    except Exception as exc:  # noqa: BLE001
        result["errors"].append({"source": "India Steel Monthly Report", "error": str(exc)})

    html = HtmlTableScraper()
    for config in DEFAULT_HTML_SOURCES:
        try:
            result["html_sources"].append(html.scrape(config))
        except Exception as exc:  # noqa: BLE001
            result["errors"].append({"source": config.name, "error": str(exc)})

    result["finished_at"] = now()
    Path("data/cache/latest_macro_data.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with Path("data/history/scrape_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

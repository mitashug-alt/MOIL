from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pdfplumber
from bs4 import BeautifulSoup

try:
    from .http_client import ResilientHttpClient
except Exception:  # pragma: no cover
    from http_client import ResilientHttpClient


MONTHLY_SUMMARY_URL = "https://steel.gov.in/monthly-summary"


@dataclass
class IndiaSteelObservation:
    indicator: str
    country: str
    source_name: str
    source_url: str
    scrape_time: str
    period: str | None
    value: float | None
    unit: str | None
    raw_text: str
    extraction_method: str
    confidence: float
    page_hash: str


class IndiaSteelScraper:
    """Official Ministry of Steel / JPC monthly-economic-report scraper.

    The generic HTML scraper can miss India crude-steel data because the useful
    figures are often inside a linked PDF on the Ministry monthly-summary page.
    This scraper finds the latest Monthly Economic Report PDF, extracts text with
    pdfplumber, and emits an India Crude Steel Production cache row that the
    regime engine can pass into Groq.
    """

    def __init__(self, output_dir: str | Path = "data/raw/india_steel"):
        self.http = ResilientHttpClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or " ").strip()

    @staticmethod
    def extract_month_score(text: str, url: str) -> int:
        """Rough recency score from title/url text."""
        month_map = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        blob = f"{text} {url}".lower()
        year_match = re.search(r"20\d{2}", blob)
        year = int(year_match.group(0)) if year_match else 0
        month = 0
        for name, number in month_map.items():
            if name in blob:
                month = max(month, number)
        return year * 100 + month

    def discover_monthly_report_pdfs(self) -> list[dict[str, Any]]:
        response = self.http.get(MONTHLY_SUMMARY_URL, accept="text/html,*/*")
        if not response.ok:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        links: dict[str, dict[str, Any]] = {}
        for a in soup.find_all("a", href=True):
            href = str(a.get("href", "")).strip()
            text = a.get_text(" ", strip=True)
            blob = f"{text} {href}".lower()
            if ".pdf" not in blob:
                continue
            if "monthly" not in blob or "economic" not in blob:
                continue
            full_url = urljoin(MONTHLY_SUMMARY_URL, href)
            links[full_url] = {
                "url": full_url,
                "title": text or full_url.rsplit("/", 1)[-1],
                "score": self.extract_month_score(text, full_url),
            }
        return sorted(links.values(), key=lambda x: x.get("score", 0), reverse=True)

    def download_pdf(self, url: str) -> tuple[bytes, str]:
        result = self.http.get(url, accept="application/pdf,*/*")
        if not result.ok:
            raise RuntimeError(result.error or f"Could not download {url}")
        data = result.content
        doc_hash = self.sha256_bytes(data)
        out = self.output_dir / f"india_steel_monthly_{doc_hash[:12]}.pdf"
        out.write_bytes(data)
        return data, doc_hash

    def extract_text_from_pdf(self, data: bytes) -> str:
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)

    def parse_observations(self, text: str, source_url: str, doc_hash: str) -> list[IndiaSteelObservation]:
        compact = self.clean_text(text)
        observations: list[IndiaSteelObservation] = []
        scrape_time = self.now()

        patterns = [
            # The production of Crude Steel during April 2025-March 2026 is 168.4 million tonnes (Mt) (10.7% increase...)
            re.compile(
                r"production of\s+Crude Steel\s+during\s+(.{4,80}?)\s+is\s+([0-9]+(?:\.[0-9]+)?)\s*(?:million tonnes|mt|million tonne).*?\(([0-9]+(?:\.[0-9]+)?)%\s*(increase|decrease)",
                re.IGNORECASE,
            ),
            # Crude steel output ... 14.09 Mt ... 5.8% YoY
            re.compile(
                r"crude steel\s+(?:output|production).*?([0-9]+(?:\.[0-9]+)?)\s*(?:mt|million tonnes|million tonne).*?([0-9]+(?:\.[0-9]+)?)%\s*(?:yoy|year[-\s]*on[-\s]*year|increase)",
                re.IGNORECASE,
            ),
        ]

        for pattern in patterns:
            match = pattern.search(compact)
            if not match:
                continue
            groups = match.groups()
            if len(groups) >= 4:
                period = self.clean_text(groups[0])
                value = float(groups[1])
                yoy = float(groups[2])
                direction = groups[3].lower()
                yoy_signed = yoy if direction == "increase" else -yoy
                raw = match.group(0)
            else:
                period = None
                value = float(groups[0])
                yoy_signed = float(groups[1])
                raw = match.group(0)
            observations.append(
                IndiaSteelObservation(
                    indicator="India Crude Steel Production",
                    country="India",
                    source_name="Ministry of Steel Monthly Economic Report",
                    source_url=source_url,
                    scrape_time=scrape_time,
                    period=period,
                    value=value,
                    unit="Mt; YoY also in raw_text",
                    raw_text=f"{raw} | inferred_yoy_pct={yoy_signed}",
                    extraction_method="pdfplumber_text_regex",
                    confidence=0.85,
                    page_hash=doc_hash,
                )
            )
            break

        # Fallback: capture official report lines around crude steel even if numeric regex misses.
        if not observations and "crude steel" in compact.lower():
            idx = compact.lower().find("crude steel")
            snippet = compact[max(0, idx - 250): idx + 700]
            observations.append(
                IndiaSteelObservation(
                    indicator="India Crude Steel Production",
                    country="India",
                    source_name="Ministry of Steel Monthly Economic Report",
                    source_url=source_url,
                    scrape_time=scrape_time,
                    period=None,
                    value=None,
                    unit="official report snippet",
                    raw_text=snippet,
                    extraction_method="pdfplumber_text_snippet",
                    confidence=0.55,
                    page_hash=doc_hash,
                )
            )
        return observations

    def scrape_latest(self) -> dict[str, Any]:
        candidates = self.discover_monthly_report_pdfs()
        diagnostics: list[str] = []
        if not candidates:
            return {
                "ok": False,
                "source": {
                    "name": "Ministry of Steel Monthly Economic Report",
                    "url": MONTHLY_SUMMARY_URL,
                    "indicator": "India Crude Steel Production",
                    "country": "India",
                    "frequency": "monthly",
                    "keywords": ["crude steel", "production", "India", "Mt"],
                    "source_quality": "official Ministry/JPC monthly report",
                    "transform_hint": "Find latest monthly report PDF and extract crude steel production.",
                },
                "error": "No Monthly Economic Report PDF candidates found.",
                "diagnostics": diagnostics,
                "observations": [],
                "observation_count": 0,
            }

        for candidate in candidates[:5]:
            url = candidate["url"]
            try:
                data, doc_hash = self.download_pdf(url)
                text = self.extract_text_from_pdf(data)
                observations = self.parse_observations(text, url, doc_hash)
                return {
                    "ok": True,
                    "source": {
                        "name": "Ministry of Steel Monthly Economic Report",
                        "url": url,
                        "indicator": "India Crude Steel Production",
                        "country": "India",
                        "frequency": "monthly",
                        "keywords": ["crude steel", "production", "India", "Mt"],
                        "source_quality": "official Ministry/JPC monthly report",
                        "transform_hint": "Extract crude steel production and YoY from latest Ministry monthly economic report PDF.",
                    },
                    "selected_pdf": url,
                    "candidate_count": len(candidates),
                    "diagnostics": diagnostics,
                    "observation_count": len(observations),
                    "observations": [asdict(x) for x in observations],
                }
            except Exception as exc:  # noqa: BLE001
                diagnostics.append(f"Failed {url}: {exc}")
        return {
            "ok": False,
            "source": {
                "name": "Ministry of Steel Monthly Economic Report",
                "url": MONTHLY_SUMMARY_URL,
                "indicator": "India Crude Steel Production",
                "country": "India",
                "frequency": "monthly",
                "keywords": ["crude steel", "production", "India", "Mt"],
                "source_quality": "official Ministry/JPC monthly report",
                "transform_hint": "Find latest monthly report PDF and extract crude steel production.",
            },
            "error": "All Ministry monthly report PDF candidates failed.",
            "diagnostics": diagnostics,
            "observations": [],
            "observation_count": 0,
        }


def main() -> None:
    result = IndiaSteelScraper().scrape_latest()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

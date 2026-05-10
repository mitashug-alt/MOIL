from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import pdfplumber
from bs4 import BeautifulSoup

try:
    from .http_client import ResilientHttpClient
except Exception:  # pragma: no cover
    from http_client import ResilientHttpClient


@dataclass
class PriceObservation:
    indicator: str
    company: str
    product_family: str | None
    ore_code: str | None
    grade: str | None
    mn_pct: str | None
    price_value: float | None
    currency: str
    unit: str
    basis: str
    effective_date: str | None
    observation_date: str
    source_name: str
    source_url: str
    document_hash: str
    extraction_method: str
    confidence: float
    raw_text: str


@dataclass
class CategoryRevision:
    category: str
    direction: str
    change_pct: float | None
    effective_date: str | None
    source_url: str
    raw_text: str
    confidence: float


class MoilPdfScraper:
    """Find public MOIL PDFs and extract visible Basic Price PMT facts.

    The scraper never bypasses password protection. Password-protected Annexure
    files are recorded as unavailable, and visible exchange summary filings are
    used as lower-granularity evidence.
    """

    START_URLS = [
        "https://www.moil.nic.in/",
        "https://moil.nic.in/",
        "https://www.moil.nic.in/content/205/Direct-Sales",
        "https://moil.nic.in/content/205/Direct-Sales",
    ]
    HINTS = ["price", "circular", "direct", "sales", "manganese", "emd", "annexure"]
    ORE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\d{2,6}\b")
    PRICE_RE = re.compile(r"(?:Rs\.?|₹|INR)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{5,7})(?:\s*/-)?\s*(?:PMT|P\.M\.T|per\s+MT|/MT)?", re.I)
    EFFECTIVE_RE = re.compile(r"(?:effective\s+from|w\.?e\.?f\.?|with\s+effect\s+from)\s*(?:midnight\s+of\s*)?([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})", re.I)
    CATEGORY_PATTERNS = {
        "Ferro Grade Mn 44%+": r"Ferro grades?.{0,90}Mn[-\s]*44% and above.{0,180}?(increased|decreased|continued).{0,80}?([0-9]+(?:\.[0-9]+)?)?%",
        "Ferro Grade below Mn 44%": r"other Ferro grades?.{0,120}below Mn[-\s]*44%.{0,180}?(increased|decreased|continued).{0,80}?([0-9]+(?:\.[0-9]+)?)?%",
        "SMGR Mn 30%": r"SMGR.{0,50}Mn[-\s]*30%.{0,180}?(increased|decreased|continued).{0,80}?([0-9]+(?:\.[0-9]+)?)?%",
        "SMGR Mn 25%": r"SMGR.{0,50}Mn[-\s]*25%.{0,180}?(increased|decreased|continued).{0,80}?([0-9]+(?:\.[0-9]+)?)?%",
        "Chemical Grade": r"Chemical grades?.{0,180}?(increased|decreased|continued).{0,80}?([0-9]+(?:\.[0-9]+)?)?%",
        "Fines": r"Fines grades?.{0,180}?(increased|decreased|continued).{0,80}?([0-9]+(?:\.[0-9]+)?)?%",
    }

    def __init__(self, output_dir: str | Path = "data/raw/moil_pdfs"):
        self.http = ResilientHttpClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def discover_pdf_links(self) -> list[dict[str, Any]]:
        links: dict[str, dict[str, Any]] = {}
        for start in self.START_URLS:
            result = self.http.get(start, accept="text/html,*/*")
            if not result.ok:
                continue
            soup = BeautifulSoup(result.text, "html.parser")
            for tag in soup.find_all("a", href=True):
                href = str(tag.get("href") or "").strip()
                url = urljoin(start, href)
                text = tag.get_text(" ", strip=True)
                haystack = f"{text} {url}".lower()
                if ".pdf" in haystack and any(h in haystack for h in self.HINTS):
                    links[url] = {"url": url, "text": text, "source_page": start, "score": self._score_link(text, url)}
        return sorted(links.values(), key=lambda x: x.get("score", 0), reverse=True)

    def _score_link(self, text: str, url: str) -> int:
        haystack = f"{text} {url}".lower()
        score = 0
        for word in ["price", "pricing", "circular", "direct", "sales"]:
            score += 10 if word in haystack else 0
        for word in ["emd", "manganese", "annexure"]:
            score += 5 if word in haystack else 0
        for m in re.finditer(r"\b([0-3]?\d)[./-]?([01]?\d)[./-]?(20\d{2})\b", haystack):
            try:
                score += int(m.group(3)) - 2020
            except Exception:
                pass
        return score

    def download_pdf(self, url: str) -> tuple[bytes, str, str]:
        result = self.http.get(url, accept="application/pdf,*/*")
        if not result.ok or not result.content:
            raise RuntimeError(f"Could not download PDF {url}: {result.error}")
        h = self.sha256(result.content)
        filename = Path(urlparse(url).path).name or f"moil_{h[:12]}.pdf"
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        out_path = self.output_dir / filename
        out_path.write_bytes(result.content)
        return result.content, h, str(out_path)

    def parse_pdf(self, pdf_bytes: bytes, url: str, doc_hash: str) -> dict[str, Any]:
        text_parts: list[str] = []
        table_rows: list[str] = []
        encrypted = False
        parse_status = "ok"
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                encrypted = bool(getattr(pdf, "is_encrypted", False))
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
                    for table in page.extract_tables() or []:
                        for row in table:
                            line = " ".join(str(cell or "").strip() for cell in row).strip()
                            if line:
                                table_rows.append(line)
        except Exception as exc:
            parse_status = f"pdf_parse_failed: {exc}"
            encrypted = "password" in str(exc).lower() or "encrypt" in str(exc).lower()
        full_text = "\n".join(text_parts)
        return {
            "source_document": {
                "source_name": "MOIL PDF circular",
                "source_url": url,
                "discovered_at": self.now(),
                "document_hash": doc_hash,
                "document_date": self.extract_effective_date(full_text),
                "encrypted": encrypted,
                "parse_status": parse_status,
                "raw_text_preview": full_text[:1200],
            },
            "price_observations": [asdict(x) for x in self.extract_price_observations(full_text, table_rows, url, doc_hash)],
            "category_revisions": [asdict(x) for x in self.extract_category_revisions(full_text, url)],
        }

    def extract_effective_date(self, text: str) -> str | None:
        m = self.EFFECTIVE_RE.search(text or "")
        if not m:
            return None
        raw = m.group(1).replace(".", "-").replace("/", "-")
        parts = raw.split("-")
        if len(parts) != 3:
            return raw
        day, month, year = parts
        year = f"20{year}" if len(year) == 2 else year
        try:
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        except Exception:
            return raw

    def classify_product(self, line: str) -> str:
        lower = line.lower()
        if "emd" in lower or "electrolytic manganese dioxide" in lower:
            return "EMD"
        if "smgr" in lower:
            if "30" in lower:
                return "SMGR Mn 30%"
            if "25" in lower:
                return "SMGR Mn 25%"
            return "SMGR"
        if "ferro" in lower or "mn-44" in lower or "mn 44" in lower:
            return "Ferro Grade"
        if "chemical" in lower:
            return "Chemical Grade"
        if "fine" in lower:
            return "Fines"
        return "Manganese Ore"

    def extract_mn_pct(self, line: str) -> str | None:
        m = re.search(r"Mn[-\s]*(\d{2}(?:\.\d+)?)%", line, re.I) or re.search(r"(\d{2}(?:\.\d+)?)%\s*Mn", line, re.I)
        return f"{m.group(1)}%" if m else None

    def extract_price(self, line: str) -> float | None:
        m = self.PRICE_RE.search(line or "")
        if not m:
            return None
        try:
            value = float(m.group(1).replace(",", ""))
            return value if value >= 1000 else None
        except Exception:
            return None

    def extract_price_observations(self, full_text: str, table_rows: list[str], url: str, doc_hash: str) -> list[PriceObservation]:
        obs: list[PriceObservation] = []
        effective = self.extract_effective_date(full_text)
        for line in table_rows:
            code = self.ORE_CODE_RE.search(line)
            price = self.extract_price(line)
            if not code and price is None and "emd" not in line.lower():
                continue
            obs.append(PriceObservation(
                indicator="MOIL Basic Price PMT",
                company="MOIL Ltd",
                product_family=self.classify_product(line),
                ore_code=code.group(0) if code else ("EMD" if "emd" in line.lower() else None),
                grade=None,
                mn_pct=self.extract_mn_pct(line),
                price_value=price,
                currency="INR",
                unit="PMT",
                basis="Basic Price PMT",
                effective_date=effective,
                observation_date=self.now(),
                source_name="MOIL PDF table",
                source_url=url,
                document_hash=doc_hash,
                extraction_method="pdfplumber_table_regex",
                confidence=0.85 if price is not None else 0.55,
                raw_text=line[:1200],
            ))
        for pattern in [
            r"basic price\s*(?:Rs\.?|₹|INR)?\s*([0-9,]+)\s*/?-?\s*PMT\s*of\s*EMD",
            r"EMD.{0,80}basic price.{0,80}(?:Rs\.?|₹|INR)?\s*([0-9,]+)",
        ]:
            m = re.search(pattern, full_text or "", re.I | re.S)
            if m:
                obs.append(PriceObservation(
                    indicator="MOIL EMD Basic Price PMT",
                    company="MOIL Ltd",
                    product_family="EMD",
                    ore_code="EMD",
                    grade="Electrolytic Manganese Dioxide",
                    mn_pct=None,
                    price_value=float(m.group(1).replace(",", "")),
                    currency="INR",
                    unit="PMT",
                    basis="Basic Price PMT",
                    effective_date=effective,
                    observation_date=self.now(),
                    source_name="MOIL PDF text",
                    source_url=url,
                    document_hash=doc_hash,
                    extraction_method="pdfplumber_text_regex",
                    confidence=0.90,
                    raw_text=m.group(0)[:1200],
                ))
                break
        return obs

    def extract_category_revisions(self, full_text: str, url: str) -> list[CategoryRevision]:
        compact = re.sub(r"\s+", " ", full_text or "")
        effective = self.extract_effective_date(full_text)
        rows: list[CategoryRevision] = []
        for cat, pattern in self.CATEGORY_PATTERNS.items():
            m = re.search(pattern, compact, re.I)
            if not m:
                continue
            pct = None
            if len(m.groups()) >= 2 and m.group(2):
                try:
                    pct = float(m.group(2))
                except Exception:
                    pct = None
            rows.append(CategoryRevision(cat, m.group(1).lower(), pct, effective, url, m.group(0)[:1200], 0.80 if pct is not None else 0.60))
        return rows

    def scrape_latest(self, seed_pdf_urls: Iterable[str] | None = None) -> dict[str, Any]:
        candidates = [{"url": u, "score": 999, "text": "seed"} for u in (seed_pdf_urls or [])]
        candidates.extend(self.discover_pdf_links())
        diagnostics: list[str] = []
        for item in candidates:
            url = item.get("url")
            if not url:
                continue
            try:
                data, h, saved_path = self.download_pdf(url)
                parsed = self.parse_pdf(data, url, h)
                return {"ok": True, "selected_pdf": url, "saved_path": saved_path, "candidate_count": len(candidates), "diagnostics": diagnostics, **parsed}
            except Exception as exc:
                diagnostics.append(f"{url}: {exc}")
        return {"ok": False, "error": "No usable MOIL PDF found", "diagnostics": diagnostics, "price_observations": [], "category_revisions": []}


def main() -> None:
    result = MoilPdfScraper().scrape_latest()
    out_dir = Path("data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "moil_latest_price_data.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

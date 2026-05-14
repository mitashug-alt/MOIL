from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

try:
    from .http_client import ResilientHttpClient
except Exception:  # pragma: no cover
    from http_client import ResilientHttpClient


@dataclass
class HtmlSourceConfig:
    name: str
    url: str
    indicator: str
    country: str
    frequency: str
    keywords: list[str]
    source_quality: str
    transform_hint: str


@dataclass
class HtmlObservation:
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


class HtmlTableScraper:
    def __init__(self):
        self.http = ResilientHttpClient()

    @staticmethod
    def now() -> str:
        return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).isoformat()

    @staticmethod
    def page_hash(html: str) -> str:
        return hashlib.sha256((html or "").encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def keyword_score(text: str, keywords: list[str]) -> int:
        lower = (text or "").lower()
        return sum(1 for kw in keywords if kw.lower() in lower)

    @staticmethod
    def visible_text(html: str) -> str:
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer"]):
            tag.decompose()
        lines = [re.sub(r"\s+", " ", x).strip() for x in soup.get_text("\n").splitlines()]
        return "\n".join(x for x in lines if x)

    @staticmethod
    def extract_period(text: str) -> str | None:
        m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2}\b", text or "", re.I)
        if m:
            return m.group(0)
        m = re.search(r"\b20\d{2}[-/](0?[1-9]|1[0-2])\b", text or "")
        return m.group(0) if m else None

    @staticmethod
    def extract_number(text: str) -> float | None:
        matches = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", text or "")
        if not matches:
            return None
        try:
            return float(matches[-1].replace(",", ""))
        except Exception:
            return None

    def parse_tables(self, html: str, config: HtmlSourceConfig) -> list[HtmlObservation]:
        obs: list[HtmlObservation] = []
        h = self.page_hash(html)
        try:
            tables = pd.read_html(io.StringIO(html))
        except Exception:
            tables = []
        for table in tables:
            if table is None or table.empty:
                continue
            table_text = table.astype(str).to_string(index=False)
            if self.keyword_score(table_text, config.keywords) == 0:
                continue
            for _, row in table.head(80).iterrows():
                raw = " | ".join(str(x) for x in row.tolist())
                if self.keyword_score(raw, config.keywords) == 0:
                    continue
                obs.append(HtmlObservation(config.indicator, config.country, config.name, config.url, self.now(), self.extract_period(raw), self.extract_number(raw), None, raw[:1200], "pandas_read_html", 0.75, h))
        return obs

    def parse_visible_text(self, html: str, config: HtmlSourceConfig) -> list[HtmlObservation]:
        text = self.visible_text(html)
        h = self.page_hash(html)
        obs: list[HtmlObservation] = []
        for line in [x for x in text.splitlines() if self.keyword_score(x, config.keywords) > 0][:25]:
            obs.append(HtmlObservation(config.indicator, config.country, config.name, config.url, self.now(), self.extract_period(line), self.extract_number(line), None, line[:1200], "beautifulsoup_visible_text", 0.45, h))
        return obs

    def scrape(self, config: HtmlSourceConfig) -> dict[str, Any]:
        result = self.http.get(config.url, accept="text/html,*/*")
        if not result.ok:
            return {"ok": False, "source": asdict(config), "error": result.error, "observations": []}
        table_obs = self.parse_tables(result.text, config)
        text_obs = self.parse_visible_text(result.text, config)
        observations = table_obs or text_obs
        return {"ok": True, "source": asdict(config), "observation_count": len(observations), "observations": [asdict(x) for x in observations]}


DEFAULT_HTML_SOURCES = [
    HtmlSourceConfig(
        name="BigMint Silico Manganese Prices",
        url="https://www.bigmint.co/prices/ferrous/ferro-alloy/silico-manganese",
        indicator="Silico-Manganese Prices",
        country="India",
        frequency="daily",
        keywords=["silico manganese", "simn", "raipur", "durgapur", "vizag", "60-14"],
        source_quality="industry price page; exact values may be gated",
        transform_hint="Extract visible India SiMn HC 60-14 regional rows; if exact price is gated, retain timestamp/region facts only.",
    ),
    HtmlSourceConfig(
        name="OfBusiness Silico Manganese Category",
        url="https://www.ofbusiness.com/category/silico-manganese",
        indicator="Silico-Manganese Prices",
        country="India",
        frequency="daily",
        keywords=["silico manganese", "price", "mt", "raipur", "durgapur", "vizag"],
        source_quality="public B2B raw material page; asking-price proxy",
        transform_hint="Extract visible SiMn category snippets; treat supplier/asking prices as lower-confidence than assessed prices.",
    ),
    HtmlSourceConfig(
        name="Worldsteel India Crude Steel Production",
        url="https://worldsteel.org/media/press-releases/",
        indicator="India Crude Steel Production",
        country="India",
        frequency="monthly",
        keywords=["crude steel production", "india", "mt"],
        source_quality="industry association monthly production release",
        transform_hint="Find latest monthly crude-steel production release and India-specific snippets.",
    ),
    HtmlSourceConfig(
        name="MacroMicro China Crude Steel Production",
        url="https://en.macromicro.me/collections/20/mm-steel/221/china-crude-steel-production",
        indicator="China Crude Steel Production",
        country="China",
        frequency="monthly",
        keywords=["crude steel", "production", "china"],
        source_quality="public chart/table proxy",
        transform_hint="Extract latest monthly China crude-steel production if visible.",
    ),
    HtmlSourceConfig(
        name="Worldsteel Press Releases",
        url="https://worldsteel.org/media/press-releases/",
        indicator="Global / India / China Crude Steel Production",
        country="Global",
        frequency="monthly",
        keywords=["crude steel production", "india", "china", "mt"],
        source_quality="industry association release page",
        transform_hint="Find latest monthly crude-steel production release and relevant rows/snippets.",
    ),
    HtmlSourceConfig(
        name="China Data Portal Customs / Steel",
        url="https://chinadata.live/",
        indicator="China Steel Exports",
        country="China",
        frequency="monthly",
        keywords=["customs", "trade", "steel", "exports"],
        source_quality="official-data aggregation portal",
        transform_hint="Extract customs trade-flow evidence for steel exports if visible.",
    ),
    HtmlSourceConfig(
        name="NBS China English Press Releases",
        url="https://www.stats.gov.cn/english/PressRelease/",
        indicator="China Power / Industrial Stress",
        country="China",
        frequency="monthly",
        keywords=["electricity", "industrial production", "power generation", "steel", "cement"],
        source_quality="official China statistics release page",
        transform_hint="Extract power/industrial snippets as stress proxy.",
    ),
]


def main() -> None:
    scraper = HtmlTableScraper()
    out = [scraper.scrape(config) for config in DEFAULT_HTML_SOURCES]
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

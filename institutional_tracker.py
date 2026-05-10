from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SYMBOL = "MOIL"
COMPANY_NAME = "MOIL Ltd"
TRENDLYNE_MOIL_DEALS_URL = "https://trendlyne.com/equity/bulk-block-deals/MOIL/872/moil-ltd/"
SCREENER_MOIL_URL = "https://www.screener.in/company/MOIL/"
MOIL_SHAREHOLDING_URL = "https://www.moil.nic.in/content/149/Shareholding-Pattern"
NSE_BULK_ARCHIVE_URLS = [
    ("bulk", "https://nsearchives.nseindia.com/content/equities/bulk.csv"),
    ("block", "https://nsearchives.nseindia.com/content/equities/block.csv"),
]
WATCHLIST = [
    "Life Insurance Corporation", "LIC", "Bandhan", "Quant", "Nippon", "HDFC", "ICICI Prudential",
    "SBI Mutual Fund", "Kotak", "Aditya Birla", "Motilal Oswal", "Axis Mutual Fund", "Graviton",
    "Graviton Research Capital", "iShares", "BlackRock", "Vanguard", "Government of India", "Ministry of Steel",
]
OUTSTANDING_SHARES = int(os.getenv("MOIL_OUTSTANDING_SHARES", "0") or "0")


@dataclass
class InstitutionalDeal:
    observation_id: str
    source_name: str
    source_url: str
    symbol: str
    company_name: str
    exchange: str | None
    trade_date: str | None
    client_name: str
    transaction_type: str
    quantity: float | None
    price: float | None
    value_inr: float | None
    deal_type: str
    scraped_at: str
    confidence: float
    raw_text: str


@dataclass
class ShareholdingObservation:
    observation_id: str
    source_name: str
    source_url: str
    symbol: str
    period: str | None
    holder_category: str
    holding_pct: float | None
    previous_period: str | None
    previous_holding_pct: float | None
    delta_pct_points: float | None
    scraped_at: str
    confidence: float
    raw_text: str


@dataclass
class SmartMoneySignal:
    signal_id: str
    symbol: str
    signal_type: str
    severity: str
    institution: str | None
    description: str
    score_impact: float
    created_at: str
    evidence: list[dict[str, Any]]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def h(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8", errors="ignore")).hexdigest()[:24]


def clean_number(value: Any) -> float | None:
    text = str(value or "").replace(",", "").replace("%", "").strip()
    if text in {"", "-", "nan", "None"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def side(text: str) -> str:
    t = str(text or "").upper()
    if "BUY" in t or "PURCHASE" in t or "ACQUIRE" in t:
        return "BUY"
    if "SELL" in t or "SALE" in t or "DISPOS" in t:
        return "SELL"
    return "UNKNOWN"


def watch_match(name: str) -> str | None:
    lower = str(name or "").lower()
    for item in WATCHLIST:
        if item.lower() in lower:
            return item
    return None


class SmartMoneyTracker:
    """Free/public smart-money tracker.

    Uses official CSV/public HTML where available. It does not bypass CAPTCHA,
    login walls, paywalls or bot-detection systems. If a source blocks cloud
    access, the output records diagnostics and your app can use manual upload.
    """

    def __init__(self, output_dir: str | Path = "data/cache", outstanding_shares: int = OUTSTANDING_SHARES):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = Path("data/history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.outstanding_shares = outstanding_shares

    def get(self, url: str) -> str:
        headers = {
            "User-Agent": "MOIL-Macro-Radar/1.0 scheduled-research-scraper",
            "Accept": "text/html,text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = self.session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text or ""

    def find_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        for cand in candidates:
            for col in df.columns:
                if cand.upper() == str(col).upper().strip() or cand.upper() in str(col).upper():
                    return col
        return None

    def scrape_nse_archives(self) -> tuple[list[InstitutionalDeal], list[str]]:
        deals: list[InstitutionalDeal] = []
        diagnostics: list[str] = []
        for deal_type, url in NSE_BULK_ARCHIVE_URLS:
            try:
                text = self.get(url)
                if "SYMBOL" not in text.upper():
                    diagnostics.append(f"{deal_type}: loaded but no SYMBOL header")
                    continue
                df = pd.read_csv(io.StringIO(text))
                df.columns = [str(c).strip() for c in df.columns]
                symbol_col = self.find_col(df, ["SYMBOL", "SCRIP"])
                if not symbol_col:
                    diagnostics.append(f"{deal_type}: symbol column missing")
                    continue
                rows = df[df[symbol_col].astype(str).str.upper().eq(SYMBOL)]
                diagnostics.append(f"{deal_type}: loaded {len(df)} rows; MOIL rows={len(rows)}")
                client_col = self.find_col(df, ["CLIENT NAME", "CLIENT", "NAME"])
                side_col = self.find_col(df, ["BUY/SELL", "TYPE", "TRANSACTION"])
                qty_col = self.find_col(df, ["QUANTITY", "QTY"])
                price_col = self.find_col(df, ["PRICE", "AVG PRICE", "TRADE PRICE"])
                date_col = self.find_col(df, ["DATE", "TRADE DATE"])
                for _, row in rows.iterrows():
                    client = str(row.get(client_col, "")).strip() if client_col else "Unknown"
                    tx = side(str(row.get(side_col, ""))) if side_col else "UNKNOWN"
                    qty = clean_number(row.get(qty_col)) if qty_col else None
                    price = clean_number(row.get(price_col)) if price_col else None
                    trade_date = str(row.get(date_col, "")).strip() if date_col else None
                    value = qty * price if qty is not None and price is not None else None
                    raw = " | ".join(str(x) for x in row.tolist())
                    deals.append(InstitutionalDeal(h("nse", url, trade_date, client, tx, qty, price), "NSE Bulk/Block Archive", url, SYMBOL, COMPANY_NAME, "NSE", trade_date, client, tx, qty, price, value, deal_type, now(), 0.90, raw[:1200]))
            except Exception as exc:
                diagnostics.append(f"{deal_type}: failed {exc}")
        return deals, diagnostics

    def scrape_trendlyne_table(self) -> tuple[list[InstitutionalDeal], list[str]]:
        deals: list[InstitutionalDeal] = []
        diagnostics: list[str] = []
        try:
            html = self.get(TRENDLYNE_MOIL_DEALS_URL)
            tables = pd.read_html(io.StringIO(html))
            diagnostics.append(f"Trendlyne: parsed {len(tables)} HTML table(s)")
            for table in tables:
                df = table.copy()
                df.columns = [str(c).strip() for c in df.columns]
                date_col = self.find_col(df, ["DATE"])
                client_col = self.find_col(df, ["CLIENT", "NAME"])
                side_col = self.find_col(df, ["TYPE", "BUY/SELL", "TRANSACTION"])
                qty_col = self.find_col(df, ["QUANTITY", "QTY"])
                price_col = self.find_col(df, ["PRICE", "AVG"])
                for _, row in df.iterrows():
                    raw = " | ".join(str(x) for x in row.tolist())
                    if "BUY" not in raw.upper() and "SELL" not in raw.upper():
                        continue
                    client = str(row.get(client_col, "")).strip() if client_col else "Unknown"
                    qty = clean_number(row.get(qty_col)) if qty_col else None
                    price = clean_number(row.get(price_col)) if price_col else None
                    tx = side(str(row.get(side_col, raw))) if side_col else side(raw)
                    trade_date = str(row.get(date_col, "")).strip() if date_col else None
                    value = qty * price if qty is not None and price is not None else None
                    deals.append(InstitutionalDeal(h("trendlyne", trade_date, client, tx, qty, price), "Trendlyne MOIL Bulk/Block Deals", TRENDLYNE_MOIL_DEALS_URL, SYMBOL, COMPANY_NAME, None, trade_date, client or "Unknown", tx, qty, price, value, "Bulk/Block", now(), 0.70, raw[:1200]))
        except Exception as exc:
            diagnostics.append(f"Trendlyne failed or blocked: {exc}")
        return deals, diagnostics

    def scrape_shareholding(self) -> tuple[list[ShareholdingObservation], list[str]]:
        observations: list[ShareholdingObservation] = []
        diagnostics: list[str] = []
        for source_name, url in [("Screener MOIL", SCREENER_MOIL_URL), ("MOIL Official Shareholding", MOIL_SHAREHOLDING_URL)]:
            try:
                html = self.get(url)
                tables = pd.read_html(io.StringIO(html))
                diagnostics.append(f"{source_name}: parsed {len(tables)} tables")
                for table in tables:
                    text = table.astype(str).to_string(index=False).lower()
                    if not any(k in text for k in ["fii", "dii", "foreign", "mutual", "insurance", "promoter", "public"]):
                        continue
                    for _, row in table.iterrows():
                        raw = " | ".join(str(x) for x in row.tolist())
                        category = next((c for c in ["FII", "Foreign", "DII", "Mutual Funds", "Insurance", "Promoter", "Public"] if c.lower() in raw.lower()), None)
                        if not category:
                            continue
                        vals = [clean_number(x) for x in row.tolist()]
                        vals = [x for x in vals if x is not None and 0 <= x <= 100]
                        if not vals:
                            continue
                        latest = vals[0]
                        previous = vals[1] if len(vals) > 1 else None
                        delta = latest - previous if latest is not None and previous is not None else None
                        observations.append(ShareholdingObservation(h(source_name, category, latest, previous, raw), source_name, url, SYMBOL, None, category, latest, None, previous, delta, now(), 0.55, raw[:1200]))
            except Exception as exc:
                diagnostics.append(f"{source_name}: failed {exc}")
        unique = {x.observation_id: x for x in observations}
        return list(unique.values()), diagnostics

    def generate_signals(self, deals: list[InstitutionalDeal], shareholding: list[ShareholdingObservation]) -> list[SmartMoneySignal]:
        signals: list[SmartMoneySignal] = []
        for d in deals:
            inst = watch_match(d.client_name)
            if not inst:
                continue
            pct = d.quantity / self.outstanding_shares * 100 if self.outstanding_shares and d.quantity else None
            if d.transaction_type == "SELL" and pct is not None and pct >= 1.0:
                sev, impact, desc = "High-Stress", -2.0, f"{inst} sold about {pct:.2f}% of equity in a bulk/block deal."
            elif d.transaction_type == "BUY" and pct is not None and pct >= 0.5:
                sev, impact, desc = "Confirmation", 1.0, f"{inst} bought about {pct:.2f}% of equity in a bulk/block deal."
            elif d.transaction_type == "SELL":
                sev, impact, desc = "Watch", -0.5, f"{inst} appeared on sell side in a bulk/block deal."
            elif d.transaction_type == "BUY":
                sev, impact, desc = "Watch", 0.5, f"{inst} appeared on buy side in a bulk/block deal."
            else:
                continue
            signals.append(SmartMoneySignal(h("deal_signal", d.observation_id), SYMBOL, "Institutional Deal", sev, inst, desc, impact, now(), [asdict(d)]))
        for obs in shareholding:
            delta = obs.delta_pct_points
            if delta is None:
                continue
            cat = obs.holder_category
            if delta <= -1.0 and any(k in cat.lower() for k in ["fii", "foreign", "dii", "mutual", "insurance"]):
                signals.append(SmartMoneySignal(h("holding_drop", obs.observation_id), SYMBOL, "Quarterly Institutional Holding Drop", "Risk Warning", cat, f"{cat} holding fell by {delta:.2f} percentage points QoQ.", -1.0, now(), [asdict(obs)]))
            if delta >= 1.0 and any(k in cat.lower() for k in ["fii", "foreign", "dii", "mutual", "insurance"]):
                signals.append(SmartMoneySignal(h("holding_rise", obs.observation_id), SYMBOL, "Quarterly Institutional Holding Increase", "Confirmation", cat, f"{cat} holding rose by {delta:.2f} percentage points QoQ.", 1.0, now(), [asdict(obs)]))
        return signals

    def run(self) -> dict[str, Any]:
        nse_deals, nse_diag = self.scrape_nse_archives()
        trend_deals, trend_diag = self.scrape_trendlyne_table()
        shareholding, share_diag = self.scrape_shareholding()
        deals = list({d.observation_id: d for d in (nse_deals + trend_deals)}.values())
        signals = self.generate_signals(deals, shareholding)
        result = {
            "ok": True,
            "symbol": SYMBOL,
            "company_name": COMPANY_NAME,
            "scraped_at": now(),
            "diagnostics": nse_diag + trend_diag + share_diag,
            "source_status": {"nse_deals_count": len(nse_deals), "trendlyne_deals_count": len(trend_deals), "shareholding_rows_count": len(shareholding)},
            "institutional_deals": [asdict(x) for x in deals],
            "shareholding_observations": [asdict(x) for x in shareholding],
            "sast_filings": [],
            "smart_money_signals": [asdict(x) for x in signals],
        }
        Path("data/cache").mkdir(parents=True, exist_ok=True)
        Path("data/cache/institutional_activity_latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        with Path("data/history/institutional_activity_runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result


def main() -> None:
    result = SmartMoneyTracker().run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

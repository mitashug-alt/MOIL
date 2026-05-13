from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests import Session

from .config import TrackerConfig

NSE_BASE = "https://www.nseindia.com"


@dataclass(slots=True)
class NSEClient:
    config: TrackerConfig
    session: Session
    _last_cookie_refresh: float = 0.0

    @classmethod
    def create(cls, config: TrackerConfig) -> "NSEClient":
        sess = requests.Session()
        headers = {
            "User-Agent": config.user_agent,
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": config.accept_language,
            "Connection": "keep-alive",
            "Referer": NSE_BASE,
        }
        headers.update(config.extra_headers or {})
        sess.headers.update(headers)
        return cls(config=config, session=sess)

    def _refresh_cookies(self) -> None:
        # hit base page to get cookies
        now = time.time()
        if now - self._last_cookie_refresh < 300:
            return
        try:
            self.session.get(NSE_BASE, timeout=self.config.request_timeout)
            self._last_cookie_refresh = now
        except Exception:
            pass

    def nse_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> tuple[bool, str, str]:
        url = endpoint if endpoint.startswith("http") else f"{NSE_BASE}{endpoint}"
        if not self.config.enable_network:
            return False, "network disabled", "network disabled"
        self._refresh_cookies()
        try:
            resp = self.session.get(url, params=params, timeout=self.config.request_timeout)
            if resp.status_code in {401, 403}:
                # refresh cookies once and retry
                self._refresh_cookies()
                resp = self.session.get(url, params=params, timeout=self.config.request_timeout)
            if resp.status_code != 200:
                return False, "", f"HTTP {resp.status_code}"
            return True, resp.text, f"HTTP 200 {len(resp.text)} bytes"
        except Exception as exc:  # noqa: BLE001
            return False, "", f"error: {exc}"

    def fetch_nse_deals(self, option_type: str, from_date: str, to_date: str) -> tuple[pd.DataFrame, list[str]]:
        diagnostics: list[str] = []
        url = f"/api/historicalOR/bulk-block-short-deals"
        ok, text, detail = self.nse_get(url, params={"optionType": option_type, "from": from_date, "to": to_date})
        diagnostics.append(f"{option_type}: {detail}")
        if not ok or not text:
            return pd.DataFrame(), diagnostics
        try:
            df = pd.read_json(io.StringIO(text)) if text.strip().startswith("{") else pd.read_csv(io.StringIO(text))
        except Exception:
            try:
                df = pd.read_csv(io.StringIO(text))
            except Exception as exc:  # noqa: BLE001
                diagnostics.append(f"{option_type}: parse failed {exc}")
                return pd.DataFrame(), diagnostics
        return self._standardize_deals(df, option_type), diagnostics

    def fetch_bulk_deals(self, from_date: str, to_date: str) -> tuple[pd.DataFrame, list[str]]:
        return self.fetch_nse_deals("bulk_deals", from_date, to_date)

    def fetch_block_deals(self, from_date: str, to_date: str) -> tuple[pd.DataFrame, list[str]]:
        return self.fetch_nse_deals("block_deals", from_date, to_date)

    def fetch_shareholding(self, symbol: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
        diagnostics: list[str] = []
        sym = (symbol or self.config.symbol).upper()
        url = f"/api/corporate-share-holdings-master"
        ok, text, detail = self.nse_get(url, params={"index": "equities", "symbol": sym})
        diagnostics.append(detail)
        if not ok or not text:
            return pd.DataFrame(), diagnostics
        try:
            data = pd.read_json(io.StringIO(text))
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(f"parse failed {exc}")
            return pd.DataFrame(), diagnostics
        return data, diagnostics

    def filter_symbol(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        sym = (symbol or self.config.symbol).upper()
        df = df.copy()
        for col in df.columns:
            if str(col).upper() == "SYMBOL" or "SYMBOL" in str(col).upper():
                return df[df[col].astype(str).str.upper() == sym]
        if "security_name" in df.columns:
            mask = df["security_name"].astype(str).str.upper().str.contains(sym, na=False)
            return df[mask]
        return df

    def _standardize_deals(self, df: pd.DataFrame, deal_type: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        rename = {
            "Date": "date",
            "DATE": "date",
            "Symbol": "symbol",
            "SYMBOL": "symbol",
            "Security Name": "security_name",
            "Client Name": "client_name",
            "CLIENT NAME": "client_name",
            "Buy/Sell": "buy_sell",
            "BUY/SELL": "buy_sell",
            "Type": "buy_sell",
            "Quantity Traded": "quantity",
            "QUANTITY TRADED": "quantity",
            "Quantity": "quantity",
            "Trade Price / Wght. Avg. Price": "price",
            "Price": "price",
            "Remarks": "remarks",
        }
        out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
        for col in ["date", "symbol", "security_name", "client_name", "buy_sell", "quantity", "price", "remarks"]:
            if col not in out.columns:
                out[col] = ""
        out["deal_type"] = deal_type
        out["source"] = "NSE" if deal_type else "NSE"
        # coerce numeric
        out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out = out.dropna(subset=["symbol", "date", "quantity", "price"], how="any")
        return out[["date", "symbol", "security_name", "client_name", "buy_sell", "quantity", "price", "remarks", "deal_type", "source"]]


def iso_today() -> str:
    return datetime.utcnow().date().isoformat()

"""Core analytics for MOIL Macro Radar.

The module intentionally separates market analytics from Streamlit UI so the
same logic can be used in local Windows runs, GitHub Codespaces and future
scheduled jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:  # allows imports before requirements are installed
    yf = None


DEFAULT_MARKET_TICKERS: Dict[str, str] = {
    "MOIL": "MOIL.NS",
    "NIFTY Metal": "^CNXMETAL",
    "JSW Steel": "JSWSTEEL.NS",
    "SAIL": "SAIL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Hindustan Zinc": "HINDZINC.NS",
    "Brent Crude": "BZ=F",
    "USDINR": "INR=X",
    "Baltic Dry Proxy": "BDRY",
}

STEEL_PEERS = ["JSW Steel", "SAIL", "Tata Steel"]
CORE_TICKERS = ["MOIL", "NIFTY Metal", "JSW Steel", "SAIL", "Tata Steel", "Brent Crude", "USDINR"]

# Expert weights to emphasise high-signal indicators (used in manual/AI macro layer)
EXPERT_WEIGHTS: Dict[str, float] = {
    # Physical market
    "silico-manganese prices": 2.0,
    "silico manganese prices": 2.0,
    "india crude steel production": 1.6,
    "china steel exports": 1.5,
    "china power / industrial stress": 1.2,
    # MOIL equity technicals
    "moil vs 200dma": 1.6,
    "moil vs 50dma": 1.3,
    "moil 3m momentum": 1.2,
    # Macro stress proxies
    "brent crude": 1.1,
    "usd inr": 1.0,
}


@dataclass(frozen=True)
class RegimeSummary:
    raw_score: float
    max_score: float
    min_score: float
    normalized_score: float
    label: str
    risk_level: str
    alert_level: str
    confidence_score: float
    data_quality_score: float
    updated_at: str

    @property
    def score(self) -> float:
        """Backwards-compatible alias for older app versions."""
        return self.raw_score


def _now_ist() -> str:
    # IST is UTC + 5:30
    ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    return ist_time.strftime("%Y-%m-%d %H:%M:%S IST")


def localize_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Safely convert a DataFrame index from UTC/Exchange time to IST (naive)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        try:
            if out.index.tz is None:
                # Assume UTC if naive
                out.index = out.index.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
            else:
                out.index = out.index.tz_convert("Asia/Kolkata").tz_localize(None)
        except Exception:
            pass
    return out


def _clean_name(name: object) -> str:
    return str(name or "").strip()


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])


def normalize_yfinance_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame from a yfinance download result."""
    if frame is None or frame.empty:
        return _empty_ohlcv()

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        ticker_str = str(ticker)
        levels = [list(map(str, level)) for level in df.columns.levels]
        try:
            if ticker_str in levels[0]:
                df = df.xs(ticker, axis=1, level=0)
            elif ticker_str in levels[1]:
                df = df.xs(ticker, axis=1, level=1)
            else:
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={col: str(col).strip() for col in df.columns})
    standard_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[standard_cols].copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in standard_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Adj Close"].notna().any():
        df["Close"] = df["Adj Close"].combine_first(df["Close"])

    return df.dropna(how="all")


def fetch_market_data(
    tickers: Mapping[str, str],
    period: str = "2y",
    interval: str = "1d",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Download market data from yfinance.

    Individual feed failures are isolated so one broken ticker does not break
    the whole dashboard.
    """
    data: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    if yf is None:
        return {}, {name: "yfinance is not installed. Run pip install -r requirements.txt." for name in tickers}

    for name, ticker in tickers.items():
        clean_name = _clean_name(name)
        clean_ticker = _clean_name(ticker)
        if not clean_ticker:
            errors[clean_name] = "Ticker not set"
            continue

        last_err = ""
        for attempt in range(3):
            try:
                raw = yf.download(
                    clean_ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                df = normalize_yfinance_frame(raw, clean_ticker)
                if df.empty or df["Close"].dropna().empty:
                    last_err = f"No close-price data returned for {clean_ticker}."
                else:
                    data[clean_name] = df
                    last_err = ""
                    break
            except Exception as exc:  # pragma: no cover - depends on network/provider
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.2)

        if last_err:
            errors[clean_name] = last_err

    return data, errors


def generate_demo_market_data(names: Iterable[str], periods: int = 520, seed: int = 7) -> Dict[str, pd.DataFrame]:
    """Create deterministic synthetic data for demos/tests when live feeds fail."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    output: Dict[str, pd.DataFrame] = {}
    base_levels = {
        "MOIL": 420,
        "NIFTY Metal": 9000,
        "JSW Steel": 920,
        "SAIL": 135,
        "Tata Steel": 160,
        "Hindustan Zinc": 500,
        "Brent Crude": 82,
        "USDINR": 83,
        "Baltic Dry Proxy": 12,
    }
    common_cycle = rng.normal(0.00035, 0.011, size=len(dates)).cumsum()
    for i, name in enumerate(names):
        base = base_levels.get(str(name), 100 + 25 * i)
        beta = 1.0 if name in {"MOIL", "NIFTY Metal", "JSW Steel", "SAIL", "Tata Steel"} else 0.35
        idio = rng.normal(0.0001, 0.014 + 0.002 * (i % 3), size=len(dates)).cumsum()
        level = base * np.exp(beta * common_cycle + idio)
        close = pd.Series(level, index=dates).clip(lower=1)
        open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.004, 0.003, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.004, 0.003, len(dates))))
        volume = rng.integers(100_000, 2_500_000, len(dates))
        output[str(name)] = pd.DataFrame(
            {
                "Open": open_.values,
                "High": high.values,
                "Low": low.values,
                "Close": close.values,
                "Adj Close": close.values,
                "Volume": volume,
            },
            index=dates,
        )
    return output


def build_price_panel(data: Mapping[str, pd.DataFrame], field: str = "Close") -> pd.DataFrame:
    prices = pd.DataFrame()
    for name, df in data.items():
        if df is None or df.empty or field not in df.columns:
            continue
        prices[_clean_name(name)] = pd.to_numeric(df[field], errors="coerce")
    if prices.empty:
        return prices
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    return prices.dropna(how="all")


def build_volume_panel(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    volumes = pd.DataFrame()
    for name, df in data.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        volumes[_clean_name(name)] = pd.to_numeric(df["Volume"], errors="coerce")
    if volumes.empty:
        return volumes
    volumes.index = pd.to_datetime(volumes.index, errors="coerce")
    volumes = volumes[~volumes.index.isna()].sort_index()
    return volumes.dropna(how="all")


def normalize_to_100(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    normalized = prices.copy()
    for col in normalized.columns:
        clean = normalized[col].dropna()
        if clean.empty or clean.iloc[0] == 0:
            normalized[col] = np.nan
        else:
            normalized[col] = normalized[col] / clean.iloc[0] * 100
    return normalized


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    return prices.sort_index().pct_change(fill_method=None)


def _last_valid(series: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= periods:
        return None
    prev = float(clean.iloc[-periods - 1])
    latest = float(clean.iloc[-1])
    if prev == 0:
        return None
    return latest / prev - 1


def _distance_to_ma(series: pd.Series, window: int, min_periods: int) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < min_periods:
        return None
    ma = clean.rolling(window, min_periods=min_periods).mean().iloc[-1]
    latest = clean.iloc[-1]
    if pd.isna(ma) or ma == 0:
        return None
    return float(latest / ma - 1)


def metric_snapshot(prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    returns = compute_returns(prices)
    for name in prices.columns:
        series = prices[name].dropna()
        if series.empty:
            continue
        latest = _last_valid(series)
        day_change = _pct_change(series, 1)
        mtd = None
        try:
            current_month = series.index[-1].to_period("M")
            month_series = series[series.index.to_period("M") == current_month]
            if len(month_series) > 1 and month_series.iloc[0] != 0:
                mtd = float(series.iloc[-1] / month_series.iloc[0] - 1)
        except Exception:
            mtd = None
        ret_20d = _pct_change(series, 20)
        ret_63d = _pct_change(series, 63)
        dist_50 = _distance_to_ma(series, 50, 30)
        dist_200 = _distance_to_ma(series, 200, 100)
        vol_20 = returns[name].rolling(20, min_periods=10).std().iloc[-1] if name in returns else np.nan

        rel_volume = np.nan
        volume_z = np.nan
        if volumes is not None and name in volumes.columns:
            v = pd.to_numeric(volumes[name], errors="coerce").replace(0, np.nan).dropna()
            if len(v) >= 20:
                avg20 = v.rolling(20, min_periods=10).mean().iloc[-1]
                std60 = v.rolling(60, min_periods=20).std().iloc[-1] if len(v) >= 20 else np.nan
                mean60 = v.rolling(60, min_periods=20).mean().iloc[-1] if len(v) >= 20 else np.nan
                rel_volume = float(v.iloc[-1] / avg20) if avg20 and not pd.isna(avg20) else np.nan
                volume_z = float((v.iloc[-1] - mean60) / std60) if std60 and not pd.isna(std60) else np.nan

        rows.append(
            {
                "asset": name,
                "latest": latest,
                "1D %": day_change,
                "MTD %": mtd,
                "20D %": ret_20d,
                "63D %": ret_63d,
                "vs 50DMA %": dist_50,
                "vs 200DMA %": dist_200,
                "20D vol ann.": None if pd.isna(vol_20) else float(vol_20 * np.sqrt(252)),
                "relative_volume": rel_volume,
                "volume_z_score": volume_z,
                "last_date": series.index[-1].strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(prices: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    returns = compute_returns(prices).tail(lookback)
    returns = returns.dropna(axis=1, how="all")
    if returns.shape[1] < 2:
        return pd.DataFrame()
    return returns.corr(min_periods=max(10, min(30, lookback // 2)))


def compute_moil_correlation_ranking(prices: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    corr = correlation_matrix(prices, lookback)
    if corr.empty or "MOIL" not in corr.columns:
        return pd.DataFrame(columns=["asset", "correlation_to_moil"])
    ranking = corr.loc["MOIL"].drop(labels=["MOIL"], errors="ignore").dropna().sort_values(ascending=False)
    return ranking.reset_index().rename(columns={"index": "asset", "MOIL": "correlation_to_moil"})


def rolling_correlation(prices: pd.DataFrame, left: str, right: str, window: int = 60) -> pd.Series:
    if prices.empty or left not in prices.columns or right not in prices.columns:
        return pd.Series(dtype=float)
    returns = compute_returns(prices[[left, right]])
    return returns[left].rolling(window, min_periods=max(10, window // 2)).corr(returns[right]).dropna()


def compute_rolling_correlations(prices: pd.DataFrame, peers: Iterable[str], window: int = 60) -> pd.DataFrame:
    out = pd.DataFrame()
    for peer in peers:
        series = rolling_correlation(prices, "MOIL", peer, window=window)
        if not series.empty:
            out[peer] = series
    return out


# Backwards-compatible alias for legacy callers
def moil_correlation_table(corr: pd.DataFrame) -> pd.DataFrame:
    return compute_moil_correlation_ranking(corr) if isinstance(corr, pd.DataFrame) else pd.DataFrame()


def detect_anomalies(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    window: int = 60,
    threshold: float = 2.5,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    returns = compute_returns(prices)
    rows: List[Dict[str, object]] = []
    for asset in returns.columns:
        r = returns[asset].dropna()
        if len(r) < max(20, window // 2):
            continue
        mu = r.rolling(window, min_periods=max(10, window // 2)).mean().iloc[-1]
        sd = r.rolling(window, min_periods=max(10, window // 2)).std().iloc[-1]
        latest = r.iloc[-1]
        z = np.nan if sd == 0 or pd.isna(sd) else (latest - mu) / sd
        volume_z = np.nan
        if volumes is not None and asset in volumes.columns:
            v = volumes[asset].replace(0, np.nan).dropna()
            if len(v) >= max(20, window // 2):
                v_mu = v.rolling(window, min_periods=max(10, window // 2)).mean().iloc[-1]
                v_sd = v.rolling(window, min_periods=max(10, window // 2)).std().iloc[-1]
                volume_z = np.nan if v_sd == 0 or pd.isna(v_sd) else (v.iloc[-1] - v_mu) / v_sd
        flag = abs(z) >= threshold if not pd.isna(z) else False
        vol_flag = volume_z >= threshold if not pd.isna(volume_z) else False
        rows.append(
            {
                "asset": asset,
                "latest_return_%": latest,
                "z_score": z,
                "volume_z_score": volume_z,
                "direction": "up" if latest > 0 else "down",
                "flag": "ANOMALY" if flag or vol_flag else "normal",
            }
        )
    return pd.DataFrame(rows).sort_values(["flag", "z_score"], ascending=[False, False])


# Backwards-compatible alias for legacy callers
def detect_return_anomalies(prices: pd.DataFrame, window: int = 60, threshold: float = 2.5) -> pd.DataFrame:
    """Detect price-return anomalies (alias to detect_anomalies without volumes).

    Kept to avoid breaking older smoke tests and scripts that expect the prior
    name. Uses the same z-score-based logic as detect_anomalies.
    """

    return detect_anomalies(prices, volumes=None, window=window, threshold=threshold)


def normalize_manual_macro(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize physical macro rows while preserving Data Quality v2 metadata.

    Legacy columns are required by the UI. Extended columns allow confidence-
    adjusted scoring from scheduled/cache evidence packs.
    """
    base_columns = ["date", "indicator", "value", "unit", "status", "score", "commentary", "source"]
    quality_columns = [
        "best_source",
        "source_tier",
        "source_type",
        "exact_data",
        "latest_period",
        "data_confidence",
        "confidence_weight",
        "scoring_allowed",
        "confidence_label",
        "confidence_notes",
        "evidence_count",
        "raw_evidence",
    ]
    columns = base_columns + quality_columns
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    for col in base_columns:
        if col not in out.columns:
            out[col] = "" if col != "score" else 0.0
    for col in quality_columns:
        if col not in out.columns:
            if col in {"data_confidence"}:
                out[col] = np.nan
            elif col in {"confidence_weight"}:
                out[col] = np.nan
            elif col in {"source_tier", "evidence_count"}:
                out[col] = np.nan
            elif col in {"scoring_allowed", "exact_data"}:
                out[col] = False
            else:
                out[col] = ""
    out = out[columns]
    out["date"] = out["date"].astype(str).str.strip()
    out["indicator"] = out["indicator"].astype(str).str.strip()
    out["status"] = out["status"].astype(str).str.strip().replace("", "neutral")
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0).clip(-2, 2)
    out["data_confidence"] = pd.to_numeric(out["data_confidence"], errors="coerce")
    out["confidence_weight"] = pd.to_numeric(out["confidence_weight"], errors="coerce")
    out["source_tier"] = pd.to_numeric(out["source_tier"], errors="coerce")
    out["evidence_count"] = pd.to_numeric(out["evidence_count"], errors="coerce")
    return out


def read_manual_macro_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return normalize_manual_macro(pd.read_csv(path))
        except Exception:
            pass
    return normalize_manual_macro(
        pd.DataFrame(
            [
                {
                    "date": "",
                    "indicator": "Silico-Manganese Prices",
                    "value": "",
                    "unit": "INR/t",
                    "status": "neutral",
                    "score": 0.0,
                    "commentary": "Enter latest SiMn price direction.",
                    "source": "Manual",
                },
                {
                    "date": "",
                    "indicator": "India Crude Steel Production",
                    "value": "",
                    "unit": "Mt / YoY %",
                    "status": "neutral",
                    "score": 0.0,
                    "commentary": "Positive when output and consumption are rising.",
                    "source": "Manual",
                },
                {
                    "date": "",
                    "indicator": "China Steel Exports",
                    "value": "",
                    "unit": "Mt / YoY %",
                    "status": "neutral",
                    "score": 0.0,
                    "commentary": "Positive when export pressure is easing.",
                    "source": "Manual",
                },
                {
                    "date": "",
                    "indicator": "China Power / Industrial Stress",
                    "value": "",
                    "unit": "qualitative",
                    "status": "neutral",
                    "score": 0.0,
                    "commentary": "Watch Chinese power stress and industrial curtailment.",
                    "source": "Manual",
                },
            ]
        )
    )


def read_news_tracker_csv(path: Path) -> pd.DataFrame:
    columns = ["date", "category", "item", "value", "impact", "confidence", "sentiment", "details", "source"]
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            if col in {"impact", "confidence"}:
                df[col] = 0.0
            elif col == "sentiment":
                df[col] = "Neutral"
            else:
                df[col] = ""
    df = df[columns]
    df["impact"] = pd.to_numeric(df["impact"], errors="coerce").fillna(0.0).clip(-2, 2)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5).clip(0, 1)
    df["sentiment"] = df["sentiment"].fillna("Neutral").astype(str).str.title()
    return df


def _component_score_from_return(ret: Optional[float], strong: float = 0.08, mild: float = 0.03) -> Tuple[float, str, float]:
    if ret is None:
        return 0.0, "Insufficient history", 0.25
    if ret >= strong:
        return 1.0, f"strong positive return {ret:.1%}", 0.90
    if ret >= mild:
        return 0.5, f"positive return {ret:.1%}", 0.75
    if ret <= -strong:
        return -1.0, f"strong negative return {ret:.1%}", 0.90
    if ret <= -mild:
        return -0.5, f"negative return {ret:.1%}", 0.75
    return 0.0, f"flat/mixed return {ret:.1%}", 0.60


def _component_score_from_distance(dist: Optional[float], strong: float = 0.08, mild: float = 0.0) -> Tuple[float, str, float]:
    if dist is None:
        return 0.0, "Insufficient moving-average history", 0.25
    if dist >= strong:
        return 1.0, f"well above trend by {dist:.1%}", 0.90
    if dist > mild:
        return 0.5, f"above trend by {dist:.1%}", 0.80
    if dist <= -strong:
        return -1.0, f"well below trend by {dist:.1%}", 0.90
    if dist < mild:
        return -0.5, f"below trend by {dist:.1%}", 0.80
    return 0.0, f"near trend by {dist:.1%}", 0.60


def _confidence_weight_from_row(row: pd.Series) -> float:
    """Confidence gates for physical macro rows.

    >=80 full impact, 60-79 70%, 40-59 40%, below 40 commentary-only.
    """
    try:
        if pd.notna(row.get("confidence_weight", np.nan)):
            return float(np.clip(row.get("confidence_weight"), 0, 1))
    except Exception:
        pass
    try:
        c = float(row.get("data_confidence", np.nan))
    except Exception:
        c = np.nan
    if pd.isna(c):
        source = str(row.get("source", "")).lower()
        if any(x in source for x in ["argus", "jpc", "gacc", "nbs", "worldsteel", "moil pdf", "official"]):
            c = 80
        elif any(x in source for x in ["bigmint", "steelmint", "ofbusiness", "mysteel"]):
            c = 55
        else:
            c = 60 if str(row.get("source", "")).strip() and "manual" not in source else 50
    if c >= 80:
        return 1.0
    if c >= 60:
        return 0.70
    if c >= 40:
        return 0.40
    return 0.0


def _manual_source_scores(manual_macro: pd.DataFrame, groq_scores: Optional[dict]) -> pd.DataFrame:
    """Resolve manual/cache/Groq scores with confidence-adjusted impact."""
    manual = normalize_manual_macro(manual_macro)

    def apply_manual_defaults(df: pd.DataFrame) -> pd.DataFrame:
        raw_scores = []
        effective_scores = []
        confidences = []
        sources = []
        rationales = []
        for _, row in df.iterrows():
            raw = float(np.clip(pd.to_numeric(row.get("score", 0), errors="coerce"), -2, 2))
            weight = _confidence_weight_from_row(row)
            data_conf = row.get("data_confidence", np.nan)
            try:
                data_conf = float(data_conf)
            except Exception:
                data_conf = weight * 100
            if pd.isna(data_conf):
                data_conf = weight * 100
            raw_scores.append(raw)
            effective_scores.append(raw * weight)
            confidences.append(float(np.clip(data_conf / 100.0, 0, 1)))
            sources.append(str(row.get("source", "Manual")))
            rationale = str(row.get("commentary", ""))
            rationale += f" | Data confidence {data_conf:.0f}/100; confidence weight {weight:.2f}."
            if weight == 0:
                rationale += " Commentary-only: below scoring threshold."
            rationales.append(rationale)
        df["applied_raw_score"] = raw_scores
        df["applied_score"] = effective_scores
        df["applied_effective_score"] = effective_scores
        df["applied_confidence"] = confidences
        df["applied_source"] = sources
        df["applied_rationale"] = rationales
        return df

    if not groq_scores or not isinstance(groq_scores, dict) or not groq_scores.get("macro_scores"):
        return apply_manual_defaults(manual)

    ai_rows = pd.DataFrame(groq_scores.get("macro_scores", []))
    if ai_rows.empty or "indicator" not in ai_rows.columns:
        return apply_manual_defaults(manual)

    ai_rows["indicator_key"] = ai_rows["indicator"].astype(str).str.strip().str.lower()
    ai_map = ai_rows.set_index("indicator_key").to_dict("index")
    applied_raw = []
    applied_effective = []
    applied_conf = []
    applied_source = []
    applied_rationale = []
    for _, row in manual.iterrows():
        key = str(row["indicator"]).strip().lower()
        ai = ai_map.get(key)
        base_weight = _confidence_weight_from_row(row)
        if ai:
            raw = float(np.clip(pd.to_numeric(ai.get("signal_score", ai.get("score", 0)), errors="coerce"), -2, 2))
            weight_adj = EXPERT_WEIGHTS.get(key, 1.0)
            data_conf = ai.get("data_confidence", row.get("data_confidence", np.nan))
            try:
                data_conf = float(data_conf)
            except Exception:
                data_conf = float(ai.get("confidence", 0.65)) * 100
            if pd.isna(data_conf):
                data_conf = float(ai.get("confidence", 0.65)) * 100
            if data_conf >= 80:
                gate = 1.0
            elif data_conf >= 60:
                gate = 0.70
            elif data_conf >= 40:
                gate = 0.40
            else:
                gate = 0.0
            effective = ai.get("effective_score", raw * gate)
            try:
                effective = float(effective)
            except Exception:
                effective = raw * gate
            applied_raw.append(raw * weight_adj)
            applied_effective.append(float(np.clip(effective * weight_adj, -2 * weight_adj, 2 * weight_adj)))
            applied_conf.append(float(np.clip(data_conf / 100.0, 0, 1)))
            applied_source.append("Groq/Llama 3.3 evidence scoring")
            rationale = str(ai.get("rationale", row.get("commentary", "")))
            rationale += f" | Raw signal {raw:+.2f}; effective signal {float(np.clip(effective * weight_adj, -2 * weight_adj, 2 * weight_adj)):+.2f}; data confidence {data_conf:.0f}/100; expert weight {weight_adj:.2f}."
            if gate == 0:
                rationale += " Commentary-only: below scoring threshold."
            applied_rationale.append(rationale)
        else:
            raw = float(row.get("score", 0))
            weight_adj = EXPERT_WEIGHTS.get(key, 1.0)
            effective = raw * base_weight * weight_adj
            data_conf = row.get("data_confidence", base_weight * 100)
            try:
                data_conf = float(data_conf)
            except Exception:
                data_conf = base_weight * 100
            applied_raw.append(raw * weight_adj)
            applied_effective.append(effective)
            applied_conf.append(float(np.clip(data_conf / 100.0, 0, 1)))
            applied_source.append(str(row.get("source", "Manual")))
            applied_rationale.append(str(row.get("commentary", "")) + f" | No Groq row; confidence weight {base_weight:.2f}; expert weight {weight_adj:.2f}.")
    manual["applied_raw_score"] = applied_raw
    manual["applied_score"] = applied_effective
    manual["applied_effective_score"] = applied_effective
    manual["applied_confidence"] = applied_conf
    manual["applied_source"] = applied_source
    manual["applied_rationale"] = applied_rationale
    return manual


def compute_data_quality(
    data: Mapping[str, pd.DataFrame],
    manual_macro: pd.DataFrame,
    tickers: Optional[Mapping[str, str]] = None,
) -> Tuple[float, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    expected = list((tickers or DEFAULT_MARKET_TICKERS).keys())
    today = pd.Timestamp.today().normalize()
    for asset in expected:
        df = data.get(asset, pd.DataFrame())
        if df is None or df.empty or "Close" not in df.columns or df["Close"].dropna().empty:
            rows.append({"source": asset, "type": "market", "score": 0.0, "status": "missing", "details": "No close price"})
            continue
        last_date = pd.to_datetime(df.index.max()).normalize()
        age_days = int(max((today - last_date).days, 0))
        row_count = len(df)
        freshness = 1.0 if age_days <= 4 else 0.75 if age_days <= 10 else 0.35
        coverage = 1.0 if row_count >= 180 else 0.75 if row_count >= 60 else 0.40
        score = 100 * min(freshness, coverage)
        rows.append(
            {
                "source": asset,
                "type": "market",
                "score": score,
                "status": "ok" if score >= 70 else "watch",
                "details": f"rows={row_count}; last={last_date.date()}; age={age_days}d",
            }
        )

    manual = normalize_manual_macro(manual_macro)
    for _, row in manual.iterrows():
        details = []
        data_conf = pd.to_numeric(row.get("data_confidence", np.nan), errors="coerce")
        if pd.notna(data_conf):
            score = float(np.clip(data_conf, 0, 100))
            details.append(f"data_confidence={score:.0f}/100")
            if str(row.get("confidence_notes", "")).strip():
                details.append(str(row.get("confidence_notes", ""))[:180])
        else:
            score = 70.0
            date_text = str(row.get("date", "")).strip()
            if date_text:
                parsed = pd.to_datetime(date_text, errors="coerce")
                if pd.notna(parsed):
                    age_days = int(max((today - parsed.normalize()).days, 0))
                    if age_days <= 7:
                        score = 90.0
                    elif age_days <= 30:
                        score = 70.0
                    else:
                        score = 45.0
                    details.append(f"age={age_days}d")
                else:
                    score = 55.0
                    details.append("date parse warning")
            else:
                score = 50.0
                details.append("missing date")
            if not str(row.get("value", "")).strip():
                score = min(score, 45.0)
                details.append("missing value")
        if not bool(row.get("scoring_allowed", True)) and pd.notna(data_conf):
            details.append("commentary-only: below scoring threshold")
        rows.append(
            {
                "source": str(row.get("indicator", "Manual indicator")),
                "type": "physical macro evidence",
                "score": score,
                "status": "ok" if score >= 70 else "watch" if score >= 40 else "weak",
                "details": "; ".join(details),
            }
        )
    quality_df = pd.DataFrame(rows)
    if quality_df.empty:
        return 0.0, quality_df
    return float(quality_df["score"].mean()), quality_df


def compute_regime_score(
    prices: pd.DataFrame,
    manual_macro: pd.DataFrame,
    groq_scores: Optional[dict] = None,
    volumes: Optional[pd.DataFrame] = None,
    data_quality_score: Optional[float] = None,
) -> Tuple[pd.DataFrame, RegimeSummary]:
    rows: List[Dict[str, object]] = []

    def add(pillar: str, signal: str, normalized_signal: float, weight: float, rationale: str, source: str, confidence: float = 0.70) -> None:
        normalized_signal = float(np.clip(normalized_signal, -1, 1))
        points = normalized_signal * weight
        rows.append(
            {
                "pillar": pillar,
                "signal": signal,
                "signal_score": round(normalized_signal, 3),
                "weight": weight,
                "points": round(points, 3),
                "max_points": weight,
                "rationale": rationale,
                "source": source,
                "confidence": round(float(np.clip(confidence, 0, 1)), 2),
            }
        )

    # Equity/market-cycle layer
    if "MOIL" in prices.columns:
        moil = prices["MOIL"]
        score, why, conf = _component_score_from_return(_pct_change(moil, 20), strong=0.10, mild=0.03)
        add("Market", "MOIL 20D momentum", score, 8, why, "yfinance", conf)
        score, why, conf = _component_score_from_distance(_distance_to_ma(moil, 50, 30), strong=0.08, mild=0.0)
        add("Market", "MOIL vs 50DMA", score, 8, why, "yfinance", conf)
        score, why, conf = _component_score_from_distance(_distance_to_ma(moil, 200, 100), strong=0.12, mild=0.0)
        add("Market", "MOIL vs 200DMA", score, 8, why, "yfinance", conf)
    else:
        add("Market", "MOIL data availability", -0.5, 8, "MOIL price feed missing", "data quality", 0.90)

    if "NIFTY Metal" in prices.columns:
        score, why, conf = _component_score_from_return(_pct_change(prices["NIFTY Metal"], 20), strong=0.08, mild=0.025)
        add("Market", "NIFTY Metal breadth", score, 7, why, "yfinance", conf)

    steel_scores = []
    steel_why = []
    for peer in STEEL_PEERS:
        if peer in prices.columns:
            score, why, _ = _component_score_from_distance(_distance_to_ma(prices[peer], 50, 30), strong=0.08, mild=0.0)
            steel_scores.append(score)
            steel_why.append(f"{peer}: {why}")
    if steel_scores:
        add("Market", "Indian steel basket trend", float(np.mean(steel_scores)), 8, "; ".join(steel_why), "yfinance", 0.78)

    if "Hindustan Zinc" in prices.columns:
        score, why, conf = _component_score_from_return(_pct_change(prices["Hindustan Zinc"], 20), strong=0.08, mild=0.025)
        add("Market", "Metal peer confirmation", score, 4, why, "yfinance", conf)

    # Volume confirmation
    if volumes is not None and "MOIL" in volumes.columns and "MOIL" in prices.columns:
        v = volumes["MOIL"].replace(0, np.nan).dropna()
        ret20 = _pct_change(prices["MOIL"], 20)
        if len(v) >= 20 and ret20 is not None:
            avg20 = v.rolling(20, min_periods=10).mean().iloc[-1]
            rel_vol = float(v.iloc[-1] / avg20) if avg20 and not pd.isna(avg20) else np.nan
            z = np.nan
            if len(v) >= 60:
                mu = v.rolling(60, min_periods=20).mean().iloc[-1]
                sd = v.rolling(60, min_periods=20).std().iloc[-1]
                z = (v.iloc[-1] - mu) / sd if sd and not pd.isna(sd) else np.nan
            vol_signal = 0.0
            why = f"relative volume {rel_vol:.2f}x; 20D return {ret20:.1%}"
            if ret20 > 0.03 and rel_vol >= 1.20:
                vol_signal = 0.75
            elif ret20 < -0.03 and rel_vol >= 1.20:
                vol_signal = -0.75
            elif ret20 > 0.03 and rel_vol < 0.80:
                vol_signal = -0.25
                why += "; rally lacks volume confirmation"
            add("Volume", "MOIL volume confirmation", vol_signal, 7, why, "yfinance volume", 0.70)

    # Macro stress layer
    if "Brent Crude" in prices.columns:
        ret = _pct_change(prices["Brent Crude"], 20)
        if ret is None:
            add("Macro stress", "Brent energy stress", 0.0, 6, "Insufficient Brent history", "yfinance", 0.25)
        else:
            # Rising Brent is an input-cost and inflation stress for industrials.
            signal = -1.0 if ret > 0.10 else -0.5 if ret > 0.04 else 0.5 if ret < -0.04 else 0.0
            add("Macro stress", "Brent energy stress", signal, 6, f"Brent 20D move {ret:.1%}", "yfinance", 0.75)

    if "USDINR" in prices.columns:
        ret = _pct_change(prices["USDINR"], 20)
        if ret is None:
            add("Macro stress", "USDINR FX stress", 0.0, 5, "Insufficient USDINR history", "yfinance", 0.25)
        else:
            signal = -1.0 if ret > 0.025 else -0.5 if ret > 0.010 else 0.5 if ret < -0.010 else 0.0
            add("Macro stress", "USDINR FX stress", signal, 5, f"USDINR 20D move {ret:.1%}", "yfinance", 0.70)

    if "Baltic Dry Proxy" in prices.columns:
        score, why, conf = _component_score_from_return(_pct_change(prices["Baltic Dry Proxy"], 20), strong=0.10, mild=0.03)
        add("Freight", "Dry-bulk freight proxy", score, 4, why, "yfinance", conf)

    # Manual / AI physical macro layer
    applied_macro = _manual_source_scores(manual_macro, groq_scores)
    for _, row in applied_macro.iterrows():
        indicator = str(row.get("indicator", "Manual macro"))
        applied = float(row.get("applied_score", row.get("score", 0)))
        confidence = float(row.get("applied_confidence", 0.65))
        rationale = str(row.get("applied_rationale", row.get("commentary", "")))
        source = str(row.get("applied_source", row.get("source", "Manual")))
        weight = 7.0
        if "Silico" in indicator or "Manganese" in indicator:
            weight = 10.0
        elif "Crude Steel" in indicator or "Steel Production" in indicator:
            weight = 8.0
        elif "China Steel Exports" in indicator:
            weight = 8.0
        elif "Power" in indicator or "Industrial Stress" in indicator:
            weight = 5.0
        add("Physical macro", indicator, applied / 2.0, weight, rationale, source, confidence)

    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        summary = RegimeSummary(0, 1, -1, 50, "Neutral / mixed cycle", "Unknown", "Watch", 0, 0, _now_ist())
        return scorecard, summary

    raw_score = float(scorecard["points"].sum())
    max_score = float(scorecard["max_points"].sum())
    min_score = -max_score
    normalized = (raw_score - min_score) / (max_score - min_score) * 100 if max_score else 50.0
    normalized = float(np.clip(normalized, 0, 100))

    if normalized >= 70:
        label = "Bullish industrial reflation"
        risk_level = "Low-to-moderate"
    elif normalized >= 55:
        label = "Constructive / watch confirmation"
        risk_level = "Moderate"
    elif normalized >= 45:
        label = "Neutral / mixed cycle"
        risk_level = "Balanced"
    else:
        label = "Risk-off commodity regime"
        risk_level = "Elevated"

    data_q = 70.0 if data_quality_score is None else float(np.clip(data_quality_score, 0, 100))
    avg_conf = float((scorecard["confidence"] * scorecard["max_points"]).sum() / scorecard["max_points"].sum() * 100)
    signal_strength = min(abs(normalized - 50) * 2, 100)
    confidence_score = float(np.clip(0.50 * avg_conf + 0.30 * data_q + 0.20 * signal_strength, 0, 100))

    alert_level = "Watch"
    if normalized >= 70 and confidence_score >= 65:
        alert_level = "Regime Shift"
    elif normalized >= 55 and confidence_score >= 60:
        alert_level = "Confirmation"
    elif normalized <= 44 and confidence_score >= 60:
        alert_level = "Risk Warning"

    summary = RegimeSummary(
        raw_score=round(raw_score, 3),
        max_score=round(max_score, 3),
        min_score=round(min_score, 3),
        normalized_score=round(normalized, 1),
        label=label,
        risk_level=risk_level,
        alert_level=alert_level,
        confidence_score=round(confidence_score, 1),
        data_quality_score=round(data_q, 1),
        updated_at=_now_ist(),
    )
    return scorecard.sort_values("points", ascending=False), summary


def generate_rule_based_commentary(
    summary: RegimeSummary,
    scorecard: pd.DataFrame,
    moil_corr: Optional[pd.DataFrame] = None,
    anomalies: Optional[pd.DataFrame] = None,
) -> str:
    if scorecard is None or scorecard.empty:
        return "No scorecard available. Load market data and manual macro inputs."
    positives = scorecard[scorecard["points"] > 0].sort_values("points", ascending=False).head(3)
    negatives = scorecard[scorecard["points"] < 0].sort_values("points").head(3)
    pos_text = "; ".join(f"{r.signal} ({r.rationale})" for r in positives.itertuples(index=False)) or "no strong positive signal"
    neg_text = "; ".join(f"{r.signal} ({r.rationale})" for r in negatives.itertuples(index=False)) or "no major pressure signal"
    corr_text = ""
    if moil_corr is not None and not moil_corr.empty:
        top = moil_corr.head(3)
        corr_text = " Leading correlation map: " + "; ".join(
            f"{row.asset} {float(row.correlation_to_moil):+.2f}" for row in top.itertuples(index=False)
        ) + "."
    anomaly_text = ""
    if anomalies is not None and not anomalies.empty and "flag" in anomalies.columns:
        flagged = anomalies[anomalies["flag"] == "ANOMALY"].head(3)
        if not flagged.empty:
            anomaly_text = " Active anomaly watch: " + "; ".join(
                f"{row.asset} {row.direction} z={float(row.z_score):+.2f}" for row in flagged.itertuples(index=False)
            ) + "."
    return (
        f"**Regime:** {summary.label} ({summary.normalized_score:.1f}/100). "
        f"Confidence {summary.confidence_score:.1f}/100; data quality {summary.data_quality_score:.1f}/100.\n\n"
        f"**Supportive drivers:** {pos_text}.\n\n"
        f"**Pressure drivers:** {neg_text}.\n\n"
        f"**Desk read:** MOIL Macro Radar is in **{summary.alert_level}** mode. "
        f"Use this as a regime lens, not as an automatic trade instruction.{corr_text}{anomaly_text}"
    )


def compute_technical_levels(data: Mapping[str, pd.DataFrame], asset: str = "MOIL") -> Dict[str, object]:
    df = data.get(asset, pd.DataFrame())
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    close = df["Close"].dropna()
    if close.empty:
        return {}
    levels: Dict[str, object] = {"asset": asset, "last_close": float(close.iloc[-1]), "last_date": close.index[-1].strftime("%Y-%m-%d")}
    if len(close) >= 20:
        levels["support_20d"] = float(close.tail(20).min())
        levels["resistance_20d"] = float(close.tail(20).max())
    if len(close) >= 50:
        levels["dma50"] = float(close.rolling(50).mean().iloc[-1])
    if len(close) >= 200:
        levels["dma200"] = float(close.rolling(200).mean().iloc[-1])
    if "Volume" in df.columns:
        v = df["Volume"].replace(0, np.nan).dropna()
        if len(v) >= 20:
            levels["relative_volume"] = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
    return levels


def build_telegram_alert_text(
    summary: RegimeSummary,
    scorecard: pd.DataFrame,
    anomalies: Optional[pd.DataFrame],
    prices: pd.DataFrame,
    manual_macro: pd.DataFrame,
    news_tracker: Optional[pd.DataFrame] = None,
    technical_levels: Optional[Dict[str, object]] = None,
    ai_commentary: Optional[str] = None,
) -> str:
    positives = scorecard[scorecard["points"] > 0].sort_values("points", ascending=False).head(2) if scorecard is not None and not scorecard.empty else pd.DataFrame()
    negatives = scorecard[scorecard["points"] < 0].sort_values("points").head(2) if scorecard is not None and not scorecard.empty else pd.DataFrame()
    lines = [
        f"🚨 MOIL Macro Radar — {summary.alert_level}",
        f"Regime: {summary.label} | {summary.normalized_score:.1f}/100",
        f"Confidence: {summary.confidence_score:.1f}/100 | Data quality: {summary.data_quality_score:.1f}/100",
    ]
    if technical_levels:
        close = technical_levels.get("last_close")
        s20 = technical_levels.get("support_20d")
        r20 = technical_levels.get("resistance_20d")
        dma50 = technical_levels.get("dma50")
        dma200 = technical_levels.get("dma200")
        tech_parts = []
        if close is not None:
            tech_parts.append(f"Close {float(close):.2f}")
        if s20 is not None and r20 is not None:
            tech_parts.append(f"20D S/R {float(s20):.2f}/{float(r20):.2f}")
        if dma50 is not None:
            tech_parts.append(f"50DMA {float(dma50):.2f}")
        if dma200 is not None:
            tech_parts.append(f"200DMA {float(dma200):.2f}")
        if tech_parts:
            lines.append("Tech: " + " | ".join(tech_parts))
    if not positives.empty:
        lines.append("Positives: " + "; ".join(f"{r.signal} {r.points:+.1f}" for r in positives.itertuples(index=False)))
    if not negatives.empty:
        lines.append("Pressures: " + "; ".join(f"{r.signal} {r.points:+.1f}" for r in negatives.itertuples(index=False)))
    manual = normalize_manual_macro(manual_macro)
    if not manual.empty:
        macro_line = "; ".join(f"{r.indicator}: {float(r.score):+.1f}" for r in manual.itertuples(index=False))
        lines.append("Manual macro: " + macro_line)
    if anomalies is not None and not anomalies.empty and "flag" in anomalies.columns:
        flagged = anomalies[anomalies["flag"] == "ANOMALY"].head(2)
        if not flagged.empty:
            lines.append("Anomalies: " + "; ".join(f"{r.asset} {r.direction} z={float(r.z_score):+.2f}" for r in flagged.itertuples(index=False)))
    if ai_commentary:
        lines.append("AI desk: " + ai_commentary.strip()[:800])
    lines.append(f"Updated: {summary.updated_at}")
    lines.append("No buy/sell advice. Regime intelligence only.")
    return "\n".join(lines)


def build_groq_prompt_for_copy(
    summary: RegimeSummary,
    scorecard: pd.DataFrame,
    manual_macro: pd.DataFrame,
    snapshot: pd.DataFrame,
    moil_corr: pd.DataFrame,
    anomalies: pd.DataFrame,
    max_words: int = 150,
) -> str:
    payload = {
        "regime": summary.__dict__,
        "manual_macro": normalize_manual_macro(manual_macro).to_dict(orient="records"),
        "market_snapshot": snapshot.to_dict(orient="records") if snapshot is not None and not snapshot.empty else [],
        "top_scorecard": scorecard.head(10).to_dict(orient="records") if scorecard is not None and not scorecard.empty else [],
        "moil_correlation": moil_corr.head(6).to_dict(orient="records") if moil_corr is not None and not moil_corr.empty else [],
        "anomalies": anomalies.head(6).to_dict(orient="records") if anomalies is not None and not anomalies.empty else [],
    }
    return (
        "You are an institutional commodity analyst writing a concise Telegram alert for MOIL Macro Radar.\n"
        "Use only the JSON context below. Do not give buy/sell advice. Do not hallucinate missing values.\n"
        f"Keep the answer under {max_words} words. Focus on regime, positives, pressures and watch items.\n\n"
        f"Context:\n{payload}"
    )


def run_market_validation(prices: pd.DataFrame, forward_days: Iterable[int] = (5, 10, 20)) -> pd.DataFrame:
    """Lightweight validation: tests a transparent market-only setup condition.

    This is intentionally simple and not a trading system: it checks whether a
    constructive technical/sector regime historically preceded better MOIL
    forward returns in the loaded data sample.
    """
    if prices.empty or "MOIL" not in prices.columns:
        return pd.DataFrame()
    moil = prices["MOIL"].dropna()
    if len(moil) < 80:
        return pd.DataFrame()
    common = pd.DataFrame({"MOIL": moil})
    if "NIFTY Metal" in prices.columns:
        common["NIFTY Metal"] = prices["NIFTY Metal"]
    common = common.dropna()
    signal = (common["MOIL"] > common["MOIL"].rolling(50, min_periods=30).mean()) & (common["MOIL"].pct_change(20) > 0)
    if "NIFTY Metal" in common.columns:
        signal = signal & (common["NIFTY Metal"].pct_change(20) > 0)
    rows = []
    for fwd in forward_days:
        fwd_ret = common["MOIL"].shift(-fwd) / common["MOIL"] - 1
        sample = pd.DataFrame({"signal": signal, "fwd_ret": fwd_ret}).dropna()
        if sample.empty or sample["signal"].sum() < 5:
            continue
        on = sample[sample["signal"]]
        off = sample[~sample["signal"]]
        rows.append(
            {
                "forward_days": fwd,
                "signal_observations": int(len(on)),
                "non_signal_observations": int(len(off)),
                "avg_return_when_signal": float(on["fwd_ret"].mean()),
                "hit_rate_when_signal": float((on["fwd_ret"] > 0).mean()),
                "avg_return_without_signal": float(off["fwd_ret"].mean()) if len(off) else np.nan,
                "excess_avg_return": float(on["fwd_ret"].mean() - off["fwd_ret"].mean()) if len(off) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fetch_nse_deals(symbol: str = "MOIL") -> pd.DataFrame:
    """Fetch NSE bulk/block deals for a symbol with archive fallback.

    Uses the official nsearchives CSV when available. Returns an empty
    DataFrame when the archive loads but no symbol-level deal exists today,
    which is different from an endpoint failure. Diagnostics are exposed by
    auto_feeds.fetch_nse_deals_auto in the UI.
    """
    try:
        from auto_feeds import fetch_nse_deals_auto

        deals, _diagnostics = fetch_nse_deals_auto(symbol)
        return deals
    except Exception:
        return pd.DataFrame()

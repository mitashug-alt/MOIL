"""Core analytics for the MOIL Macro Radar dashboard.

The module is intentionally UI-light so it can be tested independently from Streamlit.
It combines live Yahoo Finance market proxies with manually maintained industrial
series such as silico-manganese prices, India steel production and China steel exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ModuleNotFoundError:  # Allows offline tests before requirements are installed.
    yf = None

from groq_macro_scorer import generate_groq_commentary, score_macro_with_groq


DEFAULT_MARKET_TICKERS: Dict[str, str] = {
    "MOIL": "MOIL.NS",
    "NIFTY Metal": "^CNXMETAL",
    "JSW Steel": "JSWSTEEL.NS",
    "SAIL": "SAIL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Hindustan Zinc": "HINDZINC.NS",
    "Brent Crude": "BZ=F",
    "USDINR": "INR=X",
    # Yahoo coverage for physical freight indices can vary by geography/version.
    # Keep this configurable in the dashboard; BDRY is retained as the default proxy.
    "Baltic Dry Proxy": "BDRY",
}

STEEL_PEERS = ["JSW Steel", "SAIL", "Tata Steel"]
MANUAL_INDICATOR_TEMPLATE = pd.DataFrame(
    [
        {
            "date": "",
            "indicator": "Silico-Manganese Prices",
            "value": "",
            "unit": "INR/t or USD/t",
            "status": "neutral",
            "score": 0,
            "commentary": "Enter latest weekly price direction/spread signal.",
            "source": "Manual / SteelMint / broker note",
        },
        {
            "date": "",
            "indicator": "India Crude Steel Production",
            "value": "",
            "unit": "Mt or YoY %",
            "status": "neutral",
            "score": 0,
            "commentary": "Positive when output and capacity utilization are rising.",
            "source": "Manual / industry association",
        },
        {
            "date": "",
            "indicator": "China Steel Exports",
            "value": "",
            "unit": "Mt or YoY %",
            "status": "neutral",
            "score": 0,
            "commentary": "Positive for MOIL when export pressure/dumping risk is easing.",
            "source": "Manual / customs / Reuters placeholder",
        },
        {
            "date": "",
            "indicator": "China Power / Industrial Stress",
            "value": "",
            "unit": "qualitative",
            "status": "neutral",
            "score": 0,
            "commentary": "Positive when stress constrains Chinese supply or exports.",
            "source": "Manual / news placeholder",
        },
    ]
)


@dataclass(frozen=True)
class RegimeSummary:
    score: float
    max_score: float
    min_score: float
    normalized_score: float
    label: str
    risk_level: str
    updated_at: str


def _clean_name(name: str) -> str:
    return str(name).strip()


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])


def normalize_yfinance_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame from a yfinance download result."""
    if frame is None or frame.empty:
        return _empty_ohlcv()

    df = frame.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return either price-field first or ticker first. Handle both.
        levels = [list(map(str, level)) for level in df.columns.levels]
        ticker_str = str(ticker)
        try:
            if ticker_str in levels[0]:
                df = df.xs(ticker, axis=1, level=0)
            elif ticker_str in levels[1]:
                df = df.xs(ticker, axis=1, level=1)
            else:
                # Single-ticker downloads sometimes have a redundant second level.
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={str(col): str(col).strip() for col in df.columns})
    standard_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[standard_cols]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
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

    Returns a tuple of {asset_name: dataframe} and {asset_name: error_message}.
    Individual failures are isolated so one broken ticker does not break the dashboard.
    """
    data: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    if yf is None:
        return {}, {name: "yfinance is not installed. Run pip install -r requirements.txt." for name in tickers}

    for name, ticker in tickers.items():
        clean_name = _clean_name(name)
        clean_ticker = _clean_name(ticker)
        if not clean_name or not clean_ticker:
            continue
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
                errors[clean_name] = f"No close-price data returned for {clean_ticker}."
            else:
                data[clean_name] = df
        except Exception as exc:  # pragma: no cover - depends on network/provider
            errors[clean_name] = f"{type(exc).__name__}: {exc}"

    return data, errors


def generate_demo_market_data(
    names: Iterable[str],
    periods: int = 520,
    seed: int = 7,
) -> Dict[str, pd.DataFrame]:
    """Create deterministic synthetic data for demos/tests when live feeds are unavailable."""
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
        "Baltic Dry Proxy": 1800,
    }
    common_cycle = rng.normal(0.00035, 0.011, size=len(dates)).cumsum()

    for i, name in enumerate(names):
        base = base_levels.get(name, 100 + 25 * i)
        beta = 1.0 if name in {"MOIL", "NIFTY Metal", "JSW Steel", "SAIL", "Tata Steel"} else 0.35
        idio = rng.normal(0.0001, 0.014 + 0.002 * (i % 3), size=len(dates)).cumsum()
        level = base * np.exp(beta * common_cycle + idio)
        close = pd.Series(level, index=dates).clip(lower=1)
        open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.004, 0.003, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.004, 0.003, len(dates))))
        volume = rng.integers(100_000, 2_500_000, len(dates))
        output[name] = pd.DataFrame(
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
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices.dropna(how="all")
    return prices


def normalize_to_100(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    normalized = prices.copy()
    for col in normalized.columns:
        first_valid = normalized[col].dropna()
        if first_valid.empty or first_valid.iloc[0] == 0:
            normalized[col] = np.nan
        else:
            normalized[col] = normalized[col] / first_valid.iloc[0] * 100
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


def metric_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
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
            month_start = series[series.index.to_period("M") == current_month]
            if len(month_start) > 1 and month_start.iloc[0] != 0:
                mtd = float(series.iloc[-1] / month_start.iloc[0] - 1)
        except Exception:
            mtd = None
        ret_20d = _pct_change(series, 20)
        ret_63d = _pct_change(series, 63)
        ma50 = series.rolling(50, min_periods=30).mean().iloc[-1] if len(series) >= 30 else np.nan
        ma200 = series.rolling(200, min_periods=100).mean().iloc[-1] if len(series) >= 100 else np.nan
        dist_50 = None if pd.isna(ma50) or ma50 == 0 or latest is None else latest / float(ma50) - 1
        dist_200 = None if pd.isna(ma200) or ma200 == 0 or latest is None else latest / float(ma200) - 1
        vol_20 = returns[name].rolling(20, min_periods=10).std().iloc[-1] if name in returns else np.nan
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
                "20D vol ann. %": None if pd.isna(vol_20) else float(vol_20) * np.sqrt(252),
                "last_date": series.index[-1].date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(prices: pd.DataFrame, lookback: int = 90, method: str = "pearson") -> pd.DataFrame:
    returns = compute_returns(prices)
    if lookback and lookback > 0:
        returns = returns.tail(lookback)
    return returns.corr(method=method).round(3)


def rolling_correlation(
    prices: pd.DataFrame,
    base: str,
    peer: str,
    window: int = 60,
) -> pd.Series:
    returns = compute_returns(prices)
    if base not in returns or peer not in returns:
        return pd.Series(dtype=float)
    return returns[base].rolling(window, min_periods=max(10, window // 3)).corr(returns[peer])


def moil_correlation_table(corr: pd.DataFrame, base: str = "MOIL") -> pd.DataFrame:
    if corr.empty or base not in corr.columns:
        return pd.DataFrame(columns=["asset", "correlation_to_moil"])
    values = corr[base].drop(labels=[base], errors="ignore").dropna().sort_values(ascending=False)
    return values.rename("correlation_to_moil").reset_index().rename(columns={"index": "asset"})


def detect_return_anomalies(prices: pd.DataFrame, window: int = 60, threshold: float = 2.5) -> pd.DataFrame:
    returns = compute_returns(prices)
    rows: List[Dict[str, object]] = []
    for col in returns.columns:
        series = returns[col].dropna()
        if len(series) < max(15, window // 2):
            continue
        mean = series.rolling(window, min_periods=max(15, window // 2)).mean().iloc[-1]
        std = series.rolling(window, min_periods=max(15, window // 2)).std().iloc[-1]
        latest = float(series.iloc[-1])
        z = 0.0 if pd.isna(std) or std == 0 else float((latest - mean) / std)
        rows.append(
            {
                "asset": col,
                "latest_return_%": latest,
                "z_score": z,
                "flag": "ANOMALY" if abs(z) >= threshold else "normal",
                "direction": "upside" if z > 0 else "downside" if z < 0 else "flat",
            }
        )
    return pd.DataFrame(rows).sort_values("z_score", key=lambda s: s.abs(), ascending=False)


# Alias for app.py compatibility
detect_anomalies = detect_return_anomalies


def read_manual_macro_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return MANUAL_INDICATOR_TEMPLATE.copy()
    try:
        df = pd.read_csv(path)
    except Exception:
        return MANUAL_INDICATOR_TEMPLATE.copy()
    return normalize_manual_macro(df)


def normalize_manual_macro(df: pd.DataFrame) -> pd.DataFrame:
    template_cols = list(MANUAL_INDICATOR_TEMPLATE.columns)
    if df is None or df.empty:
        return MANUAL_INDICATOR_TEMPLATE.copy()
    out = df.copy()
    for col in template_cols:
        if col not in out.columns:
            out[col] = "" if col != "score" else 0
    out = out[template_cols]
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0).clip(-2, 2)
    out["status"] = out["status"].fillna("neutral").astype(str).str.lower().str.strip()
    return out


def _score_component(
    components: List[Dict[str, object]],
    category: str,
    signal: str,
    points: float,
    max_abs_points: float,
    rationale: str,
) -> None:
    components.append(
        {
            "category": category,
            "signal": signal,
            "points": float(points),
            "max_abs_points": float(max_abs_points),
            "rationale": rationale,
        }
    )


def _series_score_above_ma(
    prices: pd.DataFrame,
    asset: str,
    ma_window: int,
    points: float,
    positive_when_above: bool = True,
) -> Tuple[float, str]:
    if asset not in prices:
        return 0.0, f"{asset} data unavailable."
    series = prices[asset].dropna()
    min_periods = max(20, ma_window // 2)
    if len(series) < min_periods:
        return 0.0, f"{asset} has insufficient history for {ma_window}DMA."
    latest = float(series.iloc[-1])
    ma = float(series.rolling(ma_window, min_periods=min_periods).mean().iloc[-1])
    if not np.isfinite(ma) or ma == 0:
        return 0.0, f"{asset} moving average unavailable."
    above = latest > ma
    bullish = above if positive_when_above else not above
    distance = latest / ma - 1
    direction = "above" if above else "below"
    signed_points = points if bullish else -points
    return signed_points, f"{asset} is {direction} its {ma_window}DMA by {distance:.1%}."


def _series_return_score(
    prices: pd.DataFrame,
    asset: str,
    lookback: int,
    points: float,
    positive_when_up: bool = True,
) -> Tuple[float, str]:
    if asset not in prices:
        return 0.0, f"{asset} data unavailable."
    ret = _pct_change(prices[asset], lookback)
    if ret is None:
        return 0.0, f"{asset} has insufficient history for {lookback}D return."
    bullish = ret > 0 if positive_when_up else ret < 0
    signed_points = points if bullish else -points
    return signed_points, f"{asset} {lookback}D return is {ret:.1%}."


def classify_regime(normalized_score: float) -> Tuple[str, str]:
    if normalized_score >= 70:
        return "Bullish industrial expansion", "Risk-on"
    if normalized_score >= 55:
        return "Constructive / accumulating", "Balanced risk"
    if normalized_score >= 45:
        return "Neutral / watch", "Two-way risk"
    if normalized_score >= 30:
        return "Defensive industrial regime", "Risk-off"
    return "Stress / de-rating risk", "High risk-off"


def compute_regime_score(
    prices: pd.DataFrame,
    manual_macro: Optional[pd.DataFrame] = None,
    groq_scores: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, RegimeSummary]:
    """Compute a transparent composite regime score.

    The score intentionally mixes market-implied signals and manually maintained
    physical/industrial indicators. Each row can be audited and overwritten in future.
    """
    components: List[Dict[str, object]] = []

    score, rationale = _series_score_above_ma(prices, "MOIL", 50, 1.0, True)
    _score_component(components, "MOIL trend", "MOIL vs 50DMA", score, 1.0, rationale)

    score, rationale = _series_score_above_ma(prices, "MOIL", 200, 1.5, True)
    _score_component(components, "MOIL trend", "MOIL vs 200DMA", score, 1.5, rationale)

    score, rationale = _series_return_score(prices, "MOIL", 63, 1.0, True)
    _score_component(components, "MOIL momentum", "MOIL 3M momentum", score, 1.0, rationale)

    score, rationale = _series_score_above_ma(prices, "NIFTY Metal", 50, 1.0, True)
    _score_component(components, "India metals cycle", "NIFTY Metal vs 50DMA", score, 1.0, rationale)

    peer_returns: List[float] = []
    for peer in STEEL_PEERS:
        if peer in prices:
            ret = _pct_change(prices[peer], 63)
            if ret is not None:
                peer_returns.append(ret)
    if peer_returns:
        median_peer_ret = float(np.median(peer_returns))
        peer_points = 1.0 if median_peer_ret > 0 else -1.0
        rationale = f"Median 3M steel peer return is {median_peer_ret:.1%}."
    else:
        peer_points = 0.0
        rationale = "Steel peer return data unavailable."
    _score_component(components, "Steel demand proxy", "JSW/SAIL/Tata median momentum", peer_points, 1.0, rationale)

    score, rationale = _series_score_above_ma(prices, "Brent Crude", 50, 0.75, False)
    _score_component(components, "Energy stress", "Brent below/above 50DMA", score, 0.75, rationale)

    score, rationale = _series_return_score(prices, "USDINR", 20, 0.75, False)
    _score_component(components, "FX stress", "USDINR 1M direction", score, 0.75, rationale)

    score, rationale = _series_score_above_ma(prices, "Baltic Dry Proxy", 50, 0.75, True)
    _score_component(components, "Freight / bulk cycle", "Baltic Dry proxy vs 50DMA", score, 0.75, rationale)

    # Manual macro indicators
    if manual_macro is not None and not manual_macro.empty:
        # Use Groq scores if available, otherwise manual scores
        if groq_scores and "macro_scores" in groq_scores:
            for groq_score in groq_scores["macro_scores"]:
                indicator = groq_score.get("indicator", "")
                score = groq_score.get("score", 0)
                rationale = groq_score.get("rationale", "Groq AI assessment")
                _score_component(
                    components,
                    "Manual Macro (Groq)",
                    f"{indicator} (AI)",
                    score,
                    2.0,
                    rationale,
                )
        else:
            manual = normalize_manual_macro(manual_macro)
            for _, row in manual.iterrows():
                indicator = str(row.get("indicator", "Manual indicator")).strip() or "Manual indicator"
                points = float(row.get("score", 0))
                status = str(row.get("status", "neutral")).strip() or "neutral"
                note = str(row.get("commentary", "")).strip()
                _score_component(
                    components,
                    "Manual physical macro",
                    f"{indicator} ({status})",
                    points,
                    2.0,
                    note,
                )

    scorecard = pd.DataFrame(components)
    if scorecard.empty:
        summary = RegimeSummary(
            score=0,
            max_score=0,
            min_score=0,
            normalized_score=50,
            label="Neutral / watch",
            risk_level="Two-way risk",
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )
        return scorecard, summary

    total_score = float(scorecard["points"].sum())
    max_score = float(scorecard["max_abs_points"].sum())
    min_score = -max_score
    normalized = 50.0 if max_score == 0 else (total_score - min_score) / (max_score - min_score) * 100
    normalized = float(np.clip(normalized, 0, 100))
    label, risk_level = classify_regime(normalized)
    summary = RegimeSummary(
        score=round(total_score, 2),
        max_score=round(max_score, 2),
        min_score=round(min_score, 2),
        normalized_score=round(normalized, 1),
        label=label,
        risk_level=risk_level,
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )
    return scorecard, summary


def generate_rule_based_commentary(
    summary: RegimeSummary,
    scorecard: pd.DataFrame,
    moil_corr: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> str:
    """Generate concise macro commentary without an external LLM dependency."""
    if scorecard.empty:
        return "Macro regime is neutral because there is not enough market or manual macro data to score the dashboard."

    positives = scorecard[scorecard["points"] > 0].sort_values("points", ascending=False).head(3)
    negatives = scorecard[scorecard["points"] < 0].sort_values("points").head(3)
    neutral = scorecard[scorecard["points"] == 0].head(3)

    lines = [
        f"**Regime read-through:** {summary.label} ({summary.normalized_score:.1f}/100).",
        f"The composite score is {summary.score:+.2f} on a possible range of {summary.min_score:+.2f} to {summary.max_score:+.2f}, implying {summary.risk_level.lower()} conditions for MOIL-linked industrial exposure.",
    ]

    if not positives.empty:
        pos_text = "; ".join(
            f"{row['signal']}: {row['rationale']}" for _, row in positives.iterrows()
        )
        lines.append(f"**Supportive signals:** {pos_text}")

    if not negatives.empty:
        neg_text = "; ".join(
            f"{row['signal']}: {row['rationale']}" for _, row in negatives.iterrows()
        )
        lines.append(f"**Pressure points:** {neg_text}")

    if not moil_corr.empty:
        top_corr = moil_corr.head(3)
        corr_text = ", ".join(
            f"{row['asset']} ({row['correlation_to_moil']:+.2f})" for _, row in top_corr.iterrows()
        )
        lines.append(f"**Correlation map:** MOIL is currently most aligned with {corr_text} over the selected lookback.")

    flagged = anomalies[anomalies["flag"] == "ANOMALY"] if not anomalies.empty else pd.DataFrame()
    if not flagged.empty:
        anomaly_text = ", ".join(
            f"{row['asset']} {row['direction']} z={row['z_score']:+.1f}" for _, row in flagged.head(4).iterrows()
        )
        lines.append(f"**Anomaly watch:** {anomaly_text}.")
    elif not neutral.empty:
        missing_text = "; ".join(f"{row['signal']}: {row['rationale']}" for _, row in neutral.iterrows())
        lines.append(f"**Data gaps / watchlist:** {missing_text}")

    lines.append(
        "**Desk action:** update the manual silico-manganese, India steel output, China export and China power-stress rows before using this as a physical-market signal."
    )
    return "\n\n".join(lines)


def generate_groq_commentary_wrapper(
    summary: RegimeSummary,
    scorecard: pd.DataFrame,
    moil_corr: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> str:
    """Generate Groq commentary for the dashboard."""
    context = {
        "regime_label": summary.label,
        "regime_score": summary.normalized_score,
        "manual_macro_score": 0,  # Could be calculated from scorecard
        "top_correlations": moil_corr.head(3).to_dict('records') if not moil_corr.empty else [],
        "anomalies": anomalies.head(5).to_dict('records') if not anomalies.empty else [],
    }
    result = generate_groq_commentary(context)
    if result.get("ok"):
        return result.get("institutional_commentary", "")
    return generate_rule_based_commentary(summary, scorecard, moil_corr, anomalies)


# Legacy alias for backward compatibility
generate_openai_commentary = generate_groq_commentary_wrapper


def fetch_nse_deals(symbol: str = "MOIL") -> Optional[pd.DataFrame]:
    """Fetch recent Bulk and Block deals for a symbol from NSE India.
    
    NSE India has strict anti-scraping. This function uses a session with headers
    to try and bypass basic blocks.
    """
    import requests
    
    # NSE requires a session to get cookies first
    base_url = "https://www.nseindia.com"
    api_url = f"https://www.nseindia.com/api/report-bulk-block-deal?symbol={symbol}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/report-detail/display-bulk-and-block-deals",
    }
    
    try:
        session = requests.Session()
        # Hit home page to get cookies
        session.get(base_url, headers=headers, timeout=10)
        
        # Now hit API
        response = session.get(api_url, headers=headers, timeout=10)
        
        if response.ok:
            data = response.json()
            # NSE API returns data in 'data' key usually
            deals = data.get('data', [])
            if not deals:
                return pd.DataFrame()
            
            df = pd.DataFrame(deals)
            # Standardize columns for our tracker
            # Typical NSE columns: 'date', 'clientName', 'type', 'quantity', 'price'
            return df
        return None
    except Exception:
        return None


def read_news_tracker_csv(path: Path) -> pd.DataFrame:
    """Read or initialize the News & Institutional tracker with detailed March 2026 data."""
    if path.exists():
        return pd.read_csv(path)
    
    # Pre-populate with user-provided March 2026 data
    default_data = [
        # Mutual Funds
        {"category": "Mutual Fund", "item": "Bandhan Large & Mid Cap", "value": "0.91%", "impact": "1.0", "details": "Highest MF stake: 1,847,267 shares"},
        {"category": "Mutual Fund", "item": "Tata Resources & Energy", "value": "0.29%", "impact": "0.5", "details": "600,000 shares"},
        {"category": "Mutual Fund", "item": "Bandhan Small Cap", "value": "0.28%", "impact": "2.0", "details": "Increased holding by 107.93% in Mar 2026"},
        {"category": "Mutual Fund", "item": "LIC MF Large Cap", "value": "0.22%", "impact": "0.5", "details": "438,384 shares"},
        {"category": "Mutual Fund", "item": "LIC MF Multi Asset", "value": "0.18%", "impact": "0.5", "details": "369,800 shares"},
        {"category": "Mutual Fund", "item": "Canara Robeco Mfg", "value": "0.17%", "impact": "0.5", "details": "350,708 shares"},
        
        # Institutional ETFs/Global
        {"category": "Institutional", "item": "iShares MSCI EM ETF", "value": "0.41%", "impact": "0.5", "details": "838,153 shares"},
        {"category": "Institutional", "item": "DFA Emerging Markets Core", "value": "0.33%", "impact": "0.5", "details": "667,739 shares"},
        {"category": "Institutional", "item": "Nuveen Int Small Cap", "value": "0.30%", "impact": "0.5", "details": "605,239 shares"},
        
        # Recent Actions
        {"category": "Fund Action", "item": "ABSL Small Cap", "value": "Exited", "impact": "-2.0", "details": "Completely sold off stake (Mar 2026)"},
        {"category": "Fund Action", "item": "Samco Special Opp", "value": "Exited", "impact": "-1.5", "details": "Exited position (Mar 2026)"},
        {"category": "News / Deratings", "item": "Negative News / Fund Reports", "value": "Neutral", "impact": "0", "details": "No major negative reports."},
    ]
    return pd.DataFrame(default_data)


def format_alert_message(
    summary: RegimeSummary,
    scorecard: pd.DataFrame,
    anomalies: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    manual_macro: Optional[pd.DataFrame] = None,
    news_tracker: Optional[pd.DataFrame] = None,
) -> str:
    top_positive = scorecard[scorecard["points"] > 0].sort_values("points", ascending=False).head(2)
    top_negative = scorecard[scorecard["points"] < 0].sort_values("points").head(2)
    flagged = anomalies[anomalies["flag"] == "ANOMALY"].head(3) if not anomalies.empty else pd.DataFrame()

    parts = [
        "MOIL Macro Radar",
        f"Regime: {summary.label}",
        f"Score: {summary.normalized_score:.1f}/100 ({summary.score:+.2f})",
        f"Risk tone: {summary.risk_level}",
        "", # Spacer
    ]

    # Add Technicals
    if prices is not None and "MOIL" in prices.columns:
        moil_series = prices["MOIL"].dropna()
        if not moil_series.empty:
            latest_price = moil_series.iloc[-1]
            ma50 = moil_series.rolling(50, min_periods=10).mean().iloc[-1]
            ma200 = moil_series.rolling(200, min_periods=50).mean().iloc[-1]
            
            # Simple Support/Resistance (60-day window)
            window = 60
            if len(moil_series) >= window:
                recent = moil_series.tail(window)
                support = recent.min()
                resistance = recent.max()
            else:
                support = moil_series.min()
                resistance = moil_series.max()

            parts.append(f"MOIL Price: {latest_price:.2f}")
            parts.append(f"S/R: {support:.1f} / {resistance:.1f}")
            parts.append(f"50/200 DMA: {ma50:.1f} / {ma200:.1f}")
            parts.append("")

    # Add Manganese Price if available
    if manual_macro is not None and not manual_macro.empty:
        # Match indicator case-insensitively
        try:
            mn_row = manual_macro[manual_macro["indicator"].str.contains("manganese", case=False, na=False)].iloc[0]
            val = mn_row.get("value", "N/A")
            unit = mn_row.get("unit", "")
            parts.append(f"Manganese Price: {val} {unit}")
            parts.append("")
        except (IndexError, KeyError):
            pass

    # Add News & Institutional Activity
    if news_tracker is not None and not news_tracker.empty:
        news_parts = []
        for _, row in news_tracker.iterrows():
            item = row.get("item", "Unknown")
            val = row.get("value", "N/A")
            details = row.get("details", "")
            impact = float(row.get("impact", 0))
            
            if impact != 0 or val not in ["Neutral", "Stable"]:
                msg = f"- {item}: {val}"
                if details and details != "None":
                    msg += f" ({details})"
                news_parts.append(msg)
        
        if news_parts:
            parts.append("News & Institutional:")
            parts.extend(news_parts)
            parts.append("")

    if not top_positive.empty:
        parts.append("Support: " + "; ".join(str(x) for x in top_positive["signal"].tolist()))
    if not top_negative.empty:
        parts.append("Pressure: " + "; ".join(str(x) for x in top_negative["signal"].tolist()))
    
    if not flagged.empty:
        parts.append(
            "Anomalies: "
            + "; ".join(f"{row['asset']} z={row['z_score']:+.1f}" for _, row in flagged.iterrows())
        )
    
    parts.append(f"Updated: {summary.updated_at}")
    return "\n".join(parts)


# Alias for app.py compatibility
build_telegram_alert_text = format_alert_message

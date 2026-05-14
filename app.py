"""MOIL Macro Radar - Streamlit application.

Institutional regime-intelligence dashboard for MOIL Ltd, manganese,
ferroalloys, Indian steel-cycle proxies, China export pressure, energy/FX
stress, institutional flow monitoring and Groq/Llama 3.3 AI scoring.
"""

from __future__ import annotations
import os
import io
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _st_components

from groq_macro_scorer import (
    generate_groq_commentary,
    get_groq_config,
    is_groq_configured,
    score_macro_with_groq,
)
from macro_radar import (
    CORE_TICKERS,
    DEFAULT_MARKET_TICKERS,
    build_groq_prompt_for_copy,
    build_price_panel,
    build_telegram_alert_text,
    build_volume_panel,
    compute_data_quality,
    compute_moil_correlation_ranking,
    compute_regime_score,
    compute_rolling_correlations,
    compute_technical_levels,
    RegimeSummary,
    correlation_matrix,
    detect_anomalies,
    fetch_market_data,
    fetch_nse_deals,
    generate_demo_market_data,
    generate_rule_based_commentary,
    localize_to_ist,
    metric_snapshot,
    normalize_manual_macro,
    normalize_to_100,
    read_manual_macro_csv,
    read_news_tracker_csv,
    run_market_validation,
)
from auto_feeds import (
    fetch_nse_deals_auto,
    fetch_source_aware_macro_feed,
    is_any_vendor_feed_configured,
    is_cached_macro_feed_available,
    is_serper_configured,
    nse_deals_to_tracker_updates,
    source_provider_status,
)
from backend_cache_loader import (
    load_latest_institutional_cache,
    load_latest_macro_cache,
    macro_cache_to_evidence_rows,
    macro_cache_to_source_readiness,
    institutional_cache_summary,
    institutional_cache_to_evidence_rows,
    institutional_cache_to_scorecard,
    institutional_cache_to_source_readiness,
    institutional_quality_summary_v2,
)
from institutional_flow.config import tracker_config
from institutional_flow.features import combine_features
from institutional_flow.disclosures import classify_trigger
from institutional_flow.mf_holdings import load_mf_manual_file
from institutional_flow.moil_shp import load_manual_shp_csv
from intraday_vwap_orb.config import StrategyConfig
from intraday_vwap_orb.data import load_intraday, prepare_intraday
from intraday_vwap_orb.backtest import run_backtest
from intraday_vwap_orb.features import daily_features, merge_with_institutional
from intraday_vwap_orb.mtf import (
    run_composite_engine,
    AdaptiveWeights,
    MTFPaperTrade,
    resolve_mtf_paper_trade,
    mtf_paper_log_to_df,
    summarise_mtf_track,
    run_mtf_backtest,
    build_validation_table,
)
from intraday_vwap_orb.confidence import (
    score_signal_confidence,
    AUTO_TRADE_THRESHOLD,
    MANUAL_REVIEW_THRESHOLD,
)
from intraday_vwap_orb.paper_trading import (
    PaperSignal,
    classify_signal,
    resolve_paper_trade,
    paper_log_to_df,
    summarise_track,
)
from data_layer.evidence_store import evidence_to_macro_rows, load_evidence_from_file
from data_layer.validators import validate_evidence_dataframe
from telegram_alert import send_telegram_alert, build_trade_alert_message, build_mtf_alert_message
from model_evidence import compute_intraday_evidence, compute_mtf_evidence
from data_feeds import fetch_ohlcv, fetch_daily, FeedSource, source_badge, _get_secret


st.set_page_config(
    page_title="MOIL Macro Radar",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern UI theming — TradingView / Zerodha Kite dark-desk style
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Base ─────────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #131722 !important;
        color: #d1d4dc !important;
    }
    /* Fix: push content below Streamlit's top toolbar (~60px) */
    .block-container { padding-top: 4.5rem; padding-bottom: 2rem; max-width: 1440px; }

    /* Hide the default Streamlit header bar so it doesn't overlap */
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Sidebar ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #1e2130 !important;
        border-right: 1px solid #2a2e39;
    }
    [data-testid="stSidebar"] * { color: #d1d4dc !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label { color: #b2b5be !important; font-size: 0.82rem; }

    /* ── All text ─────────────────────────────────────────────── */
    p, span, li, td, th, label, div { color: #d1d4dc; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 600; letter-spacing: -0.3px; }
    .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] { color: #868993 !important; font-size: 0.82rem; }

    /* ── Metric cards ─────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2a2e39;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] { color: #868993 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem; }

    /* ── Tabs ─────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { background: #1e2130; border-radius: 8px; padding: 4px 6px; border: 1px solid #2a2e39; gap: 2px; }
    .stTabs [data-baseweb="tab"] { color: #868993 !important; font-weight: 500; font-size: 0.85rem; background: transparent; border: none; border-radius: 6px; padding: 6px 14px; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; background: #2962ff !important; font-weight: 600; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton>button {
        background: #2962ff;
        color: #ffffff !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.1rem;
        font-weight: 600;
        font-size: 0.85rem;
        transition: background 0.15s, transform 0.1s;
    }
    .stButton>button:hover { background: #1e53e5; transform: translateY(-1px); }
    .stDownloadButton>button { background: #1e2130; color: #d1d4dc !important; border: 1px solid #2a2e39; border-radius: 6px; font-size: 0.82rem; }
    .stDownloadButton>button:hover { background: #2a2e39; }

    /* ── DataFrames / Tables ──────────────────────────────────── */
    .stDataFrame, [data-testid="stDataFrame"] {
        background: #1e2130 !important;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame thead tr th {
        background: #131722 !important;
        color: #868993 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        border-bottom: 1px solid #2a2e39 !important;
    }
    .stDataFrame tbody tr td { color: #d1d4dc !important; font-size: 0.82rem !important; background: #1e2130 !important; }
    .stDataFrame tbody tr:hover td { background: #252a3a !important; }

    /* ── Inputs / Selects ─────────────────────────────────────── */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox [data-baseweb="select"]>div,
    .stTextArea textarea {
        background: #1e2130 !important;
        color: #d1d4dc !important;
        border: 1px solid #2a2e39 !important;
        border-radius: 6px;
    }
    .stSelectbox [data-baseweb="select"] div[aria-selected] { color: #d1d4dc !important; }
    .stSlider [data-testid="stThumbValue"] { color: #d1d4dc !important; }
    .stSlider label { color: #868993 !important; font-size: 0.82rem; }
    .stCheckbox label { color: #d1d4dc !important; }

    /* ── Expander ─────────────────────────────────────────────── */
    .streamlit-expanderHeader { background: #1e2130 !important; color: #d1d4dc !important; border: 1px solid #2a2e39; border-radius: 8px; }
    .streamlit-expanderContent { background: #1a1f2e !important; border: 1px solid #2a2e39; border-top: none; border-radius: 0 0 8px 8px; }

    /* ── Alerts / info boxes ──────────────────────────────────── */
    .stSuccess { background: #0d2a1e !important; color: #26a69a !important; border-left: 3px solid #26a69a; border-radius: 6px; }
    .stInfo { background: #0d1f3a !important; color: #2196f3 !important; border-left: 3px solid #2196f3; border-radius: 6px; }
    .stWarning { background: #2a1f0d !important; color: #ff9800 !important; border-left: 3px solid #ff9800; border-radius: 6px; }
    .stError { background: #2a0d0d !important; color: #ef5350 !important; border-left: 3px solid #ef5350; border-radius: 6px; }

    /* ── File uploader ────────────────────────────────────────── */
    [data-testid="stFileUploader"] { background: #1e2130; border: 1px dashed #2a2e39; border-radius: 8px; padding: 8px; }
    [data-testid="stFileUploader"] label { color: #868993 !important; font-size: 0.82rem; }

    /* ── Section divider ──────────────────────────────────── */
    hr { border-color: #2a2e39 !important; }

    /* ── Mobile / responsive ─────────────────────────────── */
    /* Touch-friendly tap targets */
    .stButton>button { min-height: 44px; min-width: 44px; }
    .stTabs [data-baseweb="tab"] { min-height: 40px; padding: 8px 12px; }

    @media (max-width: 768px) {
        /* Full-width block container with comfortable padding */
        .block-container { padding-top: 4rem !important; padding-left: 0.75rem !important; padding-right: 0.75rem !important; max-width: 100% !important; }

        /* Stack columns vertically on small screens */
        [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }

        /* Smaller metric values so they don't overflow */
        [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        [data-testid="metric-container"] { padding: 10px 12px !important; }

        /* Tabs scroll horizontally on mobile */
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; gap: 0; }
        .stTabs [data-baseweb="tab"] { font-size: 0.78rem !important; padding: 6px 10px !important; white-space: nowrap; }

        /* Sidebar collapses by default — Streamlit handles this natively */
        /* Charts fill full width */
        .js-plotly-plot, .plotly { width: 100% !important; }

        /* Title font scale */
        h1 { font-size: 1.3rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1rem !important; }

        /* Button full width */
        .stButton>button { width: 100%; font-size: 0.9rem; }

        /* DataFrames scroll horizontally */
        .stDataFrame { overflow-x: auto !important; }

        /* Sliders full width */
        .stSlider { padding: 0 !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path("data")
MANUAL_TEMPLATE_PATH = DATA_DIR / "manual_macro_template.csv"
NEWS_TRACKER_PATH = DATA_DIR / "news_tracker.csv"


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_data(ticker_items: Tuple[Tuple[str, str], ...], period: str, interval: str, demo: bool):
    tickers = dict(ticker_items)
    if demo:
        return generate_demo_market_data(tickers.keys()), {}
    return fetch_market_data(tickers, period=period, interval=interval)


@st.cache_data(ttl=300, show_spinner=False)
def cached_intraday_data(ticker_items: Tuple[Tuple[str, str], ...], demo: bool):
    tickers = dict(ticker_items)
    if demo:
        return generate_demo_market_data(tickers.keys(), periods=240), {}
    return fetch_market_data(tickers, period="5d", interval="5m")


def pct_fmt(value) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):+.2%}"
    except Exception:
        return ""


def num_fmt(value, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def build_institutional_features(prefer_live: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Build institutional features, preferring live NSE deals when available."""
    cfg = tracker_config({"enable_network": prefer_live})
    sample_dir = Path("data/sample")
    diagnostics: list[str] = []

    # Deals: try live NSE first, then sample fallback
    deals = pd.DataFrame()
    if prefer_live:
        try:
            deals, diag_live = fetch_nse_deals_auto(cfg.symbol)
            diagnostics.extend(diag_live)
            if deals is None:
                deals = pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(f"Live NSE fetch failed: {exc}")
    if deals.empty:
        deals_path = sample_dir / "sample_deals.csv"
        deals = pd.read_csv(deals_path) if deals_path.exists() else pd.DataFrame()
        diagnostics.append(f"Deals fallback rows: {len(deals)} from {deals_path}" if not deals.empty else "Deals sample missing or empty")

    # Shareholding (manual CSV template)
    shp_path = sample_dir / "manual_shp_template.csv"
    shp = load_manual_shp_csv(shp_path) if shp_path.exists() else pd.DataFrame()
    diagnostics.append(f"SHP rows: {len(shp)} from {shp_path}" if not shp.empty else "SHP sample missing or empty")

    # Mutual funds (manual CSV/XLSX)
    mf_path = sample_dir / "sample_mf.csv"
    mf = load_mf_manual_file(mf_path) if mf_path.exists() else pd.DataFrame()
    diagnostics.append(f"MF rows: {len(mf)} from {mf_path}" if not mf.empty else "MF sample missing or empty")

    # Disclosures (manual CSV)
    disc_path = sample_dir / "sample_disclosures.csv"
    disclosures = pd.read_csv(disc_path) if disc_path.exists() else pd.DataFrame()
    if not disclosures.empty and "trigger_type" not in disclosures.columns:
        disclosures["trigger_type"] = disclosures.apply(lambda r: classify_trigger(r, cfg.outstanding_shares), axis=1)
    diagnostics.append(f"Disclosure rows: {len(disclosures)} from {disc_path}" if not disclosures.empty else "Disclosure sample missing or empty")

    features = combine_features(deals, shp, mf, disclosures, cfg)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not features.empty:
        features.to_csv(reports_dir / "moil_institutional_features.csv", index=False)
    summary = reports_dir / "moil_institutional_flow_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# MOIL Institutional Flow (auto/manual blend)",
                f"Deals rows: {len(deals)}; SHP rows: {len(shp)}; MF rows: {len(mf)}; Disclosure rows: {len(disclosures)}",
            ]
        ),
        encoding="utf-8",
    )
    return features, diagnostics


def format_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    out = snapshot.copy()
    pct_cols = ["1D %", "MTD %", "20D %", "63D %", "vs 50DMA %", "vs 200DMA %", "20D vol ann."]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].apply(pct_fmt)
    for col in ["latest", "relative_volume", "volume_z_score"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: num_fmt(x, 2))
    return out


def ensure_regime_summary(obj) -> RegimeSummary:
    if isinstance(obj, RegimeSummary):
        return obj
    if isinstance(obj, dict):
        try:
            return RegimeSummary(**obj)
        except Exception:
            pass
    return RegimeSummary(0.0, 1.0, -1.0, 50.0, "Neutral / mixed cycle", "Unknown", "Watch", 0.0, 0.0, "n/a")


def reset_market_cache():
    cached_market_data.clear()


@st.cache_data(ttl=60, show_spinner=False)
def cached_index_quotes() -> dict:
    """Fetch NIFTY 50 and Sensex last close / change via priority chain. Cached 60 s."""
    result = {"NIFTY 50": None, "Sensex": None}
    for label, ticker in [("NIFTY 50", "^NSEI"), ("Sensex", "^BSESN")]:
        try:
            df, _src, _err = fetch_ohlcv(ticker, interval="1d", days=3)
            if df is not None and not df.empty and "close" in df.columns:
                _last = float(df["close"].iloc[-1])
                _prev = float(df["close"].iloc[-2]) if len(df) >= 2 else _last
                _chg  = _last - _prev
                _pct  = (_chg / _prev * 100) if _prev else 0.0
                result[label] = {"price": _last, "change": _chg, "pct": _pct, "source": _src}
        except Exception:
            pass
    return result


# Sidebar -----------------------------------------------------------------
st.sidebar.title("MOIL Macro Radar")
st.sidebar.caption("Regime intelligence for MOIL, manganese and Indian metals")

# ── Data feed status ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📡 Data feed")
try:
    _dhan_ok  = bool(_get_secret("dhan", "client_id")   and _get_secret("dhan", "access_token"))
    _kite_ok  = bool(_get_secret("kite", "api_key")     and _get_secret("kite", "access_token"))
    _tv_ok    = True  # tvdatafeed works without credentials
    if _dhan_ok:
        st.sidebar.success("🟢 TradingView → Dhan → Kite → yfinance")
    elif _kite_ok:
        st.sidebar.success("🟢 TradingView → Zerodha Kite → yfinance")
    else:
        st.sidebar.success("🟢 TradingView (free, live NSE data)")
        st.sidebar.caption("Optionally add `[dhan]` or `[kite]` in secrets for broker-grade feeds.")
except Exception:
    pass

# ── Mode selector ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("#### Dashboard mode")
_mode_col1, _mode_col2 = st.sidebar.columns(2)
_is_trader = _mode_col1.button(
    "📈 Trader",
    use_container_width=True,
    type="primary" if st.session_state.get("app_mode", "investor") == "trader" else "secondary",
)
_is_investor = _mode_col2.button(
    "🏦 Investor",
    use_container_width=True,
    type="primary" if st.session_state.get("app_mode", "investor") == "investor" else "secondary",
)
if _is_trader:
    st.session_state.app_mode = "trader"
    st.rerun()
if _is_investor:
    st.session_state.app_mode = "investor"
    st.rerun()
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "investor"
APP_MODE = st.session_state.app_mode
st.sidebar.markdown(
    f"<div style='background:#1e2130;border:1px solid #2a2e39;border-radius:6px;"
    f"padding:6px 12px;text-align:center;font-size:0.82rem;color:"
    f"{'#26a69a' if APP_MODE == 'trader' else '#2962ff'};font-weight:600;margin-bottom:4px;'>"
    f"{'📈 Trader mode active — intraday + signals focused' if APP_MODE == 'trader' else '🏦 Investor mode active — macro + regime focused'}"
    f"</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

period = st.sidebar.selectbox("Market history", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Sampling interval", ["1d", "1wk"], index=0)
if APP_MODE == "investor":
    correlation_lookback = st.sidebar.slider("Correlation window", 30, 252, 90, 5)
    anomaly_window = st.sidebar.slider("Anomaly window", 20, 180, 60, 5)
else:
    correlation_lookback = 90
    anomaly_window = 60
use_demo_data = st.sidebar.checkbox("Use synthetic demo data", value=False)

with st.sidebar.expander("Ticker map", expanded=False):
    edited_tickers: Dict[str, str] = {}
    for name, ticker in DEFAULT_MARKET_TICKERS.items():
        edited_tickers[name] = st.text_input(name, value=ticker, key=f"ticker_{name}")

if APP_MODE == "trader":
    with st.sidebar.expander("Trader quick settings", expanded=True):
        st.caption("Thresholds used in the Intraday tab")
        st.metric("Auto-trade confidence", f"≥ {AUTO_TRADE_THRESHOLD}")
        st.metric("Manual-review confidence", f"≥ {MANUAL_REVIEW_THRESHOLD}")
        enable_groq_scoring = False
        enable_groq_commentary = False
        enable_telegram = st.checkbox("Enable Telegram alerts", value=True, key="tg_trader")
else:
    with st.sidebar.expander("AI / alert controls", expanded=True):
        enable_groq_scoring = st.checkbox("Enable Groq/Llama 3.3 macro scoring", value=True)
        enable_groq_commentary = st.checkbox("Enable Groq commentary", value=True)
        enable_telegram = st.checkbox("Enable Telegram alerts", value=True)

if st.sidebar.button("Reload market data"):
    reset_market_cache()
    st.rerun()

# Initialize durable session state ----------------------------------------
if "manual_macro" not in st.session_state:
    st.session_state.manual_macro = read_manual_macro_csv(MANUAL_TEMPLATE_PATH)
if "news_tracker" not in st.session_state:
    st.session_state.news_tracker = read_news_tracker_csv(NEWS_TRACKER_PATH)
    if "sentiment" not in st.session_state.news_tracker.columns:
        st.session_state.news_tracker["sentiment"] = "Neutral"
if "groq_scores" not in st.session_state:
    st.session_state.groq_scores = None
if "groq_scores_applied" not in st.session_state:
    st.session_state.groq_scores_applied = None
if "groq_commentary" not in st.session_state:
    st.session_state.groq_commentary = None
if "auto_macro_diagnostics" not in st.session_state:
    st.session_state.auto_macro_diagnostics = []
if "nse_sync_diagnostics" not in st.session_state:
    st.session_state.nse_sync_diagnostics = []
if "evidence_validation_warnings" not in st.session_state:
    st.session_state.evidence_validation_warnings = []
if "paper_log" not in st.session_state:
    st.session_state.paper_log = []
if "pending_signals" not in st.session_state:
    st.session_state.pending_signals = []
if "mtf_adaptive_weights" not in st.session_state:
    st.session_state.mtf_adaptive_weights = AdaptiveWeights()
if "mtf_trade_log" not in st.session_state:
    st.session_state.mtf_trade_log = []
if "mtf_auto_intraday_count" not in st.session_state:
    st.session_state.mtf_auto_intraday_count = 0
if "mtf_auto_margin_count" not in st.session_state:
    st.session_state.mtf_auto_margin_count = 0
if "mtf_paper_log" not in st.session_state:
    st.session_state.mtf_paper_log = []
if "mtf_backtest_result" not in st.session_state:
    st.session_state.mtf_backtest_result = None


def _fire_trade_alert(sc: dict, opp_date, opp_or_high: float, opp_or_low: float,
                      or_range_pct: float, track: str, approved_by: str,
                      opp_day, enabled: bool) -> None:
    """Send Telegram trade alert if enabled; show toast result in UI."""
    if not enabled:
        return
    _vwap_val = float(sc["row"].get("vwap", sc["entry_px"]) if hasattr(sc["row"], "get") else sc["row"]["vwap"])
    _signal_time_val = str(sc["row"].get("time", "") if hasattr(sc["row"], "get") else sc["row"]["time"])
    _msg = build_trade_alert_message(
        symbol="MOIL",
        side=sc["sig"].side,
        entry_price=sc["entry_px"],
        stop=sc["stop_px"],
        target=sc["target_px"],
        confidence_score=sc["cr"].score,
        confidence_label=sc["cr"].label,
        volume_ratio=sc["vol_ratio"],
        or_high=opp_or_high,
        or_low=opp_or_low,
        or_range_pct=or_range_pct,
        vwap=_vwap_val,
        signal_time=_signal_time_val,
        trade_date=str(opp_date),
        track=track,
        approved_by=approved_by,
    )
    _ok, _detail = send_telegram_alert(_msg)
    if _ok:
        st.toast("📲 Trade alert sent to Telegram!", icon="✅")
    else:
        st.toast(f"Telegram: {_detail}", icon="⚠️")


# Data load ----------------------------------------------------------------
_mode_badge_color = "#26a69a" if APP_MODE == "trader" else "#2962ff"
_mode_badge_label = "📈 Trader Mode" if APP_MODE == "trader" else "🏦 Investor Mode"
_mode_badge_desc = "Intraday signals · Trade opportunities · Paper trading" if APP_MODE == "trader" else "Macro regime · Institutional flow · AI scoring · Validation"
st.markdown(
    f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:4px;'>"
    f"<span style='font-size:1.6rem;font-weight:700;color:#d1d4dc;'>MOIL Macro Radar</span>"
    f"<span style='background:{_mode_badge_color}22;border:1px solid {_mode_badge_color};border-radius:20px;"
    f"padding:3px 14px;font-size:0.78rem;font-weight:600;color:{_mode_badge_color};'>"
    f"{_mode_badge_label}</span></div>"
    f"<div style='color:#868993;font-size:0.82rem;margin-bottom:8px;'>{_mode_badge_desc}</div>",
    unsafe_allow_html=True,
)

# ── Live market ticker bar (IST clock + indices) ──────────────────────────
_idx_quotes = cached_index_quotes()

def _idx_chip(label: str, q: dict | None) -> str:
    if q is None:
        return (
            f"<span style='display:inline-flex;align-items:center;gap:6px;"
            f"background:#1e2130;border:1px solid #2a2e39;border-radius:6px;"
            f"padding:4px 12px;font-size:0.78rem;color:#868993;'>"
            f"{label} <span style='color:#4a4e5a;'>n/a</span></span>"
        )
    _color = "#26a69a" if q["change"] >= 0 else "#ef5350"
    _arrow = "▲" if q["change"] >= 0 else "▼"
    return (
        f"<span style='display:inline-flex;align-items:center;gap:6px;"
        f"background:#1e2130;border:1px solid #2a2e39;border-radius:6px;"
        f"padding:4px 14px;font-size:0.78rem;'>"
        f"<span style='color:#868993;font-weight:500;'>{label}</span>"
        f"<span style='color:#d1d4dc;font-weight:600;'>{q['price']:,.2f}</span>"
        f"<span style='color:{_color};font-size:0.72rem;'>"
        f"{_arrow} {abs(q['change']):,.2f} ({q['pct']:+.2f}%)</span></span>"
    )

_n50_chip = _idx_chip("NIFTY 50", _idx_quotes.get("NIFTY 50"))
_bse_chip = _idx_chip("Sensex", _idx_quotes.get("Sensex"))

_ticker_left, _ticker_right = st.columns([3, 1])
with _ticker_left:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;padding:6px 0;'>"
        f"{_n50_chip}{_bse_chip}"
        f"</div>",
        unsafe_allow_html=True,
    )
with _ticker_right:
    _st_components.html(
        """
<style>
  body { margin:0; background:transparent; font-family:'Inter',sans-serif; }
  #wrap { display:flex; flex-direction:column; align-items:flex-end; justify-content:center;
          height:48px; padding-right:4px; }
  #ist-clock { color:#d1d4dc; font-size:1.05rem; font-weight:700; font-family:monospace;
               background:#1e2130; border:1px solid #2a2e39; border-radius:5px;
               padding:3px 10px; letter-spacing:1px; }
  #ist-date  { color:#868993; font-size:0.72rem; margin-top:2px; }
  #ist-label { color:#4a4e5a; font-size:0.65rem; margin-bottom:1px; }
</style>
<div id='wrap'>
  <div id='ist-label'>IST</div>
  <div id='ist-clock'>--:--:--</div>
  <div id='ist-date'>--- -- ----</div>
</div>
<script>
(function() {
  function updateClock() {
    var now = new Date();
    var ist = new Date(now.toLocaleString('en-US', {timeZone: 'Asia/Kolkata'}));
    var h = String(ist.getHours()).padStart(2,'0');
    var m = String(ist.getMinutes()).padStart(2,'0');
    var s = String(ist.getSeconds()).padStart(2,'0');
    var days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    document.getElementById('ist-clock').innerText = h+':'+m+':'+s;
    document.getElementById('ist-date').innerText  =
      days[ist.getDay()]+' '+String(ist.getDate()).padStart(2,'0')+' '+
      months[ist.getMonth()]+' '+ist.getFullYear();
  }
  updateClock();
  setInterval(updateClock, 1000);
})();
</script>
        """,
        height=50,
    )

with st.spinner("Loading market data..."):
    data, errors = cached_market_data(tuple(edited_tickers.items()), period, interval, use_demo_data)

if not data:
    st.warning("No live market data returned. Switching to deterministic demo data for dashboard continuity.")
    data = generate_demo_market_data(edited_tickers.keys())
    errors = {"fallback": "Live feed returned no usable market data; demo data is active."}

prices = build_price_panel(data)
volumes = build_volume_panel(data)
snapshot = metric_snapshot(prices, volumes)
data_quality_score, data_quality_table = compute_data_quality(data, st.session_state.manual_macro, edited_tickers)
active_groq_scores = st.session_state.groq_scores_applied
scorecard, summary = compute_regime_score(
    prices,
    st.session_state.manual_macro,
    active_groq_scores,
    volumes=volumes,
    data_quality_score=data_quality_score,
)
summary = ensure_regime_summary(summary)
corr = correlation_matrix(prices, lookback=correlation_lookback)
moil_corr = compute_moil_correlation_ranking(prices, lookback=correlation_lookback)
rolling_corrs = compute_rolling_correlations(
    prices,
    [x for x in ["NIFTY Metal", "JSW Steel", "SAIL", "Tata Steel", "Brent Crude", "USDINR", "Baltic Dry Proxy"] if x in prices.columns],
    window=max(30, min(correlation_lookback, 120)),
)
anomalies = detect_anomalies(prices, volumes, window=anomaly_window, threshold=2.5)
technical_levels = compute_technical_levels(data, "MOIL")
validation = run_market_validation(prices)

# Top metrics --------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Regime score", f"{summary.normalized_score:.1f}/100", summary.alert_level)
col2.metric("Regime", summary.label)
col3.metric("Confidence", f"{summary.confidence_score:.1f}/100")
col4.metric("Data quality", f"{summary.data_quality_score:.1f}/100")
moil_row = snapshot[snapshot["asset"] == "MOIL"] if not snapshot.empty and "asset" in snapshot.columns else pd.DataFrame()
if not moil_row.empty:
    col5.metric("MOIL latest", num_fmt(moil_row.iloc[0].get("latest")), pct_fmt(moil_row.iloc[0].get("20D %")))
else:
    col5.metric("MOIL latest", "n/a")

# Tabs — conditional on mode ---------------------------------------------
if APP_MODE == "trader":
    _tab_labels = [
        "Market radar",
        "Correlation engine",
        "Regime score",
        "Institutional sync",
        "Alerts & commentary",
        "📈 Intraday",
        "📊 Model Evidence",
    ]
else:
    _tab_labels = [
        "Market radar",
        "Correlation engine",
        "Regime score",
        "Manual macro tracker",
        "Groq AI desk",
        "Institutional sync",
        "Alerts & commentary",
        "Validation",
        "Data quality",
    ]
tabs = st.tabs(_tab_labels)

# Helper: resolve tab index safely (returns None if not in this mode)
def _tab(name: str):
    try:
        return tabs[_tab_labels.index(name)]
    except ValueError:
        return None

_t0 = _tab("Market radar")
with _t0:
    st.subheader("Market radar")
    left, right = st.columns([2, 1])
    with left:
        default_assets = [x for x in ["MOIL", "NIFTY Metal", "JSW Steel", "SAIL", "Tata Steel", "Brent Crude", "USDINR"] if x in prices.columns]
        selected_assets = st.multiselect("Assets to plot", list(prices.columns), default=default_assets[:5])
        if selected_assets:
            norm = normalize_to_100(prices[selected_assets]).dropna(how="all")
            fig = go.Figure()
            for col in norm.columns:
                fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
            fig.update_layout(title="Normalized performance indexed to 100", yaxis_title="Index", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        asset = st.selectbox("Instrument detail", list(prices.columns), index=list(prices.columns).index("MOIL") if "MOIL" in prices.columns else 0)
        if asset in prices.columns:
            series = prices[asset].dropna()
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=asset))
            for window in [20, 50, 200]:
                if len(series) >= window:
                    price_fig.add_trace(go.Scatter(x=series.index, y=series.rolling(window).mean(), mode="lines", name=f"{window}DMA"))
            price_fig.update_layout(title=f"{asset} trend", yaxis_title="Price / index", hovermode="x unified")
            st.plotly_chart(price_fig, use_container_width=True)

        if "MOIL" in data and not data["MOIL"].empty:
            moil_df = data["MOIL"].dropna(subset=["Open", "High", "Low", "Close"])
            if not moil_df.empty:
                candle = go.Figure(
                    go.Candlestick(
                        x=moil_df.index,
                        open=moil_df["Open"],
                        high=moil_df["High"],
                        low=moil_df["Low"],
                        close=moil_df["Close"],
                        name="MOIL",
                    )
                )
                candle.update_layout(title="MOIL candlestick structure", xaxis_rangeslider_visible=False, yaxis_title="Price")
                st.plotly_chart(candle, use_container_width=True)

    with right:
        st.markdown("#### Latest snapshot")
        st.dataframe(format_snapshot(snapshot), use_container_width=True, hide_index=True)
        st.markdown("#### MOIL technical levels")
        st.json(technical_levels or {"status": "MOIL OHLCV unavailable"})

_t1 = _tab("Correlation engine")
with _t1:
    st.subheader("Correlation engine")
    if corr.empty:
        st.info("Not enough overlapping return history for correlations.")
    else:
        heatmap = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, title=f"{correlation_lookback}-session return correlation matrix")
        st.plotly_chart(heatmap, use_container_width=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### MOIL correlation ranking")
            st.dataframe(moil_corr, use_container_width=True, hide_index=True)
        with c2:
            if not rolling_corrs.empty:
                roll_fig = go.Figure()
                for col in rolling_corrs.columns:
                    roll_fig.add_trace(go.Scatter(x=rolling_corrs.index, y=rolling_corrs[col], mode="lines", name=col))
                roll_fig.update_layout(title="Rolling MOIL correlations", yaxis_range=[-1, 1], hovermode="x unified")
                st.plotly_chart(roll_fig, use_container_width=True)

_t2 = _tab("Regime score")
with _t2:
    st.subheader("Composite regime score")
    g1, g2 = st.columns([1, 2])
    with g1:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=summary.normalized_score,
                number={"suffix": "/100"},
                title={"text": summary.label},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 45], "color": "#eeeeee"},
                        {"range": [45, 55], "color": "#dddddd"},
                        {"range": [55, 70], "color": "#cccccc"},
                        {"range": [70, 100], "color": "#bbbbbb"},
                    ],
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.markdown("#### Score audit trail")
        st.dataframe(scorecard, use_container_width=True, hide_index=True)

    st.markdown("#### Rule-based desk commentary")
    st.markdown(generate_rule_based_commentary(summary, scorecard, moil_corr, anomalies))

    st.markdown("#### Signal contribution by pillar")
    if not scorecard.empty:
        pillar = scorecard.groupby("pillar", as_index=False)["points"].sum().sort_values("points", ascending=False)
        bar = px.bar(pillar, x="pillar", y="points", title="Net points by signal pillar")
        st.plotly_chart(bar, use_container_width=True)

_t3 = _tab("Manual macro tracker")
if _t3 is not None:
    with _t3:
        st.subheader("Manual physical macro tracker")
        st.write("Edit physical-market scores here. Use -2 to +2. These remain session values until downloaded and saved back to data/manual_macro_template.csv.")
        uploaded = st.file_uploader("Upload manual macro CSV", type=["csv"])
        if uploaded is not None:
            st.session_state.manual_macro = normalize_manual_macro(pd.read_csv(uploaded))
            st.success("Uploaded manual macro table applied to current session.")

        evidence_upload = st.file_uploader("Upload verified evidence CSV/JSON", type=["csv", "json"], key="verified_evidence_upload")
        if evidence_upload is not None:
            try:
                suffix = ".json" if evidence_upload.name.lower().endswith(".json") else ".csv"
                tmp_path = Path("data/cache/_uploaded_verified_evidence" + suffix)
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(evidence_upload.getvalue())
                evidence_df = load_evidence_from_file(tmp_path)
                evidence_df, warnings = validate_evidence_dataframe(evidence_df)
                st.session_state.evidence_validation_warnings = warnings
                st.session_state.manual_macro = normalize_manual_macro(evidence_to_macro_rows(evidence_df))
                st.session_state.groq_scores = None
                st.session_state.groq_scores_applied = None
                st.success(f"Verified evidence upload converted into {len(st.session_state.manual_macro)} macro row(s).")
                if warnings:
                    with st.expander("Evidence validation warnings", expanded=True):
                        for warning in warnings:
                            st.write(f"- {warning}")
            except Exception as exc:
                st.error(f"Could not parse verified evidence upload: {exc}")

        edited_manual = st.data_editor(
            st.session_state.manual_macro,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            height=330,
            column_config={
                "score": st.column_config.NumberColumn("score", min_value=-2.0, max_value=2.0, step=0.5),
                "status": st.column_config.SelectboxColumn("status", options=["bullish", "constructive", "mildly constructive", "neutral", "watch", "bearish", "stress"]),
                "commentary": st.column_config.TextColumn("commentary", width="large"),
            },
        )
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Apply manual macro changes"):
                st.session_state.manual_macro = normalize_manual_macro(edited_manual)
                st.session_state.groq_scores_applied = None
                st.success("Manual macro scores applied. Recomputing regime.")
                st.rerun()
        with b2:
            buffer = io.StringIO()
            normalize_manual_macro(edited_manual).to_csv(buffer, index=False)
            st.download_button("Download updated manual macro CSV", buffer.getvalue(), file_name="manual_macro_template.csv", mime="text/csv")

        st.markdown("#### Scoring scale")
        st.dataframe(
            pd.DataFrame(
                [
                    {"score": 2.0, "meaning": "strongly bullish"},
                    {"score": 1.0, "meaning": "bullish / constructive"},
                    {"score": 0.5, "meaning": "mildly constructive"},
                    {"score": 0.0, "meaning": "neutral / mixed / unavailable"},
                    {"score": -0.5, "meaning": "mildly negative"},
                    {"score": -1.0, "meaning": "bearish"},
                    {"score": -2.0, "meaning": "severe stress"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

_t4 = _tab("Groq AI desk")
if _t4 is not None:
    with _t4:
        st.subheader("Groq / Llama 3.3 AI desk")
        config = get_groq_config()
        c1, c2, c3, c4, c5 = st.columns(5)
        auto_source_ready = (
            is_any_vendor_feed_configured()
            or is_serper_configured()
            or is_cached_macro_feed_available()
            or bool(st.secrets.get("HEADLINEFEED_TOKEN", ""))
        )
        c1.metric("Groq configured", "Yes" if is_groq_configured() else "No")
        c2.metric("Auto source pull", "Yes" if auto_source_ready else "No")
        c3.metric("Model", config["model"])
        c4.metric("Key name", config["key_name"])
        c5.metric("Project", config["project"])
        st.caption("API keys are never displayed. Groq scores supplied facts; source-specific vendor feeds are used first, then SERPER web-search fallback, then manual rows.")

        if not is_groq_configured():
            st.warning("Groq is not configured. Manual macro scoring and rule-based commentary remain active.")
        if not auto_source_ready:
            st.info("Automatic physical macro pull requires scheduled backend cache output, source-specific vendor feed URLs/API keys, or SERPER_API_KEY. Without them, Groq scores the current manual/news tracker rows only.")
        else:
            st.caption(
                "Auto source priority: vendor endpoints → Serper (SERPER_API_KEY) → HeadlineFeed (HEADLINEFEED_TOKEN) → manual. Results are cached ~1h to avoid burning quotas."
            )

        st.markdown("#### Source readiness v2")
        readiness_df = source_provider_status()
        st.dataframe(readiness_df, use_container_width=True, hide_index=True)
        if not readiness_df.empty:
            blocked = readiness_df[readiness_df.get("scoring_allowed", False) == False] if "scoring_allowed" in readiness_df.columns else pd.DataFrame()
            if not blocked.empty:
                st.caption("Rows below the confidence threshold are commentary-only and do not move the regime score.")

        st.markdown("#### Automatic physical macro feed")
        st.write("Pull fresh source-aware data for SiMn prices, India steel output, China exports and China power/industrial stress, then let Groq convert the supplied facts/snippets into scores. Paid/vendor endpoints are used first; targeted web search is the fallback.")
        auto_col1, auto_col2 = st.columns([1, 2])
        with auto_col1:
            if st.button("Pull auto macro feed", disabled=not auto_source_ready):
                with st.spinner("Pulling fresh macro data from configured sources/search fallback..."):
                    auto_macro, diagnostics = fetch_source_aware_macro_feed()
                    st.session_state.auto_macro_diagnostics = diagnostics
                    if not auto_macro.empty:
                        st.session_state.manual_macro = normalize_manual_macro(auto_macro)
                        st.session_state.groq_scores = None
                        st.session_state.groq_scores_applied = None
                        st.success(f"Auto macro feed applied to current session: {len(auto_macro)} rows.")
                        st.rerun()
                    else:
                        st.warning("No automatic macro rows were returned. Check SERPER_API_KEY and diagnostics.")
        with auto_col2:
            if st.session_state.auto_macro_diagnostics:
                with st.expander("Auto macro feed diagnostics", expanded=False):
                    for item in st.session_state.auto_macro_diagnostics:
                        st.write(f"- {item}")

        run_col, apply_col, clear_col = st.columns(3)
        with run_col:
            if st.button("Run Groq macro scoring", disabled=not (enable_groq_scoring and is_groq_configured())):
                with st.spinner("Groq/Llama 3.3 is scoring macro indicators..."):
                    source_readiness_records = source_provider_status().to_dict(orient="records")
                    regime_context = {
                        "label": summary.label,
                        "normalized_score": summary.normalized_score,
                        "confidence_score": summary.confidence_score,
                        "data_quality_score": summary.data_quality_score,
                        "source_readiness": source_readiness_records,
                        "confidence_policy": {
                            ">=80": "full score impact",
                            "60-79": "70% score impact",
                            "40-59": "40% score impact",
                            "<40": "commentary only; no regime impact",
                        },
                    }
                    result = score_macro_with_groq(st.session_state.manual_macro, snapshot, regime_context, st.session_state.news_tracker)
                    st.session_state.groq_scores = result
                    if result.get("ok"):
                        st.success("Groq macro scoring completed.")
                    else:
                        st.error(result.get("error", "Groq scoring failed."))
        with apply_col:
            if st.button("Apply Groq scores to regime", disabled=not (st.session_state.groq_scores and st.session_state.groq_scores.get("macro_scores"))):
                st.session_state.groq_scores_applied = st.session_state.groq_scores
        st.markdown("#### Institutional features (live NSE preferred, sample fallback)")
        if st.button("Build institutional features"):
            with st.spinner("Building institutional features (live NSE if available, else sample/manual)..."):
                feats, diag = build_institutional_features(prefer_live=True)
                st.session_state.offline_inst_features = feats
                st.session_state.offline_inst_diag = diag
            st.success("Institutional features built and saved to reports/moil_institutional_features.csv")
        if st.session_state.get("offline_inst_diag"):
            st.caption("; ".join(st.session_state.offline_inst_diag))
        if st.session_state.get("offline_inst_features") is not None:
            feats = st.session_state.offline_inst_features
            if feats is not None and not feats.empty:
                show_cols = [c for c in [
                    "date", "bulk_net_qty", "bulk_net_value", "block_net_qty", "block_net_value",
                    "moil_bulk_threshold_hit", "moil_block_deal_flag", "mf_net_shares", "named_1pct_holder_count",
                    "sast_event_flag", "pit_event_flag"
                ] if c in feats.columns]
                st.dataframe(feats[show_cols], use_container_width=True, hide_index=True)
                csv_buffer = io.StringIO()
                feats.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download institutional features CSV",
                    csv_buffer.getvalue(),
                    file_name="moil_institutional_features.csv",
                    mime="text/csv",
                )
            else:
                st.info("Institutional features empty; ensure live NSE available or sample files exist under data/sample/.")
        if st.session_state.groq_scores:
            result = st.session_state.groq_scores
            if result.get("macro_scores"):
                st.markdown("#### Groq macro scores")
                st.dataframe(pd.DataFrame(result["macro_scores"]), use_container_width=True, hide_index=True)
            st.metric("Overall macro score", f"{float(result.get('overall_manual_macro_score', 0)):+.2f}")
            st.markdown("#### Regime commentary")
            st.write(result.get("regime_commentary", ""))
            if result.get("watch_items"):
                st.markdown("#### Watch items")
                for item in result.get("watch_items", []):
                    st.write(f"- {item}")
            if result.get("risks"):
                st.markdown("#### Risks")
                for item in result.get("risks", []):
                    st.write(f"- {item}")

        st.markdown("#### Prompt for manual review")
        copy_prompt = build_groq_prompt_for_copy(summary, scorecard, st.session_state.manual_macro, snapshot, moil_corr, anomalies)
        st.text_area("Copy/paste prompt", copy_prompt, height=300)

_t5 = _tab("Institutional sync")
with _t5:
    st.subheader("Institutional & exchange sync")
    st.write("Track qualitative news, fund actions and exchange activity. NSE requests can be blocked by cloud environments, so failures are handled safely.")

    inst_cache = load_latest_institutional_cache()
    inst_quality = institutional_quality_summary_v2(inst_cache)
    inst_evidence = institutional_cache_to_evidence_rows(inst_cache)
    inst_scorecard = institutional_cache_to_scorecard(inst_cache)
    inst_readiness = institutional_cache_to_source_readiness(inst_cache)

    st.markdown("#### Institutional Quality v2")
    iq1, iq2, iq3, iq4 = st.columns(4)
    iq1.metric("Evidence rows", inst_quality.get("evidence_rows", 0))
    iq2.metric("Scoring rows", inst_quality.get("scoring_rows", 0))
    iq3.metric("Avg confidence", f"{float(inst_quality.get('average_confidence', 0)):.0f}/100")
    iq4.metric("Net smart-money impact", f"{float(inst_quality.get('net_effective_impact', 0)):+.2f}")
    st.caption(f"Institutional tone: {inst_quality.get('institutional_tone', 'n/a')}")

    with st.expander("Institutional source readiness", expanded=False):
        if inst_readiness is None or inst_readiness.empty:
            st.info("No institutional source-readiness rows yet. Run python institutional_tracker.py or use manual CSV upload.")
        else:
            st.dataframe(inst_readiness, use_container_width=True, hide_index=True)

    with st.expander("Confidence-adjusted smart-money scorecard", expanded=True):
        if inst_scorecard is None or inst_scorecard.empty:
            st.info("No confidence-adjusted smart-money signals yet. No material institution-level event has passed quality gates.")
        else:
            st.dataframe(inst_scorecard, use_container_width=True, hide_index=True)

    with st.expander("Institutional evidence viewer", expanded=False):
        if inst_evidence is None or inst_evidence.empty:
            st.info("No institutional evidence rows found in data/cache/institutional_activity_latest.json.")
        else:
            show_cols = [c for c in ["evidence_type", "source_name", "event_date", "institution", "transaction_type", "quantity", "price", "pct_equity", "holder_category", "holding_pct", "delta_pct_points", "data_confidence", "confidence_weight", "effective_impact", "scoring_allowed", "confidence_notes"] if c in inst_evidence.columns]
            st.dataframe(inst_evidence[show_cols], use_container_width=True, hide_index=True)

    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("NSE Sync: Bulk/Block Deals"):
            with st.spinner("Fetching NSE archive deal data..."):
                deals, diagnostics = fetch_nse_deals_auto("MOIL")
                st.session_state.nse_sync_diagnostics = diagnostics
                if deals is not None and not deals.empty:
                    updates = nse_deals_to_tracker_updates(deals)
                    st.session_state.news_tracker = pd.concat([updates, st.session_state.news_tracker], ignore_index=True)
                    st.success(f"Added {len(updates)} NSE tracker rows.")
                else:
                    loaded = any("OK" in str(x) or "cache hit" in str(x) for x in diagnostics)
                    if loaded:
                        st.info("NSE archive feed loaded, but no MOIL bulk/block deal row was present in the latest archive.")
                    else:
                        st.warning("NSE archive endpoints were blocked or unavailable in this environment. Use the CSV upload fallback below.")
                if diagnostics:
                    with st.expander("NSE sync diagnostics", expanded=True):
                        for item in diagnostics:
                            st.write(f"- {item}")
    with n2:
        if st.button("Save news tracker"):
            DATA_DIR.mkdir(exist_ok=True)
            st.session_state.news_tracker.to_csv(NEWS_TRACKER_PATH, index=False)
            st.success(f"Saved to {NEWS_TRACKER_PATH}")
    with n3:
        buffer = io.StringIO()
        st.session_state.news_tracker.to_csv(buffer, index=False)
        st.download_button("Download tracker CSV", buffer.getvalue(), file_name="news_tracker.csv", mime="text/csv")

    st.markdown("#### NSE CSV fallback")
    st.write("If Codespaces blocks NSE, download bulk/block CSV from NSE in your browser and upload it here. The app will filter symbol MOIL and add institutional tracker rows.")
    nse_upload = st.file_uploader("Upload NSE bulk/block CSV", type=["csv"], key="nse_csv_upload")
    if nse_upload is not None:
        try:
            raw_deals = pd.read_csv(nse_upload)
            from auto_feeds import _standardize_nse_deal_columns
            normalized_deals = _standardize_nse_deal_columns(raw_deals, "uploaded")
            moil_deals = normalized_deals[normalized_deals["symbol"].astype(str).str.upper() == "MOIL"] if not normalized_deals.empty else pd.DataFrame()
            if not moil_deals.empty:
                updates = nse_deals_to_tracker_updates(moil_deals)
                st.session_state.news_tracker = pd.concat([updates, st.session_state.news_tracker], ignore_index=True)
                st.success(f"Uploaded CSV added {len(updates)} MOIL deal row(s) to tracker.")
                st.rerun()
            else:
                st.info("Uploaded NSE CSV parsed, but no MOIL rows were found.")
        except Exception as exc:
            st.error(f"Could not parse uploaded NSE CSV: {exc}")

    edited_news = st.data_editor(
        st.session_state.news_tracker,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=360,
        column_config={
            "impact": st.column_config.NumberColumn("impact", min_value=-2.0, max_value=2.0, step=0.5),
            "confidence": st.column_config.NumberColumn("confidence", min_value=0.0, max_value=1.0, step=0.05),
            "sentiment": st.column_config.SelectboxColumn("sentiment", options=["Bullish", "Neutral", "Bearish"]),
            "details": st.column_config.TextColumn("details", width="large"),
        },
    )
    if st.button("Apply tracker edits"):
        st.session_state.news_tracker = edited_news
        st.rerun()

_t6 = _tab("Alerts & commentary")
with _t6:
    st.subheader("Alerts & commentary")
    st.markdown("#### Rule-based commentary")
    rule_commentary = generate_rule_based_commentary(summary, scorecard, moil_corr, anomalies)
    st.markdown(rule_commentary)

    if enable_groq_commentary and is_groq_configured():
        if st.button("Generate Groq commentary"):
            with st.spinner("Generating institutional commentary with Groq/Llama 3.3..."):
                context = {
                    "regime": summary.__dict__,
                    "scorecard": scorecard.to_dict(orient="records") if not scorecard.empty else [],
                    "manual_macro": st.session_state.manual_macro.to_dict(orient="records"),
                    "groq_macro_scores": st.session_state.groq_scores_applied or st.session_state.groq_scores or {},
                    "market_snapshot": snapshot.to_dict(orient="records") if not snapshot.empty else [],
                    "moil_correlation": moil_corr.to_dict(orient="records") if not moil_corr.empty else [],
                    "anomalies": anomalies.to_dict(orient="records") if not anomalies.empty else [],
                    "technical_levels": technical_levels,
                    "news_tracker": st.session_state.news_tracker.to_dict(orient="records") if not st.session_state.news_tracker.empty else [],
                    "source_readiness": source_provider_status().to_dict(orient="records"),
                    "institutional_quality": institutional_quality_summary_v2(load_latest_institutional_cache()),
                    "institutional_smart_money_scorecard": institutional_cache_to_scorecard(load_latest_institutional_cache()).to_dict(orient="records"),
                }
                st.session_state.groq_commentary = generate_groq_commentary(context)

    if st.session_state.groq_commentary:
        result = st.session_state.groq_commentary
        if result.get("ok"):
            st.markdown("#### Groq Telegram alert")
            st.text_area("AI alert", result.get("telegram_alert", ""), height=150)
            st.markdown("#### Groq institutional commentary")
            st.write(result.get("institutional_commentary", ""))
            if result.get("watchlist_triggers"):
                st.markdown("#### AI watchlist triggers")
                for item in result.get("watchlist_triggers", []):
                    st.write(f"- {item}")
        else:
            st.warning(result.get("error", "Groq commentary unavailable."))

    ai_alert = ""
    if st.session_state.groq_commentary and st.session_state.groq_commentary.get("ok"):
        ai_alert = st.session_state.groq_commentary.get("telegram_alert", "")

    alert_text = build_telegram_alert_text(
        summary,
        scorecard,
        anomalies,
        prices,
        st.session_state.manual_macro,
        st.session_state.news_tracker,
        technical_levels,
        ai_commentary=ai_alert,
    )
    st.markdown("#### Telegram snapshot")
    alert_text = st.text_area("Alert preview", alert_text, height=240)
    send_disabled = not enable_telegram
    if st.button("Send Telegram alert", disabled=send_disabled):
        ok, detail = send_telegram_alert(alert_text)
        if ok:
            st.success(detail)
        else:
            st.error(detail)
    if not enable_telegram:
        st.info("Enable Telegram alerts in the sidebar after setting TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    st.download_button("Download scorecard CSV", scorecard.to_csv(index=False), file_name="moil_regime_scorecard.csv", mime="text/csv")

_t7 = _tab("Validation")
if _t7 is not None:
    with _t7:
        st.subheader("Validation & real-world testing")
        st.write("This tab helps address model-risk feedback. It is not a trading-system backtest; it validates whether a simple transparent regime condition historically aligned with better MOIL forward returns.")
        if validation.empty:
            st.info("Not enough MOIL/NIFTY Metal history for validation.")
        else:
            display_validation = validation.copy()
            for col in ["avg_return_when_signal", "hit_rate_when_signal", "avg_return_without_signal", "excess_avg_return"]:
                if col in display_validation.columns:
                    display_validation[col] = display_validation[col].apply(pct_fmt)
            st.dataframe(display_validation, use_container_width=True, hide_index=True)
            chart_df = validation.melt(id_vars=["forward_days"], value_vars=["avg_return_when_signal", "avg_return_without_signal"], var_name="bucket", value_name="return")
            st.plotly_chart(px.bar(chart_df, x="forward_days", y="return", color="bucket", barmode="group", title="Forward returns by simple constructive regime"), use_container_width=True)
        st.markdown("#### Suggested validation log")
        st.write("Track hit rate, false positives, average 5D/10D/20D forward return, maximum drawdown, data gaps and factor changes monthly.")

_t8 = _tab("Data quality")
if _t8 is not None:
    with _t8:
        st.subheader("Data Quality Command Center")
        # On-demand macro cache refresh
        if st.button("Refresh macro cache now"):
            with st.spinner("Refreshing macro cache..."):
                rows, diag = fetch_source_aware_macro_feed(max_results_per_query=3)
                cache_dir = Path("data/cache")
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / "latest_macro_data.json").write_text(rows.to_json(orient="records"), encoding="utf-8")
            st.success("Macro cache refreshed to data/cache/latest_macro_data.json")
            if diag:
                st.caption("; ".join(diag))
            st.rerun()

        macro_cache = load_latest_macro_cache()
        evidence_df = macro_cache_to_evidence_rows(macro_cache)
        readiness_v2 = macro_cache_to_source_readiness(macro_cache)
        inst_cache = load_latest_institutional_cache()
        inst_summary = institutional_cache_summary(inst_cache)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Evidence rows", int(len(evidence_df)) if evidence_df is not None else 0)
        if readiness_v2 is not None and not readiness_v2.empty:
            scoring_count = int(readiness_v2["scoring_allowed"].sum()) if "scoring_allowed" in readiness_v2.columns else 0
            avg_conf = float(pd.to_numeric(readiness_v2.get("confidence", pd.Series(dtype=float)), errors="coerce").fillna(0).mean())
        else:
            scoring_count = 0
            avg_conf = 0.0
        q2.metric("Scoring-ready indicators", scoring_count)
        q3.metric("Avg source confidence", f"{avg_conf:.0f}/100")
        inst_quality_v2 = institutional_quality_summary_v2(inst_cache)
        q4.metric("Institutional confidence", f"{float(inst_quality_v2.get('average_confidence', 0)):.0f}/100")

        st.markdown("#### Source readiness v2")
        st.dataframe(readiness_v2, use_container_width=True, hide_index=True)

        st.markdown("#### Evidence viewer")
        if evidence_df is None or evidence_df.empty:
            st.info("No cached evidence rows found. Run python -m scrapers.run_daily_scrape.")
        else:
            show_cols = [c for c in ["indicator", "source_name", "source_tier", "exact_data", "period", "value", "unit", "data_confidence", "confidence_weight", "scoring_allowed", "confidence_label", "confidence_notes"] if c in evidence_df.columns]
            st.dataframe(evidence_df[show_cols].sort_values("data_confidence", ascending=False), use_container_width=True, hide_index=True)
            with st.expander("Raw evidence snippets", expanded=False):
                selected_indicator = st.selectbox("Evidence indicator", sorted(evidence_df["indicator"].dropna().unique().tolist())) if "indicator" in evidence_df.columns else None
                if selected_indicator:
                    subset = evidence_df[evidence_df["indicator"] == selected_indicator]
                    for _, row in subset.head(10).iterrows():
                        st.markdown(f"**{row.get('source_name','source')} — confidence {row.get('data_confidence',0)}/100**")
                        st.code(str(row.get("raw_text", ""))[:1200])

        st.markdown("#### Institutional Quality v2")
        inst_evidence_v2 = institutional_cache_to_evidence_rows(inst_cache)
        inst_readiness_v2 = institutional_cache_to_source_readiness(inst_cache)
        inst_scorecard_v2 = institutional_cache_to_scorecard(inst_cache)
        c_inst1, c_inst2, c_inst3 = st.columns(3)
        c_inst1.metric("Institutional evidence rows", int(len(inst_evidence_v2)) if inst_evidence_v2 is not None else 0)
        c_inst2.metric("Institutional scoring rows", int(inst_evidence_v2["scoring_allowed"].sum()) if inst_evidence_v2 is not None and not inst_evidence_v2.empty and "scoring_allowed" in inst_evidence_v2.columns else 0)
        c_inst3.metric("Net institutional impact", f"{float(inst_quality_v2.get('net_effective_impact', 0)):+.2f}")
        with st.expander("Institutional source readiness v2", expanded=False):
            st.dataframe(inst_readiness_v2, use_container_width=True, hide_index=True)
        with st.expander("Institutional evidence and scorecard", expanded=False):
            if inst_scorecard_v2 is not None and not inst_scorecard_v2.empty:
                st.markdown("**Confidence-adjusted smart-money scorecard**")
                st.dataframe(inst_scorecard_v2, use_container_width=True, hide_index=True)
            if inst_evidence_v2 is not None and not inst_evidence_v2.empty:
                st.markdown("**Institutional evidence rows**")
                show_cols = [c for c in ["evidence_type", "source_name", "event_date", "institution", "transaction_type", "quantity", "price", "holder_category", "holding_pct", "delta_pct_points", "data_confidence", "confidence_weight", "effective_impact", "scoring_allowed"] if c in inst_evidence_v2.columns]
                st.dataframe(inst_evidence_v2[show_cols], use_container_width=True, hide_index=True)

        st.markdown("#### Feed quality")
        st.dataframe(data_quality_table, use_container_width=True, hide_index=True)
        st.markdown("#### Ticker map and rows")
        q_rows = []
        for asset, ticker in edited_tickers.items():
            df = data.get(asset, pd.DataFrame())
            q_rows.append(
                {
                    "asset": asset,
                    "ticker": ticker,
                    "rows": int(len(df)) if df is not None else 0,
                    "last_date": df.index.max().strftime("%Y-%m-%d") if df is not None and not df.empty else "No data",
                    "status": "OK" if df is not None and not df.empty else "Missing",
                }
            )
        st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)
        if errors:
            with st.expander("Feed warnings", expanded=True):
                for asset, err in errors.items():
                    st.write(f"**{asset}:** {err}")
        st.markdown("#### Raw snapshot")
        st.dataframe(format_snapshot(snapshot), use_container_width=True, hide_index=True)
        if not prices.empty:
            st.download_button("Download price panel CSV", prices.to_csv(index=True), file_name="moil_price_panel.csv", mime="text/csv")
            with st.expander("Price panel preview"):
                st.dataframe(prices.tail(100), use_container_width=True)

_t9 = _tab("📈 Intraday")
if _t9 is not None:
    with _t9:
        st.subheader("📈 Intraday — MOIL")
        _intra_sub, _mtf_sub = st.tabs(["📊 Intraday ORB", "🧠 MTF Engine"])

    with _t9:
        with _intra_sub:
            import datetime as _dt_mod

            # ── Section 1: 5m candle chart ────────────────────────────────────
            with st.expander("📈 Intraday 5-min chart", expanded=st.session_state.get("show_5m", True)):
                _5m_ctrl1, _5m_ctrl2, _5m_ctrl3 = st.columns([2, 1, 1])
                _5m_date_sel = _5m_ctrl1.date_input(
                    "Select date (leave today for live)",
                    value=_dt_mod.date.today(),
                    max_value=_dt_mod.date.today(),
                    key="intraday_5m_date",
                )
                _5m_fetch_btn = _5m_ctrl2.button("🔄 Fetch 5m data", key="fetch_5m", use_container_width=True)
                _5m_live_only = _5m_ctrl3.checkbox("Today only (live)", value=False, key="5m_live_only")

                # Fetch logic — TradingView → Dhan → Zerodha → yfinance
                _5m_cache_key = f"intraday_5m_{_5m_date_sel}"
                if _5m_fetch_btn or _5m_cache_key not in st.session_state:
                    with st.spinner("Fetching 5m OHLCV (TradingView → Dhan → Zerodha → yfinance)…"):
                        _5m_period = "1d" if (_5m_live_only or _5m_date_sel == _dt_mod.date.today()) else None
                        _5m_start  = _5m_date_sel.strftime("%Y-%m-%d")
                        _5m_end    = (_5m_date_sel + _dt_mod.timedelta(days=1)).strftime("%Y-%m-%d")
                        _5m_df, _5m_src, _5m_err = fetch_ohlcv(
                            "MOIL.NS", interval="5m",
                            from_date=_5m_start, to_date=_5m_end,
                            period=_5m_period,
                        )
                        st.session_state[_5m_cache_key] = _5m_df
                        st.session_state[f"{_5m_cache_key}_src"] = _5m_src
                        if _5m_err:
                            st.warning(f"5m feed: {_5m_err}")
                        else:
                            st.caption(f"5m data source: {source_badge(_5m_src)} — {len(_5m_df)} bars")

                # Last-resort fallback
                _idf_src = st.session_state.get(_5m_cache_key)
                if _idf_src is None or _idf_src.empty:
                    _intraday_data, _intraday_err = cached_intraday_data(
                        tuple(DEFAULT_MARKET_TICKERS.items()), demo=use_demo_data)
                    if _intraday_data.get("MOIL") is not None and not _intraday_data["MOIL"].empty:
                        _idf_src = _intraday_data["MOIL"]

                if _idf_src is not None and not _idf_src.empty:
                    _idf = localize_to_ist(_idf_src) if hasattr(_idf_src.index, 'tz') or hasattr(_idf_src.index, 'tzinfo') else _idf_src
                    _idf.columns = [c.lower() if isinstance(c, str) else c for c in _idf.columns]
                    _open_col  = next((c for c in _idf.columns if "open"  in c.lower()), None)
                    _high_col  = next((c for c in _idf.columns if "high"  in c.lower()), None)
                    _low_col   = next((c for c in _idf.columns if "low"   in c.lower()), None)
                    _close_col = next((c for c in _idf.columns if "close" in c.lower()), None)
                    _vol_col   = next((c for c in _idf.columns if "volume" in c.lower()), None)
                    if all([_open_col, _high_col, _low_col, _close_col]):
                        _ifig = go.Figure()
                        _ifig.add_trace(go.Candlestick(
                            x=_idf.index,
                            open=_idf[_open_col], high=_idf[_high_col],
                            low=_idf[_low_col],   close=_idf[_close_col],
                            name="MOIL 5m",
                            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                            increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
                        ))
                        if _vol_col:
                            _ifig.add_trace(go.Bar(
                                x=_idf.index, y=_idf[_vol_col],
                                name="Volume", marker_color="#2962ff", opacity=0.35, yaxis="y2",
                            ))
                        _ifig.update_layout(
                            template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
                            xaxis=dict(gridcolor="#2a2e39"),
                            yaxis=dict(gridcolor="#2a2e39", title="Price"),
                            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
                            xaxis_rangeslider_visible=False, height=400, hovermode="x unified",
                            title=f"MOIL 5m — {_5m_date_sel.strftime('%d %b %Y')} ({len(_idf)} bars)",
                            legend=dict(bgcolor="#1e2130", bordercolor="#2a2e39"),
                        )
                        st.plotly_chart(_ifig, use_container_width=True)
                        _lr = _idf.iloc[-1]; _fr = _idf.iloc[0]
                        _fo  = float(_fr[_open_col])
                        _ret = (float(_lr[_close_col]) / _fo - 1) * 100 if _fo != 0 else 0.0
                        _sh  = float(_idf[_high_col].max())
                        _sl  = float(_idf[_low_col].min())
                        _pos = (float(_lr[_close_col]) - _sl) / (_sh - _sl) if _sh > _sl else None
                        st.caption(
                            f"Change: {_ret:+.2f}% | Last: ₹{float(_lr[_close_col]):.2f} | "
                            f"H/L: ₹{_sh:.2f}/₹{_sl:.2f}"
                            + (f" | Range pos: {_pos*100:.1f}%" if _pos is not None else "")
                            + f" | Bars: {len(_idf)} | Last: {_idf.index[-1]}"
                        )
                        _ibuf = io.StringIO(); _idf.to_csv(_ibuf)
                        st.download_button("⬇ Download 5m OHLCV", _ibuf.getvalue(),
                                           file_name=f"moil_5m_{_5m_date_sel}.csv", mime="text/csv",
                                           key="dl_5m")
                    else:
                        st.warning("5m data loaded but columns not recognised.")
                else:
                    st.info("Click **Fetch 5m data** above to load the chart.")

            st.markdown("---")

            # ── Section 2: 1-min candle chart ────────────────────────────────
            with st.expander("🔍 1-min drill-down chart (visual only — not used by engine)", expanded=False):
                st.caption("Fetches today's 1m data automatically. Select any date or upload historical 1m CSV/Parquet.")
                _1m_ctrl1, _1m_ctrl2, _1m_ctrl3 = st.columns([2, 1, 1])
                _1m_date_sel = _1m_ctrl1.date_input(
                    "Select date",
                    value=_dt_mod.date.today(),
                    max_value=_dt_mod.date.today(),
                    key="intraday_1m_date_sel",
                )
                _1m_fetch_btn = _1m_ctrl2.button("🔄 Fetch 1m data", key="fetch_1m", use_container_width=True)

                # Auto-fetch: TradingView → Dhan → Zerodha → yfinance
                _1m_yf_key = f"intraday_1m_yf_{_1m_date_sel}"
                if _1m_fetch_btn or _1m_yf_key not in st.session_state:
                    with st.spinner("Fetching 1m data (TradingView → Dhan → Zerodha → yfinance)…"):
                        _1m_period = "1d" if _1m_date_sel == _dt_mod.date.today() else None
                        _1m_start  = _1m_date_sel.strftime("%Y-%m-%d")
                        _1m_end    = (_1m_date_sel + _dt_mod.timedelta(days=1)).strftime("%Y-%m-%d")
                        _1m_fetched, _1m_src, _1m_err = fetch_ohlcv(
                            "MOIL.NS", interval="1m",
                            from_date=_1m_start, to_date=_1m_end,
                            period=_1m_period,
                        )
                        if not _1m_fetched.empty:
                            _1m_fetched = _1m_fetched.reset_index()
                            _dt_found = next((c for c in _1m_fetched.columns if "date" in c.lower() or "time" in c.lower()), None)
                            if _dt_found and _dt_found != "datetime":
                                _1m_fetched = _1m_fetched.rename(columns={_dt_found: "datetime"})
                            _1m_fetched["datetime"] = pd.to_datetime(_1m_fetched["datetime"])
                            st.session_state[_1m_yf_key] = _1m_fetched
                            st.caption(f"1m data source: {source_badge(_1m_src)} — {len(_1m_fetched)} bars")
                        else:
                            st.session_state[_1m_yf_key] = pd.DataFrame()
                            _1m_hint = "yfinance only provides 7 days of 1m history" if "yfinance" in str(_1m_err) else ""
                            st.info(f"No 1m data for {_1m_date_sel}. {_1m_hint} Upload a CSV below.")
                            if _1m_err:
                                st.caption(f"Feed errors: {_1m_err}")

                # Upload fallback
                _1m_upload = _1m_ctrl3.file_uploader(
                    "Or upload CSV/Parquet",
                    type=["csv", "parquet", "pq"],
                    accept_multiple_files=False,
                    key="intraday_1m_upload",
                    label_visibility="collapsed",
                )
                if _1m_upload is not None:
                    try:
                        if _1m_upload.name.lower().endswith((".parquet", ".pq")):
                            _1m_df_raw = pd.read_parquet(_1m_upload)
                        else:
                            _1m_df_raw = pd.read_csv(_1m_upload)
                        _1m_df_raw.columns = [c.lower().strip() for c in _1m_df_raw.columns]
                        _dt_col = next((c for c in _1m_df_raw.columns if "date" in c or "time" in c), _1m_df_raw.columns[0])
                        _1m_df_raw = _1m_df_raw.rename(columns={_dt_col: "datetime"})
                        _1m_df_raw["datetime"] = pd.to_datetime(_1m_df_raw["datetime"], errors="coerce")
                        _1m_df_raw = _1m_df_raw.dropna(subset=["datetime"]).sort_values("datetime")
                        st.session_state["intraday_1m_df"] = _1m_df_raw
                        st.success(f"Loaded {len(_1m_df_raw):,} rows from upload.")
                    except Exception as _e:
                        st.error(f"Could not parse 1-min file: {_e}")

                # Merge: live fetch takes priority, upload as fallback
                _1m_yf_data  = st.session_state.get(_1m_yf_key, pd.DataFrame())
                _1m_up_data  = st.session_state.get("intraday_1m_df", pd.DataFrame())
                if _1m_yf_data is not None and not _1m_yf_data.empty:
                    _1m_df = _1m_yf_data.copy()
                elif _1m_up_data is not None and not _1m_up_data.empty:
                    _1m_df = _1m_up_data[_1m_up_data["datetime"].dt.date == _1m_date_sel].copy()
                else:
                    _1m_df = pd.DataFrame()

                if _1m_df is not None and not _1m_df.empty:
                    _day_1m = _1m_df.copy()
                    _day_1m.columns = [c.lower().strip() for c in _day_1m.columns]
                    _open_1m  = next((c for c in _day_1m.columns if "open"   in c), None)
                    _high_1m  = next((c for c in _day_1m.columns if "high"   in c), None)
                    _low_1m   = next((c for c in _day_1m.columns if "low"    in c), None)
                    _close_1m = next((c for c in _day_1m.columns if "close"  in c), None)
                    _vol_1m   = next((c for c in _day_1m.columns if "volume" in c), None)
                    _dt_1m    = next((c for c in _day_1m.columns if "date"   in c or "time" in c), None)
                    if all([_open_1m, _high_1m, _low_1m, _close_1m, _dt_1m]):
                        _fig1m = go.Figure()
                        _fig1m.add_trace(go.Candlestick(
                            x=_day_1m[_dt_1m],
                            open=_day_1m[_open_1m], high=_day_1m[_high_1m],
                            low=_day_1m[_low_1m],   close=_day_1m[_close_1m],
                            name="MOIL 1m",
                            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                            increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
                        ))
                        if _vol_1m:
                            _fig1m.add_trace(go.Bar(
                                x=_day_1m[_dt_1m], y=_day_1m[_vol_1m],
                                name="Volume", marker_color="#2962ff", opacity=0.4, yaxis="y2",
                            ))
                        _fig1m.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#131722", plot_bgcolor="#131722",
                            xaxis=dict(gridcolor="#2a2e39", showgrid=True),
                            yaxis=dict(gridcolor="#2a2e39", showgrid=True, title="Price ₹"),
                            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
                            xaxis_rangeslider_visible=False,
                            height=460, hovermode="x unified",
                            title=f"MOIL 1-min — {_1m_date_sel.strftime('%d %b %Y')} ({len(_day_1m)} bars)",
                            legend=dict(bgcolor="#1e2130", bordercolor="#2a2e39", font=dict(color="#d1d4dc")),
                        )
                        st.plotly_chart(_fig1m, use_container_width=True)
                        _d1_open  = float(_day_1m[_open_1m].iloc[0])
                        _d1_close = float(_day_1m[_close_1m].iloc[-1])
                        _d1_ret   = (_d1_close / _d1_open - 1) * 100 if _d1_open != 0 else 0.0
                        c1m_1, c1m_2, c1m_3, c1m_4 = st.columns(4)
                        c1m_1.metric("Open",     f"₹{_d1_open:.2f}")
                        c1m_2.metric("Close",    f"₹{_d1_close:.2f}", delta=f"{_d1_ret:+.2f}%")
                        c1m_3.metric("Day High", f"₹{float(_day_1m[_high_1m].max()):.2f}")
                        c1m_4.metric("Day Low",  f"₹{float(_day_1m[_low_1m].min()):.2f}")
                        _buf1m = io.StringIO(); _day_1m.to_csv(_buf1m, index=False)
                        st.download_button(
                            f"⬇ Download 1m data ({_1m_date_sel})",
                            _buf1m.getvalue(),
                            file_name=f"moil_1m_{_1m_date_sel}.csv",
                            mime="text/csv",
                            key="dl_1m",
                        )
                    else:
                        st.warning("1m data loaded but required columns (open/high/low/close) not found.")
                else:
                    st.info("💡 Click **Fetch 1m data** above, or pick a past date. For older data, upload a CSV/Parquet.")

            st.markdown("---")

            # ── Section 3: Trade Opportunities + Paper Trading ────────────────────
            st.markdown("### Trade Opportunities & Paper Trading")
            st.caption(
                f"Signals scored 0–100 across 6 pillars. "
                f"**Auto-execute** ≥ {AUTO_TRADE_THRESHOLD} | "
                f"**Manual review** {MANUAL_REVIEW_THRESHOLD}–{AUTO_TRADE_THRESHOLD-1} | "
                f"**Skip** < {MANUAL_REVIEW_THRESHOLD}"
            )

            _opp_prepared = st.session_state.get("intraday_prepared")
            if _opp_prepared is None:
                st.info("Run the backtest (Section 4 below) first to load prepared data, then come back here to see live opportunities.")
            else:
                _opp_dates = sorted(_opp_prepared["datetime"].dt.date.unique(), reverse=True)
                _opp_date = st.selectbox(
                    "Select day to analyse",
                    _opp_dates,
                    format_func=lambda d: d.strftime("%A, %d %b %Y"),
                    key="opp_day_select",
                )
                _opp_day = _opp_prepared[_opp_prepared["datetime"].dt.date == _opp_date].copy().reset_index(drop=True)

                if not _opp_day.empty:
                    _opp_cfg = StrategyConfig()
                    _opp_or_high = _opp_day["opening_range_high"].iloc[0]
                    _opp_or_low = _opp_day["opening_range_low"].iloc[0]
                    _opp_or_range_pct = (
                        (_opp_or_high - _opp_or_low) / _opp_or_low * 100
                        if pd.notna(_opp_or_high) and pd.notna(_opp_or_low) and _opp_or_low else 0.0
                    )

                    from intraday_vwap_orb.signals import generate_signals_with_stats, volume_ratio_ok
                    _opp_sigs, _opp_stats = generate_signals_with_stats(_opp_day, _opp_cfg, _opp_or_high, _opp_or_low)

                    # ── Day summary metrics ───────────────────────────────────────
                    om1, om2, om3, om4, om5 = st.columns(5)
                    om1.metric("OR High", f"₹{_opp_or_high:.2f}",
                        help="Opening Range High — highest price in the first 15 min. "
                             "A close ABOVE this level triggers a LONG breakout signal.")
                    om2.metric("OR Low", f"₹{_opp_or_low:.2f}",
                        help="Opening Range Low — lowest price in the first 15 min. "
                             "A close BELOW this level triggers a SHORT breakout signal.")
                    om3.metric("OR Range", f"{_opp_or_range_pct:.2f}%",
                        help="Width of the Opening Range as % of OR Low. "
                             "Narrow range (<0.3%) = tight coil, explosive breakout potential. "
                             "Wide range (>1.5%) = large SL needed, lower R:R trades.")
                    om4.metric("Signals found", len(_opp_sigs),
                        help="Number of ORB signals that passed entry criteria (volume, VWAP, EMA filters). "
                             "More signals on a day can indicate choppy conditions — "
                             "prefer first clean breakout of the day.")
                    _opp_open = _opp_day["open"].iloc[0]
                    om5.metric("Day open", f"₹{_opp_open:.2f}",
                        help="MOIL opening price for this session. "
                             "A gap-up open vs previous close adds gap risk; "
                             "small gap = cleaner OR, large gap = wider SL required.")

                    if not _opp_sigs:
                        st.info("No ORB signals met the criteria on this day.")
                    else:
                        # ── Score each signal ─────────────────────────────────────
                        _scored = []
                        for _sig in _opp_sigs:
                            _srow = _opp_day.iloc[_sig.idx]
                            _ok_vol, _cur_vol, _avg_vol = volume_ratio_ok(_opp_day, _sig.idx, _opp_cfg.volume_lookback_bars, _opp_cfg.volume_multiplier)
                            _vol_ratio = _cur_vol / _avg_vol if _avg_vol and _avg_vol > 0 else 0.0
                            _cr = score_signal_confidence(
                                row=_srow,
                                volume_ratio=_vol_ratio,
                                or_range_pct=_opp_or_range_pct,
                                gap_pct=None,
                                side=_sig.side,
                                signal_time=_srow.get("time") if hasattr(_srow, "get") else _srow["time"],
                                or_high=_opp_or_high,
                                or_low=_opp_or_low,
                            )
                            _entry_bar_idx = min(_sig.idx + 1, len(_opp_day) - 1)
                            _entry_bar = _opp_day.iloc[_entry_bar_idx]
                            _entry_px = float(_entry_bar["open"])
                            _vwap_e = float(_srow.get("vwap", _entry_px) if hasattr(_srow, "get") else _srow["vwap"])
                            if _sig.side == "LONG":
                                _stop_px = min(_vwap_e, float(_srow["low"])) if _vwap_e < _entry_px else float(_srow["low"])
                                _stop_px = _stop_px if _stop_px < _entry_px else _entry_px * 0.995
                            else:
                                _stop_px = max(_vwap_e, float(_srow["high"])) if _vwap_e > _entry_px else float(_srow["high"])
                                _stop_px = _stop_px if _stop_px > _entry_px else _entry_px * 1.005
                            _risk = abs(_entry_px - _stop_px)
                            _target_px = (_entry_px + _opp_cfg.risk_reward * _risk) if _sig.side == "LONG" else (_entry_px - _opp_cfg.risk_reward * _risk)
                            _track = classify_signal(_cr.score)
                            _scored.append({
                                "sig": _sig,
                                "row": _srow,
                                "cr": _cr,
                                "entry_px": _entry_px,
                                "stop_px": _stop_px,
                                "target_px": _target_px,
                                "vol_ratio": _vol_ratio,
                                "track": _track,
                                "entry_bar_idx": _entry_bar_idx,
                            })

                        # ── Opportunities table ───────────────────────────────────
                        st.markdown("#### All signal opportunities")
                        _opp_rows = []
                        for _sc in _scored:
                            _cr = _sc["cr"]
                            _opp_rows.append({
                                "Time": str(_sc["sig"].idx and _opp_day.iloc[_sc["sig"].idx]["time"]),
                                "Side": _sc["sig"].side,
                                "Entry ₹": f"₹{_sc['entry_px']:.2f}",
                                "Stop ₹": f"₹{_sc['stop_px']:.2f}",
                                "Target ₹": f"₹{_sc['target_px']:.2f}",
                                "Risk/share": f"₹{abs(_sc['entry_px']-_sc['stop_px']):.2f}",
                                "Vol ratio": f"{_sc['vol_ratio']:.2f}x",
                                "Confidence": _cr.score,
                                "Label": _cr.label,
                                "Track": _sc["track"].replace("_", " ").title(),
                                "Reason": _sc["sig"].reason,
                            })
                        _opp_df = pd.DataFrame(_opp_rows)

                        def _conf_color(v):
                            if v >= AUTO_TRADE_THRESHOLD:
                                return "background-color: #1a3a2a; color: #26a69a"
                            elif v >= MANUAL_REVIEW_THRESHOLD:
                                return "background-color: #2a2a1a; color: #f7c948"
                            else:
                                return "background-color: #2a1a1a; color: #ef5350"

                        st.dataframe(
                            _opp_df.style.map(_conf_color, subset=["Confidence"]),
                            use_container_width=True, hide_index=True,
                        )

                        # ── Column legend / tooltips ──────────────────────────────
                        with st.expander("📖 What do these columns mean?", expanded=False):
                            st.markdown("""
    | Column | What it means | Trading impact |
    |---|---|---|
    | **Side** | LONG = buy breakout above OR High; SHORT = sell breakdown below OR Low | Determines order direction in broker |
    | **Entry ₹** | Projected fill price — open of bar *after* the signal bar | Place limit/market order at this level |
    | **Stop ₹** | Initial stop-loss — below VWAP or OR Low (LONG) / above VWAP or OR High (SHORT) | Hard SL in bracket/cover order |
    | **Target ₹** | Profit target = Entry ± (Risk × R:R ratio, default 2×) | Book partial or full profit here |
    | **Risk/share** | Entry − Stop in ₹. Multiply by qty to get total rupee risk | Size your position: max loss ÷ Risk/share = qty |
    | **Vol ratio** | Current bar volume ÷ 20-bar average volume. ≥1.5× = strong breakout confirmation | Below 1.2× = weak signal, skip or reduce size |
    | **Confidence** | Composite score 0–100 across 6 pillars (volume, VWAP distance, OR range, gap risk, EMA alignment, time-of-day) | ≥70 = auto-execute; 45–69 = review; <45 = skip |
    | **Label** | Human-readable confidence band | Use as quick sanity check |
    | **Track** | Auto = machine-approved; Manual = user-approved. Tracked separately | Compare win rates to calibrate auto threshold |
    | **Reason** | Which filter the signal passed (e.g. "ORB Long above OR High + VWAP") | Confirms the technical basis |
    """)

                        # ── Per-signal Send Alert buttons ─────────────────────────
                        st.markdown("#### Send alert for a specific signal")
                        for _sc_i, _sc in enumerate(_scored):
                            _cr_i = _sc["cr"]
                            _sig_label = f"{_sc['sig'].side} @ ₹{_sc['entry_px']:.2f} — {_cr_i.score}/100 ({_cr_i.label})"
                            _btn_color = "#26a69a" if _cr_i.auto_trade else ("#f7c948" if _cr_i.manual_review else "#868993")
                            _sc_col1, _sc_col2 = st.columns([3, 1])
                            _sc_col1.markdown(
                                f"<span style='color:{_btn_color};font-size:0.85rem;'>{_sig_label}</span>",
                                unsafe_allow_html=True,
                            )
                            if _sc_col2.button("📲 Send alert", key=f"tg_manual_send_{_sc_i}",
                                               help="Send a full trade execution alert to Telegram with all levels"):
                                _fire_trade_alert(
                                    _sc, _opp_date, _opp_or_high, _opp_or_low,
                                    _opp_or_range_pct, _sc["track"], "user-manual",
                                    _opp_day, True,
                                )

                        # ── Confidence breakdown per signal ───────────────────────
                        with st.expander("Confidence breakdown by pillar", expanded=False):
                            for _i, _sc in enumerate(_scored):
                                _cr = _sc["cr"]
                                _bar_color = "#26a69a" if _cr.auto_trade else ("#f7c948" if _cr.manual_review else "#ef5350")
                                st.markdown(
                                    f"<div style='background:#1e2130;border:1px solid #2a2e39;border-radius:8px;"
                                    f"padding:12px 16px;margin-bottom:8px;'>"
                                    f"<span style='color:{_bar_color};font-weight:700;font-size:0.95rem;'>"
                                    f"{_sc['sig'].side} @ ₹{_sc['entry_px']:.2f} — "
                                    f"<span style='font-size:1.1rem;'>{_cr.score}/100</span> "
                                    f"<span style='font-size:0.8rem;font-weight:400;'>({_cr.label})</span></span>"
                                    f"<div style='background:#131722;border-radius:6px;height:8px;margin:6px 0 10px;overflow:hidden;'>"
                                    f"<div style='background:{_bar_color};width:{_cr.score}%;height:100%;'></div></div>",
                                    unsafe_allow_html=True,
                                )
                                _bd_rows = []
                                for _pname, _pval in _cr.breakdown.items():
                                    _bd_rows.append({
                                        "Pillar": _pname.replace("_", " ").title(),
                                        "Value": str(_pval["value"]),
                                        "Points": f"{_pval['pts']} / {_pval['max']}",
                                    })
                                st.dataframe(pd.DataFrame(_bd_rows), use_container_width=True, hide_index=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown("---")

                        # ── Auto paper trade execution ────────────────────────────
                        st.markdown("#### Paper trading")
                        _auto_col, _thresh_col = st.columns([2, 1])
                        with _thresh_col:
                            st.metric("Auto threshold", f"≥ {AUTO_TRADE_THRESHOLD}")
                            st.metric("Manual review threshold", f"≥ {MANUAL_REVIEW_THRESHOLD}")

                        with _auto_col:
                            if st.button("🤖 Auto-execute high-confidence signals", key="pt_auto_exec"):
                                _new_count = 0
                                _existing_ids = {t.id for t in st.session_state.paper_log}
                                for _sc in _scored:
                                    if _sc["track"] != "auto":
                                        continue
                                    _sig_id = f"{_opp_date}_{_sc['sig'].idx}_{_sc['sig'].side}"
                                    if _sig_id in _existing_ids:
                                        continue
                                    _ps = PaperSignal(
                                        id=_sig_id,
                                        trade_date=str(_opp_date),
                                        signal_time=str(_opp_day.iloc[_sc["sig"].idx]["time"]),
                                        side=_sc["sig"].side,
                                        entry_price=_sc["entry_px"],
                                        stop=_sc["stop_px"],
                                        target=_sc["target_px"],
                                        confidence_score=_sc["cr"].score,
                                        confidence_label=_sc["cr"].label,
                                        volume_ratio=round(_sc["vol_ratio"], 2),
                                        or_range_pct=round(_opp_or_range_pct, 3),
                                        gap_pct=None,
                                        track="auto",
                                        approved_by="auto",
                                        approved_at=str(pd.Timestamp.now()),
                                    )
                                    _future_bars = _opp_day.iloc[_sc["entry_bar_idx"] + 1:]
                                    _ps = resolve_paper_trade(_ps, _future_bars)
                                    st.session_state.paper_log.append(_ps)
                                    _new_count += 1
                                if enable_telegram:
                                    for _sc in _scored:
                                        if _sc["track"] != "auto":
                                            continue
                                        _sig_id_chk = f"{_opp_date}_{_sc['sig'].idx}_{_sc['sig'].side}"
                                        if _sig_id_chk not in _existing_ids:
                                            _fire_trade_alert(
                                                _sc, _opp_date, _opp_or_high, _opp_or_low,
                                                _opp_or_range_pct, "auto", "auto",
                                                _opp_day, True,
                                            )
                                st.success(f"Auto-executed {_new_count} paper trade(s).")
                                st.rerun()

                        # ── Manual approval queue ─────────────────────────────────
                        _pending_manual = [
                            _sc for _sc in _scored if _sc["track"] == "manual_review"
                        ]
                        if _pending_manual:
                            st.markdown("#### Manual approval queue")
                            st.caption("These signals scored below the auto threshold but above the minimum review level. Approve to paper-trade them separately.")
                            _existing_ids = {t.id for t in st.session_state.paper_log}
                            _pending_ids_in_session = {
                                ps["id"] for ps in st.session_state.pending_signals
                            }
                            for _sc in _pending_manual:
                                _sig_id = f"{_opp_date}_{_sc['sig'].idx}_{_sc['sig'].side}"
                                _already_logged = _sig_id in _existing_ids
                                _cr = _sc["cr"]
                                _bar_color = "#f7c948"
                                with st.container():
                                    st.markdown(
                                        f"<div style='background:#1e2130;border:1px solid #2a2e39;"
                                        f"border-left:4px solid {_bar_color};border-radius:8px;"
                                        f"padding:12px 18px;margin-bottom:8px;'>"
                                        f"<span style='color:{_bar_color};font-weight:700;'>"
                                        f"{_sc['sig'].side} @ ₹{_sc['entry_px']:.2f} &nbsp;|&nbsp; "
                                        f"Confidence: {_cr.score}/100 ({_cr.label})</span><br/>"
                                        f"<span style='color:#868993;font-size:0.82rem;'>"
                                        f"Stop ₹{_sc['stop_px']:.2f} &nbsp;|&nbsp; "
                                        f"Target ₹{_sc['target_px']:.2f} &nbsp;|&nbsp; "
                                        f"Vol ratio {_sc['vol_ratio']:.2f}x &nbsp;|&nbsp; "
                                        f"OR range {_opp_or_range_pct:.2f}%</span></div>",
                                        unsafe_allow_html=True,
                                    )
                                    _ab, _rb = st.columns([1, 1])
                                    with _ab:
                                        if not _already_logged:
                                            if st.button(f"✅ Approve {_sig_id}", key=f"approve_{_sig_id}"):
                                                _ps = PaperSignal(
                                                    id=_sig_id,
                                                    trade_date=str(_opp_date),
                                                    signal_time=str(_opp_day.iloc[_sc["sig"].idx]["time"]),
                                                    side=_sc["sig"].side,
                                                    entry_price=_sc["entry_px"],
                                                    stop=_sc["stop_px"],
                                                    target=_sc["target_px"],
                                                    confidence_score=_cr.score,
                                                    confidence_label=_cr.label,
                                                    volume_ratio=round(_sc["vol_ratio"], 2),
                                                    or_range_pct=round(_opp_or_range_pct, 3),
                                                    gap_pct=None,
                                                    track="manual",
                                                    approved_by="user",
                                                    approved_at=str(pd.Timestamp.now()),
                                                )
                                                _future_bars = _opp_day.iloc[_sc["entry_bar_idx"] + 1:]
                                                _ps = resolve_paper_trade(_ps, _future_bars)
                                                st.session_state.paper_log.append(_ps)
                                                _fire_trade_alert(
                                                    _sc, _opp_date, _opp_or_high, _opp_or_low,
                                                    _opp_or_range_pct, "manual", "user",
                                                    _opp_day, enable_telegram,
                                                )
                                                st.success(f"Approved and paper-traded: {_sig_id}")
                                                st.rerun()
                                        else:
                                            st.success("Already logged")
                                    with _rb:
                                        if st.button(f"❌ Skip {_sig_id}", key=f"reject_{_sig_id}"):
                                            st.info(f"Skipped signal {_sig_id}")

                        st.markdown("---")

                        # ── Paper trade results ───────────────────────────────────
                        st.markdown("#### Paper trade results")
                        _pt_log = st.session_state.paper_log
                        if not _pt_log:
                            st.info("No paper trades yet. Use the buttons above to execute.")
                        else:
                            _pt_df = paper_log_to_df(_pt_log)

                            # Success metrics comparison
                            _auto_stats = summarise_track(_pt_df, "auto")
                            _manual_stats = summarise_track(_pt_df, "manual")
                            st.markdown("##### Success metrics: Auto vs Manual-approved")
                            _sm1, _sm2 = st.columns(2)
                            with _sm1:
                                st.markdown(
                                    "<div style='background:#1a3a2a;border:1px solid #26a69a;border-radius:8px;"
                                    "padding:14px 18px;'><span style='color:#26a69a;font-weight:700;"
                                    "font-size:0.9rem;'>🤖 AUTO TRACK</span>",
                                    unsafe_allow_html=True,
                                )
                                _sa1, _sa2, _sa3 = st.columns(3)
                                _sa1.metric("Trades", _auto_stats["trades"])
                                _sa2.metric("Win rate", f"{_auto_stats['win_rate']:.1%}")
                                _sa3.metric("Avg R", f"{_auto_stats['avg_r']:.2f}R")
                                st.metric("Total PnL (pts)", f"{_auto_stats['total_pnl']:+.2f}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            with _sm2:
                                st.markdown(
                                    "<div style='background:#2a2a1a;border:1px solid #f7c948;border-radius:8px;"
                                    "padding:14px 18px;'><span style='color:#f7c948;font-weight:700;"
                                    "font-size:0.9rem;'>👤 MANUAL TRACK</span>",
                                    unsafe_allow_html=True,
                                )
                                _mm1, _mm2, _mm3 = st.columns(3)
                                _mm1.metric("Trades", _manual_stats["trades"])
                                _mm2.metric("Win rate", f"{_manual_stats['win_rate']:.1%}")
                                _mm3.metric("Avg R", f"{_manual_stats['avg_r']:.2f}R")
                                st.metric("Total PnL (pts)", f"{_manual_stats['total_pnl']:+.2f}")
                                st.markdown("</div>", unsafe_allow_html=True)

                            st.markdown("---")

                            # Resolve open trades with latest 5m data
                            _open_pt = [t for t in _pt_log if t.status == "open"]
                            if _open_pt:
                                st.info(f"🟡 {len(_open_pt)} trade(s) are **open** — signal fired today, market bars after entry not yet available. "
                                        f"Click below after market close (or later today) to resolve them.")
                                if st.button("🔄 Resolve open trades (fetch latest 5m data)", key="pt_resolve_open"):
                                    with st.spinner("Fetching latest 5m data to resolve open trades…"):
                                        _res_df, _res_src, _res_err = fetch_ohlcv("MOIL.NS", interval="5m", period="1d")
                                    if not _res_df.empty:
                                        _res_df = _res_df.reset_index()
                                        _res_df.columns = [c.lower() for c in _res_df.columns]
                                        # Normalise to IST then drop tz for naive comparison
                                        _res_df = localize_to_ist(_res_df) if hasattr(_res_df.index, "tz") and _res_df.index.tz is not None else _res_df
                                        _dt_col = next((c for c in _res_df.columns if "datetime" in c or c == "index"), _res_df.columns[0])
                                        _dts = pd.to_datetime(_res_df[_dt_col])
                                        if _dts.dt.tz is not None:
                                            _dts = _dts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                                        _res_df = _res_df.copy()
                                        _res_df[_dt_col] = _dts
                                        for _ot in _open_pt:
                                            _entry_ts = pd.to_datetime(f"{_ot.trade_date} {_ot.signal_time}")
                                            _after_entry = _res_df[_res_df[_dt_col] > _entry_ts]
                                            resolve_paper_trade(_ot, _after_entry)
                                        st.success(f"Resolved {len(_open_pt)} trade(s) from {source_badge(_res_src)} data.")
                                        st.rerun()
                                    else:
                                        st.warning(f"Could not fetch 5m data: {_res_err}. Try again after market close.")

                            # Full log
                            _disp_cols = [c for c in [
                                "trade_date", "signal_time", "side", "track",
                                "confidence_score", "confidence_label",
                                "entry_price", "stop", "target",
                                "status", "exit_price", "exit_reason",
                                "pnl", "r_multiple", "approved_by",
                            ] if c in _pt_df.columns]
                            _pt_disp = _pt_df[_disp_cols].copy()
                            for _c in ["entry_price", "stop", "target", "exit_price"]:
                                if _c in _pt_disp.columns:
                                    _pt_disp[_c] = pd.to_numeric(_pt_disp[_c], errors="coerce").map(lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
                            if "pnl" in _pt_disp.columns:
                                _pt_disp["pnl"] = pd.to_numeric(_pt_disp["pnl"], errors="coerce").map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
                            if "r_multiple" in _pt_disp.columns:
                                _pt_disp["r_multiple"] = pd.to_numeric(_pt_disp["r_multiple"], errors="coerce").map(lambda x: f"{x:+.2f}R" if pd.notna(x) else "—")
                            if "status" in _pt_disp.columns:
                                _pt_disp["status"] = _pt_disp["status"].map(lambda s: {
                                    "open":   "🟡 open (live)",
                                    "closed": "✅ closed",
                                    "invalid": "⚠️ invalid",
                                }.get(str(s), str(s)))
                            if "exit_reason" in _pt_disp.columns:
                                _pt_disp["exit_reason"] = _pt_disp["exit_reason"].map(lambda r: {
                                    "awaiting_market":  "⏳ market still open",
                                    "stop_hit":         "🛑 stop hit",
                                    "target_hit":       "🎯 target hit",
                                    "square_off_15:15": "🔔 sq-off 15:15",
                                    "eod_square_off":   "🔔 EOD sq-off",
                                    "zero_risk":        "⚠️ zero risk",
                                }.get(str(r), str(r)) if pd.notna(r) else "—")
                            st.dataframe(_pt_disp, use_container_width=True, hide_index=True)

                            _pt_csv = io.StringIO()
                            _pt_df.to_csv(_pt_csv, index=False)
                            _dl1, _dl2 = st.columns([1, 4])
                            _dl1.download_button("⬇ Paper trade log CSV", _pt_csv.getvalue(), file_name="paper_trades.csv", mime="text/csv")
                            if _dl2.button("🗑️ Clear paper trade log", key="pt_clear"):
                                st.session_state.paper_log = []
                                st.rerun()

            st.markdown("---")

            # ── Section 4: VWAP + ORB backtest (5m) ─────────────────────────────
            st.markdown("### VWAP + ORB backtest (5m)")

            with st.expander("ℹ️ Why 5-min? How this engine works", expanded=False):
                st.markdown("""
**5-min is the right timeframe for MOIL ORB** — here's why:

| What | Detail |
|------|--------|
| **Opening Range** | 09:15–09:30 = 3 clean 5m bars — enough to form a reliable OR, low noise |
| **Volume baseline** | Looks back 5 bars = 25 min of context. At 1m this shrinks to 5 min — too noisy |
| **Entry window** | 09:30–11:00 = 18 bars at 5m. At 1m that's 90 bars with far more false breakouts |
| **MOIL liquidity** | Mid-cap (~₹20Cr/day). Many 1m bars are thin/zero-volume — VWAP & volume signals break down |
| **1m role** | 1m chart is only for **visual drill-down** and fine-tuning entry price after a 5m signal fires |

The engine auto-fetches the last 60 days of real 5m MOIL data from TradingView. Upload a longer CSV for extended backtests.
""")

            _bt_c1, _bt_c2 = st.columns([3, 1])
            _bt_days = _bt_c1.slider("Lookback (days of 5m history)", 7, 90, 60, 7, key="it_bt_days",
                help="How many calendar days of 5m OHLCV to fetch. TradingView provides ~60 days free.")
            _bt_upload = _bt_c2.file_uploader(
                "Or upload 5m CSV/Parquet",
                type=["csv", "parquet", "pq"],
                accept_multiple_files=False,
                key="intraday_bt_upload",
                label_visibility="collapsed",
            )

            with st.expander("⚙️ Strategy settings", expanded=False):
                _vol_mult = st.slider(
                    "Volume ratio threshold (x avg of prev 5 bars)", 1.0, 4.0, 2.0, 0.1, key="it_vol",
                    help="A breakout bar must have volume ≥ this multiple of the average of the prior 5 bars. "
                         "2.0 means the breakout candle must be at least 2× the recent average volume — filters out low-conviction moves. "
                         "Lower = more signals (more noise), higher = fewer but stronger signals.")
                _rr = st.slider(
                    "Risk-reward ratio (target / stop)", 1.0, 3.0, 2.0, 0.1, key="it_rr",
                    help="For every ₹1 risked on the stop-loss, the target is set at ₹R. "
                         "2.0 means target = 2× the distance from entry to stop. "
                         "Higher R gives better per-trade payout but fewer targets get hit.")
                _one_trade = st.checkbox(
                    "One trade per day", value=True, key="it_one",
                    help="Restricts the engine to one entry per trading session. "
                         "Recommended for MOIL: after the first ORB breakout, liquidity often thins out and re-entries add risk. "
                         "Uncheck only to test multiple intraday signals.")
                _ema_filter = st.checkbox(
                    "Require EMA20 filter", value=False, key="it_ema",
                    help="When enabled, a LONG signal is only taken if price > EMA20, and SHORT only if price < EMA20. "
                         "Aligns trades with the short-term trend. "
                         "Currently off by default — MOIL's intraday trend reversals can make this filter too restrictive.")
                _slip_pct = st.number_input(
                    "Total cost % per side (brokerage + STT + slippage)",
                    min_value=0.0, max_value=2.0, value=0.07, step=0.01,
                    format="%.2f", key="it_slip",
                    help="All-in cost on each trade leg as % of traded amount.\n\n"
                         "Example: buy 10 shares @ ₹300 = ₹3,000 + sell @ ₹310 = ₹3,100\n"
                         "At 0.07%: cost = 3000×0.07% + 3100×0.07% = ₹2.10 + ₹2.17 = ₹4.27 total\n\n"
                         "Typical breakdown per side:\n"
                         "  • Brokerage: ~0.03%\n"
                         "  • STT (intraday sell): ~0.025%\n"
                         "  • Market impact / spread: ~0.015%\n\n"
                         "Use 0.10–0.15% for conservative estimates on thin days.")
                _slip = _slip_pct * 100   # % → bps for StrategyConfig.slippage_bps
                _txn_cost_bps = 0.0      # already included in _slip — avoid double-counting

            if st.button("▶️ Run intraday backtest", use_container_width=True, key="it_run", type="primary"):
                with st.spinner("Running VWAP + ORB backtest..."):
                    try:
                        if _bt_upload is not None:
                            # User-supplied file takes priority
                            if _bt_upload.name.lower().endswith((".parquet", ".pq")):
                                _df_raw = pd.read_parquet(_bt_upload)
                            else:
                                _df_raw = pd.read_csv(_bt_upload)
                            if "symbol" not in _df_raw.columns:
                                _df_raw["symbol"] = "MOIL"
                            st.caption("📁 Using uploaded file.")
                        else:
                            # Auto-fetch real 5m MOIL data: TradingView → Dhan → Zerodha → yfinance
                            with st.spinner(f"Fetching {_bt_days}d of real MOIL 5m data (TradingView → yfinance)…"):
                                _bt_5m_df, _bt_5m_src, _bt_5m_err = fetch_ohlcv(
                                    "MOIL.NS", interval="5m", days=_bt_days
                                )
                            if not _bt_5m_df.empty:
                                _bt_5m_df = _bt_5m_df.reset_index()
                                _dt_c = next((c for c in _bt_5m_df.columns if "date" in c.lower() or "time" in c.lower()), None)
                                if _dt_c and _dt_c != "datetime":
                                    _bt_5m_df = _bt_5m_df.rename(columns={_dt_c: "datetime"})
                                _bt_5m_df["symbol"] = "MOIL"
                                _df_raw = _bt_5m_df
                                st.caption(f"Live data: {source_badge(_bt_5m_src)} — {len(_df_raw):,} bars ({_bt_days}d)")
                            else:
                                # Final fallback: sample file
                                _sample_path = Path("data/sample/intraday_sample_moil.csv")
                                if _sample_path.exists():
                                    _df_raw = load_intraday(_sample_path).df
                                    st.warning(f"All live feeds failed ({_bt_5m_err}). Running on sample data — results are illustrative only.")
                                else:
                                    st.error(f"No data available. Live feeds: {_bt_5m_err}. Upload a 5m CSV to proceed.")
                                    st.stop()
                        _cfg = StrategyConfig(
                            volume_multiplier=_vol_mult, risk_reward=_rr,
                            one_trade_per_day=_one_trade, require_ema_filter=_ema_filter,
                            slippage_bps=_slip, transaction_cost_bps=0,
                        )
                        _prepared = prepare_intraday(_df_raw, _cfg)
                        _trades, _daily, _bt_summary, _stats = run_backtest(_prepared, _cfg)
                        _feats = daily_features(_trades, _stats, _prepared)
                        st.session_state["intraday_trades"] = _trades
                        st.session_state["intraday_daily"] = _daily
                        st.session_state["intraday_feats"] = _feats
                        st.session_state["intraday_summary"] = _bt_summary
                        st.session_state["intraday_prepared"] = _prepared
                        st.success(f"Backtest complete: {_bt_summary['total_trades']} trades")
                    except Exception as _e:
                        st.error(f"Backtest failed: {_e}")

            if "intraday_summary" in st.session_state and st.session_state.get("intraday_trades") is not None:
                _s = st.session_state["intraday_summary"]
                _trades = st.session_state["intraday_trades"]
                _daily = st.session_state.get("intraday_daily", pd.DataFrame())
                _feats = st.session_state.get("intraday_feats", pd.DataFrame())

                st.markdown("#### Engine performance")
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Total trades", _s.get("total_trades", 0))
                k2.metric("Win rate", f"{_s.get('win_rate', 0):.1%}")
                k3.metric("Net PnL ₹", f"{_s.get('net_pnl', 0):,.0f}")
                k4.metric("Avg R", f"{_s.get('avg_r', 0):.2f}R")
                _pf = _s.get("profit_factor")
                k5.metric("Profit factor", f"{_pf:.2f}" if _pf else "—")
                st.markdown("---")

                if not _daily.empty:
                    st.markdown("##### Daily P&L summary")
                    _dd = _daily.copy()
                    _dd["result"] = _dd["net_pnl"].apply(lambda x: "✅ Win" if x > 0 else ("❌ Loss" if x < 0 else "➖ Flat"))
                    _dd["net_pnl"] = _dd["net_pnl"].map(lambda x: f"₹{x:,.0f}")
                    _dd["gross_pnl"] = _dd["gross_pnl"].map(lambda x: f"₹{x:,.0f}")
                    _dd["avg_r"] = _dd["avg_r"].map(lambda x: f"{x:.2f}R" if pd.notna(x) else "—")
                    if "cumulative_net_pnl" in _dd.columns:
                        _dd["cumulative_net_pnl"] = _dd["cumulative_net_pnl"].map(lambda x: f"₹{x:,.0f}")
                    _show = [c for c in ["trade_date", "symbol", "trades", "winners", "losers", "result", "gross_pnl", "net_pnl", "avg_r", "cumulative_net_pnl"] if c in _dd.columns]
                    st.dataframe(_dd[_show], use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("##### Drill-down: trades on a specific date")
                if not _trades.empty:
                    _trade_dates = sorted(_trades["trade_date"].astype(str).unique(), reverse=True)
                    _sel_date = st.selectbox("Select trade date", _trade_dates, key="it_drill_date")
                    _day_trades = _trades[_trades["trade_date"].astype(str) == _sel_date]

                    if not _day_trades.empty:
                        _val_cols = [c for c in [
                            "trade_date", "entry_datetime", "exit_datetime", "side",
                            "entry_price", "exit_price", "initial_stop", "target",
                            "risk_per_share", "realized_r_multiple", "net_pnl",
                            "exit_reason", "bars_held", "entry_vwap", "entry_ema20",
                            "opening_range_high", "opening_range_low", "breakout_volume_ratio", "notes"
                        ] if c in _day_trades.columns]
                        _vdf = _day_trades[_val_cols].copy()
                        for _col in ["entry_price", "exit_price", "initial_stop", "target", "entry_vwap", "entry_ema20", "opening_range_high", "opening_range_low"]:
                            if _col in _vdf.columns:
                                _vdf[_col] = _vdf[_col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
                        if "realized_r_multiple" in _vdf.columns:
                            _vdf["realized_r_multiple"] = _vdf["realized_r_multiple"].map(lambda x: f"{x:.2f}R" if pd.notna(x) else "—")
                        if "breakout_volume_ratio" in _vdf.columns:
                            _vdf["breakout_volume_ratio"] = _vdf["breakout_volume_ratio"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
                        if "net_pnl" in _vdf.columns:
                            _vdf["net_pnl"] = _vdf["net_pnl"].map(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
                        st.dataframe(_vdf, use_container_width=True, hide_index=True)

                        st.markdown(f"###### Intraday chart — {_sel_date} (5m bars with trade annotations)")
                        _prep_df = st.session_state.get("intraday_prepared")
                        _day_bars = _prep_df[_prep_df["datetime"].dt.date.astype(str) == _sel_date] if _prep_df is not None else pd.DataFrame()
                        if not _day_bars.empty:
                            _fd = go.Figure()
                            _fd.add_trace(go.Candlestick(
                                x=_day_bars["datetime"], open=_day_bars["open"],
                                high=_day_bars["high"], low=_day_bars["low"], close=_day_bars["close"],
                                name="MOIL 5m", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                                increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
                            ))
                            if "vwap" in _day_bars.columns:
                                _fd.add_trace(go.Scatter(x=_day_bars["datetime"], y=_day_bars["vwap"], mode="lines", name="VWAP", line=dict(color="#f7c948", width=1.5, dash="dot")))
                            if "ema20" in _day_bars.columns:
                                _fd.add_trace(go.Scatter(x=_day_bars["datetime"], y=_day_bars["ema20"], mode="lines", name="EMA20", line=dict(color="#2196f3", width=1.2, dash="dash")))
                            _or_h = _day_trades["opening_range_high"].iloc[0] if "opening_range_high" in _day_trades.columns else None
                            _or_l = _day_trades["opening_range_low"].iloc[0] if "opening_range_low" in _day_trades.columns else None
                            if pd.notna(_or_h) and _or_h:
                                _fd.add_hline(y=float(_or_h), line=dict(color="#ff9800", width=1, dash="dot"), annotation_text="OR High", annotation_position="right")
                            if pd.notna(_or_l) and _or_l:
                                _fd.add_hline(y=float(_or_l), line=dict(color="#ff9800", width=1, dash="dot"), annotation_text="OR Low", annotation_position="right")
                            for _, _tr in _day_trades.iterrows():
                                _tc = "#26a69a" if _tr.get("side") == "LONG" else "#ef5350"
                                _ms = "triangle-up" if _tr.get("side") == "LONG" else "triangle-down"
                                try:
                                    _fd.add_trace(go.Scatter(x=[_tr["entry_datetime"]], y=[float(_tr["entry_price"])], mode="markers+text", marker=dict(symbol=_ms, size=14, color=_tc), text=[f"Entry {_tr['side']}"], textposition="top center", textfont=dict(color=_tc, size=10), name="Entry", showlegend=False))
                                    _fd.add_trace(go.Scatter(x=[_tr["exit_datetime"]], y=[float(_tr["exit_price"])], mode="markers+text", marker=dict(symbol="x", size=12, color="#b2b5be"), text=[f"Exit ({_tr.get('exit_reason','')})"], textposition="top center", textfont=dict(color="#b2b5be", size=10), name="Exit", showlegend=False))
                                    _fd.add_hline(y=float(_tr["initial_stop"]), line=dict(color="#ef5350", width=1, dash="dot"), annotation_text="Stop", annotation_position="left")
                                    _fd.add_hline(y=float(_tr["target"]), line=dict(color="#26a69a", width=1, dash="dot"), annotation_text="Target", annotation_position="left")
                                except Exception:
                                    pass
                            _fd.update_layout(
                                template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
                                xaxis=dict(gridcolor="#2a2e39", showgrid=True),
                                yaxis=dict(gridcolor="#2a2e39", showgrid=True),
                                xaxis_rangeslider_visible=False, height=420, hovermode="x unified",
                                legend=dict(bgcolor="#1e2130", bordercolor="#2a2e39", font=dict(color="#d1d4dc")),
                            )
                            st.plotly_chart(_fd, use_container_width=True)
                        else:
                            st.caption("Re-run the backtest to populate the intraday chart.")

                        with st.expander("Signal validation — why this trade fired", expanded=False):
                            for _, _tr in _day_trades.iterrows():
                                _sc = "#26a69a" if _tr.get("side") == "LONG" else "#ef5350"
                                st.markdown(f"""
        <div style="background:#1e2130;border:1px solid #2a2e39;border-radius:8px;padding:14px 18px;margin-bottom:10px;">
        <span style="color:{_sc};font-weight:700;font-size:1rem;">{_tr.get('side','?')} @ ₹{float(_tr.get('entry_price',0)):.2f}</span>
        <span style="color:#868993;font-size:0.8rem;margin-left:12px;">{_tr.get('entry_datetime','')}</span><br/>
        <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:0.82rem;">
        <tr><td style="color:#868993;padding:3px 8px;">OR High / Low</td><td style="color:#d1d4dc;padding:3px 8px;">₹{float(_tr.get('opening_range_high',0)):.2f} / ₹{float(_tr.get('opening_range_low',0)):.2f}</td>
            <td style="color:#868993;padding:3px 8px;">VWAP at entry</td><td style="color:#d1d4dc;padding:3px 8px;">₹{float(_tr.get('entry_vwap',0)):.2f}</td></tr>
        <tr><td style="color:#868993;padding:3px 8px;">Volume ratio</td><td style="color:#d1d4dc;padding:3px 8px;">{float(_tr.get('breakout_volume_ratio',0)):.2f}x</td>
            <td style="color:#868993;padding:3px 8px;">EMA20 at entry</td><td style="color:#d1d4dc;padding:3px 8px;">₹{float(_tr.get('entry_ema20',0)):.2f}</td></tr>
        <tr><td style="color:#868993;padding:3px 8px;">Stop</td><td style="color:#ef5350;padding:3px 8px;">₹{float(_tr.get('initial_stop',0)):.2f}</td>
            <td style="color:#868993;padding:3px 8px;">Target</td><td style="color:#26a69a;padding:3px 8px;">₹{float(_tr.get('target',0)):.2f}</td></tr>
        <tr><td style="color:#868993;padding:3px 8px;">Exit reason</td><td style="color:#d1d4dc;padding:3px 8px;">{_tr.get('exit_reason','?')}</td>
            <td style="color:#868993;padding:3px 8px;">Realized R</td><td style="color:#d1d4dc;padding:3px 8px;">{float(_tr.get('realized_r_multiple',0)):.2f}R</td></tr>
        <tr><td style="color:#868993;padding:3px 8px;">Net PnL</td><td style="color:{_sc};padding:3px 8px;font-weight:600;">₹{float(_tr.get('net_pnl',0)):,.0f}</td>
            <td style="color:#868993;padding:3px 8px;">Bars held</td><td style="color:#d1d4dc;padding:3px 8px;">{_tr.get('bars_held','?')}</td></tr>
        </table></div>""", unsafe_allow_html=True)
                    else:
                        st.info(f"No trades on {_sel_date}.")

                st.markdown("---")
                _tbuf = io.StringIO(); _trades.to_csv(_tbuf, index=False)
                _dbuf = io.StringIO(); _daily.to_csv(_dbuf, index=False)
                _fbuf = io.StringIO(); _feats.to_csv(_fbuf, index=False)
                _dc1, _dc2, _dc3 = st.columns(3)
                _dc1.download_button("⬇ Trades CSV", _tbuf.getvalue(), file_name="intraday_trades.csv", mime="text/csv", use_container_width=True)
                _dc2.download_button("⬇ Daily PnL CSV", _dbuf.getvalue(), file_name="intraday_daily_pnl.csv", mime="text/csv", use_container_width=True)
                _dc3.download_button("⬇ Features CSV", _fbuf.getvalue(), file_name="intraday_features.csv", mime="text/csv", use_container_width=True)

    with _t9:
        with _mtf_sub:
            st.subheader("🧠 MTF Adaptive Trading Engine")
            st.caption(
                "Multi-Timeframe Analysis + MTF Leverage intelligence for MOIL. "
                "Four independent agents → composite confidence score → actionable trade plan."
            )

            # ── Inputs ──────────────────────────────────────────────────────
            st.markdown("#### Configuration")
            _mtf_c1, _mtf_c2, _mtf_c3 = st.columns(3)
            _mtf_capital = _mtf_c1.number_input(
                "Capital (₹)", min_value=10_000, max_value=10_000_000,
                value=100_000, step=10_000,
                help="Your total capital for position sizing. Risk/share × qty = total ₹ at risk.",
            )
            _mtf_risk_pct = _mtf_c2.slider(
                "Max risk per trade (%)", min_value=0.5, max_value=3.0, value=1.5, step=0.25,
                help="Max % of capital to risk on a single trade. 1-2% is standard professional sizing.",
            )
            _mtf_sl_mult = _mtf_c3.slider(
                "SL ATR multiplier", min_value=0.5, max_value=3.0, value=1.5, step=0.25,
                help="Stop = Entry ± (ATR × this). Higher = wider stop, fewer stop-outs but larger risk.",
            )

            _mtf_c4, _mtf_c5 = st.columns(2)
            _mtf_rr_t1 = _mtf_c4.slider(
                "R:R Target 1", min_value=1.0, max_value=3.0, value=1.5, step=0.25,
                help="T1 = Entry ± (Risk × this). First target for partial profit booking.",
            )
            _mtf_rr_t2 = _mtf_c5.slider(
                "R:R Target 2", min_value=1.5, max_value=5.0, value=2.5, step=0.25,
                help="T2 = Entry ± (Risk × this). Final target, let remaining position ride.",
            )

            # ── Data sources ─────────────────────────────────────────────────
            st.markdown("#### Data")
            st.caption(
                "The engine uses the market data already loaded in this session. "
                "Upload an optional LTF (1H/15m) CSV for more precise entry timing."
            )
            _mtf_ltf_upload = st.file_uploader(
                "LTF OHLCV (1H or 15m) — optional",
                type=["csv", "parquet"],
                key="mtf_ltf_upload",
                help="If not uploaded the engine uses daily data for LTF entry agent too.",
            )
            _mtf_ltf_df = None
            if _mtf_ltf_upload is not None:
                try:
                    if _mtf_ltf_upload.name.lower().endswith(".parquet"):
                        _mtf_ltf_df = pd.read_parquet(_mtf_ltf_upload)
                    else:
                        _mtf_ltf_df = pd.read_csv(_mtf_ltf_upload)
                    _mtf_ltf_df.columns = [c.lower() for c in _mtf_ltf_df.columns]
                    st.success(f"LTF data loaded: {len(_mtf_ltf_df):,} bars")
                except Exception as _e:
                    st.error(f"LTF parse error: {_e}")

            # ── Adaptive weights display + reset ─────────────────────────────
            with st.expander("⚙️ Adaptive weights", expanded=False):
                _aw = st.session_state.mtf_adaptive_weights
                _aw_dict = _aw.get_weights()
                st.caption("Weights self-adjust based on which pillars predicted winning trades.")
                _wc1, _wc2, _wc3, _wc4 = st.columns(4)
                _wc1.metric("Trend weight",  f"{_aw_dict['trend_weight']:.3f}",
                    help="Weight for HTF Trend Agent score (EMA alignment, structure).")
                _wc2.metric("Setup weight",  f"{_aw_dict['setup_weight']:.3f}",
                    help="Weight for Setup Detection score (pullback/breakout/reversal quality).")
                _wc3.metric("Conf weight",   f"{_aw_dict['conf_weight']:.3f}",
                    help="Weight for LTF confirmation (candlestick, RSI bounce, micro breakout).")
                _wc4.metric("Volume weight", f"{_aw_dict['volume_weight']:.3f}",
                    help="Weight for volume expansion factor vs 20-bar average.")
                if st.button("Reset weights to defaults", key="mtf_reset_weights"):
                    st.session_state.mtf_adaptive_weights = AdaptiveWeights()
                    st.toast("Weights reset to defaults.", icon="🔄")

            st.markdown("---")

            # ── Run engine ───────────────────────────────────────────────────
            if st.button("🚀 Run MTF Analysis", key="run_mtf", type="primary"):
                _daily_for_mtf = prices.copy() if not prices.empty else pd.DataFrame()

                # Build sector df from prices if available
                _sector_for_mtf = None
                if "NIFTY Metal" in prices.columns:
                    _sector_for_mtf = prices[["NIFTY Metal"]].rename(columns={"NIFTY Metal": "close"})

                if _daily_for_mtf.empty:
                    st.error("No daily price data available. Load market data first.")
                else:
                    # Fetch real MOIL.NS daily OHLCV: TradingView → Dhan → Zerodha → yfinance
                    _moil_daily, _mtf_src, _mtf_err = fetch_daily("MOIL.NS", years=1)
                    if not _moil_daily.empty:
                        st.caption(f"MTF engine data: {source_badge(_mtf_src)} — {len(_moil_daily)} daily bars")
                    else:
                        if _mtf_err:
                            st.warning(f"All feeds failed: {_mtf_err}. Falling back to price panel.")
                        # Graceful fallback: approximate OHLCV from close panel
                        _moil_close = _daily_for_mtf["MOIL"].dropna() if "MOIL" in _daily_for_mtf.columns else pd.Series(dtype=float)
                        if _moil_close.empty:
                            st.error("MOIL data not found. Configure Dhan/Kite credentials or ensure market data is loaded.")
                            _moil_daily = None
                        else:
                            st.warning("Using close-only fallback: OHLCV approximated. ATR and candle patterns will be inaccurate.")
                            _moil_daily = pd.DataFrame({
                                "open":   _moil_close, "high": _moil_close,
                                "low":    _moil_close, "close": _moil_close,
                                "volume": float("nan"),
                            })

                    if _moil_daily is not None and not _moil_daily.empty:
                        with st.spinner("Running 4-agent MTF engine…"):
                            _mtf_result = run_composite_engine(
                                daily_df=_moil_daily,
                                ltf_df=_mtf_ltf_df,
                                sector_df=_sector_for_mtf,
                                symbol="MOIL",
                                capital_inr=float(_mtf_capital),
                                max_risk_pct=float(_mtf_risk_pct),
                                adaptive_weights=st.session_state.mtf_adaptive_weights,
                            )
                        st.session_state["mtf_last_result"] = _mtf_result

            # ── Results ──────────────────────────────────────────────────────
            _mtf_res = st.session_state.get("mtf_last_result")
            if _mtf_res:
                _bias    = _mtf_res["bias"]
                _score   = _mtf_res["confidence_score"]
                _dec     = _mtf_res["decision"]
                _action  = _mtf_res["action"]

                # Decision colour
                _dec_color = {"strong_trade": "#26a69a", "moderate_trade": "#f7c948", "avoid": "#ef5350"}.get(_dec, "#868993")
                _bias_color = {"bullish": "#26a69a", "bearish": "#ef5350", "neutral": "#868993"}.get(_bias, "#868993")
                _bar_fill = min(100, _score)

                # ── Headline card ─────────────────────────────────────────────
                st.markdown(
                    f"<div style='background:#0f1117;border:1px solid {_dec_color};"
                    f"border-radius:12px;padding:20px 24px;margin-bottom:16px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='color:{_bias_color};font-size:1.4rem;font-weight:700;'>MOIL — {_bias.upper()}</span>"
                    f"<span style='background:{_dec_color}22;border:1px solid {_dec_color};"
                    f"border-radius:20px;padding:4px 16px;font-size:0.85rem;"
                    f"font-weight:600;color:{_dec_color};'>{_dec.replace('_',' ').title()}</span>"
                    f"</div>"
                    f"<div style='margin:10px 0 6px;font-size:0.8rem;color:#868993;'>Confidence Score</div>"
                    f"<div style='background:#131722;border-radius:6px;height:12px;overflow:hidden;margin-bottom:6px;'>"
                    f"<div style='background:{_dec_color};width:{_bar_fill}%;height:100%;'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;'>"
                    f"<span style='color:{_dec_color};font-size:1.6rem;font-weight:700;'>{_score}/100</span>"
                    f"<span style='color:#868993;font-size:0.82rem;align-self:flex-end;'>"
                    f"Action: <strong style='color:#d1d4dc;'>{_action.upper()}</strong></span></div>"
                    f"<div style='color:#868993;font-size:0.78rem;margin-top:8px;'>{_mtf_res['notes']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Classify trade type (intraday vs margin) ──────────────────
                # Intraday: leverage ≤ 1x OR action="wait" → intraday same-day
                # Margin  : leverage > 1x → MTF positional/swing with leverage
                _lev = _mtf_res["leverage"]
                _trade_type = "intraday" if _lev <= 1.0 else "margin"
                _tt_label = "⚡ Intraday" if _trade_type == "intraday" else "🏦 Margin (MTF)"
                _tt_color = "#2962ff" if _trade_type == "intraday" else "#f7c948"

                # ── Daily auto-cap counters ───────────────────────────────────
                MAX_AUTO_INTRADAY = 2
                MAX_AUTO_MARGIN   = 1
                _ai_count = st.session_state.mtf_auto_intraday_count
                _am_count = st.session_state.mtf_auto_margin_count
                _can_auto_intraday = _ai_count < MAX_AUTO_INTRADAY
                _can_auto_margin   = _am_count < MAX_AUTO_MARGIN

                # ── Build helper for sending MTF alert ────────────────────────
                def _send_mtf_alert(auto: bool) -> None:
                    _htf_ag = _mtf_res["agent_outputs"]["htf"]
                    _sd_ag  = _mtf_res["agent_outputs"]["setup"]
                    _rk_ag  = _mtf_res["agent_outputs"]["risk"]
                    _msg = build_mtf_alert_message(
                        symbol="MOIL",
                        trade_type=_trade_type,
                        action=_mtf_res["action"],
                        bias=_mtf_res["bias"],
                        confidence_score=_mtf_res["confidence_score"],
                        decision=_mtf_res["decision"],
                        entry=_mtf_res["entry"],
                        stop_loss=_mtf_res["stop_loss"],
                        targets=_mtf_res["targets"],
                        leverage=_mtf_res["leverage"],
                        position_size=_mtf_res["position_size"],
                        risk_per_trade=_mtf_res["risk_per_trade"],
                        capital_inr=float(_mtf_capital),
                        notes=_mtf_res["notes"],
                        auto_executed=auto,
                        ema20=_htf_ag.get("ema20", 0.0),
                        ema50=_htf_ag.get("ema50", 0.0),
                        setup_type=_sd_ag.get("setup_type", "—"),
                        setup_quality=_sd_ag.get("setup_quality", 0.0),
                        vol_ratio=_sd_ag.get("volume_ratio", 1.0),
                        regime_capped=_rk_ag.get("regime_capped", False),
                    )
                    _ok, _detail = send_telegram_alert(_msg)
                    if _ok:
                        st.toast("📲 MTF alert sent to Telegram!", icon="✅")
                    else:
                        st.toast(f"Telegram: {_detail}", icon="⚠️")

                # ── Trade card ────────────────────────────────────────────────
                st.markdown("#### Trade plan")
                _risk_inr = _mtf_res["risk_per_trade"]

                # Confidence rating label
                _score = _mtf_res["confidence_score"]
                _dec   = _mtf_res["decision"]
                if _dec == "strong_trade":
                    _conf_label = "🔥 HIGH CONFIDENCE"
                    _conf_color_card = "#26a69a"
                elif _dec == "moderate_trade":
                    _conf_label = "⚡ MODERATE CONFIDENCE"
                    _conf_color_card = "#f7c948"
                else:
                    _conf_label = "⛔ LOW CONFIDENCE — AVOID"
                    _conf_color_card = "#ef5350"

                _bar_pct = min(100, _score)

                st.markdown(
                    f"<div style='background:#0f1117;border:1px solid {_conf_color_card};"
                    f"border-radius:10px;padding:16px 20px;margin-bottom:12px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
                    f"<span style='color:{_conf_color_card};font-weight:700;font-size:1rem;'>{_conf_label}</span>"
                    f"<span style='background:{_tt_color}22;border:1px solid {_tt_color};"
                    f"border-radius:12px;padding:2px 12px;font-size:0.75rem;"
                    f"font-weight:600;color:{_tt_color};'>{_tt_label}</span>"
                    f"</div>"
                    f"<div style='background:#131722;border-radius:4px;height:8px;overflow:hidden;margin-bottom:6px;'>"
                    f"<div style='background:{_conf_color_card};width:{_bar_pct}%;height:100%;'></div></div>"
                    f"<div style='color:{_conf_color_card};font-size:1.4rem;font-weight:700;'>{_score:.1f}<span style='font-size:0.85rem;color:#868993;font-weight:400;'>/100</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Metrics row
                _tl1, _tl2, _tl3, _tl4, _tl5, _tl6 = st.columns(6)
                _tl1.metric("Entry",    f"₹{_mtf_res['entry']:.2f}",
                    help="Projected entry price — open of confirmation bar.")
                _tl2.metric("Stop",     f"₹{_mtf_res['stop_loss']:.2f}",
                    help=f"Hard stop-loss = Entry ± (ATR × {_mtf_sl_mult}). Place in bracket/cover order.")
                _tl3.metric("Target 1", f"₹{_mtf_res['targets'][0]:.2f}",
                    help=f"T1 = Entry ± (Risk × {_mtf_rr_t1}). Book 50-70% position here.")
                _tl4.metric("Target 2", f"₹{_mtf_res['targets'][1]:.2f}",
                    help=f"T2 = Entry ± (Risk × {_mtf_rr_t2}). Trail remaining position.")
                _tl5.metric("Leverage", f"{_mtf_res['leverage']:.1f}x",
                    help="Recommended MTF leverage. Capped at 2x if sector is misaligned.")
                _tl6.metric("Position size", f"{_mtf_res['position_size']} sh",
                    help="Shares to buy/sell given your capital and risk constraints.")

                st.caption(f"Risk per trade: ₹{_risk_inr:,.0f}  ({_risk_inr / _mtf_capital * 100:.2f}% of ₹{_mtf_capital:,} capital)")

                # ── Action buttons row ────────────────────────────────────────
                st.markdown("---")
                _btn_c1, _btn_c2, _btn_c3 = st.columns([2, 2, 1])

                # Auto-execute button
                with _btn_c1:
                    _cap_ok   = _can_auto_intraday if _trade_type == "intraday" else _can_auto_margin
                    _cap_used = _ai_count if _trade_type == "intraday" else _am_count
                    _cap_max  = MAX_AUTO_INTRADAY if _trade_type == "intraday" else MAX_AUTO_MARGIN
                    _cap_tag  = f"{_cap_used}/{_cap_max} used today"

                    if _dec == "avoid":
                        st.button("🤖 Auto-execute", disabled=True, key="mtf_auto_exec",
                                  help="Confidence too low — engine will not auto-execute this trade.")
                        st.caption("⛔ Auto-execute blocked: confidence < 60")
                    elif not _cap_ok:
                        st.button("🤖 Auto-execute", disabled=True, key="mtf_auto_exec",
                                  help=f"Daily cap reached ({_cap_tag}).")
                        st.caption(f"🚫 Daily {_tt_label} cap reached ({_cap_tag})")
                    else:
                        if st.button(f"🤖 Auto-execute  [{_cap_tag}]", key="mtf_auto_exec",
                                     type="primary",
                                     help=f"Automatically execute this {_tt_label} trade and send Telegram alert."):
                            # Increment cap counter
                            if _trade_type == "intraday":
                                st.session_state.mtf_auto_intraday_count += 1
                            else:
                                st.session_state.mtf_auto_margin_count += 1

                            # Log to lightweight trade log (for adaptive feedback)
                            st.session_state.mtf_trade_log.append({
                                "result"       : "pending",
                                "trade_type"   : _trade_type,
                                "action"       : _mtf_res["action"],
                                "entry"        : _mtf_res["entry"],
                                "stop"         : _mtf_res["stop_loss"],
                                "target1"      : _mtf_res["targets"][0],
                                "target2"      : _mtf_res["targets"][1],
                                "leverage"     : _mtf_res["leverage"],
                                "confidence"   : _mtf_res["confidence_score"],
                                "decision"     : _dec,
                                "auto"         : True,
                                "trend_score"  : _mtf_res["agent_outputs"]["htf"].get("strength", 50),
                                "setup_score"  : _mtf_res["agent_outputs"]["setup"].get("setup_quality", 50),
                                "conf_score"   : _mtf_res["agent_outputs"]["ltf"].get("confirmation_strength", 50),
                                "volume_ratio" : _mtf_res["agent_outputs"]["setup"].get("volume_ratio", 1.0),
                            })
                            # Create full MTFPaperTrade record
                            _mpt = MTFPaperTrade(
                                logged_at        = str(pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M IST")),
                                trade_type       = _trade_type,
                                action           = _mtf_res["action"],
                                bias             = _mtf_res["bias"],
                                confidence_score = _mtf_res["confidence_score"],
                                decision         = _dec,
                                leverage         = _mtf_res["leverage"],
                                entry            = _mtf_res["entry"],
                                stop             = _mtf_res["stop_loss"],
                                target1          = _mtf_res["targets"][0],
                                target2          = _mtf_res["targets"][1],
                                position_size    = _mtf_res["position_size"],
                                risk_per_trade   = _mtf_res["risk_per_trade"],
                                setup_type       = _mtf_res["agent_outputs"]["setup"].get("setup_type", "none"),
                                trend_score      = _mtf_res["agent_outputs"]["htf"].get("strength", 50),
                                setup_score      = _mtf_res["agent_outputs"]["setup"].get("setup_quality", 50),
                                conf_score       = _mtf_res["agent_outputs"]["ltf"].get("confirmation_strength", 50),
                                volume_ratio     = _mtf_res["agent_outputs"]["setup"].get("volume_ratio", 1.0),
                                track            = "auto",
                                notes            = _mtf_res["notes"],
                            )
                            st.session_state.mtf_paper_log.append(_mpt)

                            # Auto-send Telegram alert
                            _send_mtf_alert(auto=True)
                            st.success(f"✅ {_tt_label} trade auto-executed and alert sent! ({_cap_tag})")
                            st.rerun()

                # Manual send alert button
                with _btn_c2:
                    if st.button("📲 Send Telegram alert", key="mtf_tg_manual",
                                 help="Send this trade plan to Telegram manually (no auto-execute)."):
                        _send_mtf_alert(auto=False)

                # Reset daily caps
                with _btn_c3:
                    if st.button("🔄 Reset caps", key="mtf_reset_caps",
                                 help="Reset today's auto-execute trade counters."):
                        st.session_state.mtf_auto_intraday_count = 0
                        st.session_state.mtf_auto_margin_count   = 0
                        st.toast("Daily trade caps reset.", icon="🔄")
                        st.rerun()

                # Cap status bar
                st.markdown(
                    f"<div style='display:flex;gap:16px;margin-top:4px;'>"
                    f"<span style='font-size:0.78rem;color:{'#26a69a' if _ai_count < MAX_AUTO_INTRADAY else '#ef5350'};'>"
                    f"⚡ Intraday auto: {_ai_count}/{MAX_AUTO_INTRADAY}</span>"
                    f"<span style='font-size:0.78rem;color:{'#26a69a' if _am_count < MAX_AUTO_MARGIN else '#ef5350'};'>"
                    f"🏦 Margin auto: {_am_count}/{MAX_AUTO_MARGIN}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Agent breakdown ───────────────────────────────────────────
                st.markdown("#### Agent breakdown")
                _ag_cols = st.columns(4)
                _agents_info = [
                    ("HTF Trend",   _mtf_res["agent_outputs"]["htf"],
                     "trend", "strength",
                     "Higher timeframe EMA + market structure bias"),
                    ("Setup",       _mtf_res["agent_outputs"]["setup"],
                     "setup_type", "setup_quality",
                     "Pullback / Breakout / Reversal quality on daily chart"),
                    ("LTF Entry",   _mtf_res["agent_outputs"]["ltf"],
                     "entry_signal", "confirmation_strength",
                     "1H/15m candlestick + RSI + micro-structure timing"),
                    ("Risk Engine", _mtf_res["agent_outputs"]["risk"],
                     "leverage", "risk_score",
                     "Position size, leverage, stop & target optimisation"),
                ]
                for _col, (_name, _ag, _label_key, _score_key, _desc) in zip(_ag_cols, _agents_info):
                    _ag_score = _ag.get(_score_key, 0)
                    _ag_label = str(_ag.get(_label_key, "—"))
                    _ag_color = "#26a69a" if float(_ag_score) >= 60 else ("#f7c948" if float(_ag_score) >= 40 else "#ef5350")
                    _col.markdown(
                        f"<div style='background:#1e2130;border:1px solid #2a2e39;"
                        f"border-radius:8px;padding:12px 14px;height:140px;'>"
                        f"<div style='color:#868993;font-size:0.72rem;margin-bottom:4px;'>{_name}</div>"
                        f"<div style='color:{_ag_color};font-weight:700;font-size:1.1rem;'>{_ag_label}</div>"
                        f"<div style='color:#d1d4dc;font-size:0.85rem;margin:4px 0;'>{_ag_score}/100</div>"
                        f"<div style='background:#131722;border-radius:4px;height:6px;overflow:hidden;margin-bottom:6px;'>"
                        f"<div style='background:{_ag_color};width:{min(100,float(_ag_score))}%;height:100%;'></div></div>"
                        f"<div style='color:#4a4e5a;font-size:0.68rem;'>{_desc}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── HTF detail ────────────────────────────────────────────────
                with st.expander("HTF Trend Agent — detail", expanded=False):
                    _htf = _mtf_res["agent_outputs"]["htf"]
                    _hc1, _hc2, _hc3, _hc4 = st.columns(4)
                    _hc1.metric("EMA 20",  f"₹{_htf.get('ema20', 0):.2f}",
                        help="20-day EMA. Price > EMA20 = short-term bullish momentum.")
                    _hc2.metric("EMA 50",  f"₹{_htf.get('ema50', 0):.2f}",
                        help="50-day EMA. Primary trend filter. EMA20 > EMA50 = bullish alignment.")
                    _hc3.metric("EMA 200", f"₹{_htf.get('ema200', 0):.2f}",
                        help="200-day EMA. Long-term bull/bear market divider.")
                    _hc4.metric("Structure", _htf.get("structure", "—"),
                        help="HH_HL = Higher Highs & Higher Lows (uptrend); LH_LL = downtrend; mixed = choppy.")
                    _hs1, _hs2 = st.columns(2)
                    _hs1.metric("Support",    f"₹{_htf.get('key_levels', {}).get('support', 0):.2f}",
                        help="Recent swing low — key buying zone for long setups.")
                    _hs2.metric("Resistance", f"₹{_htf.get('key_levels', {}).get('resistance', 0):.2f}",
                        help="Recent swing high — breakout above triggers fresh LONG opportunity.")
                    if _htf.get("log"):
                        st.code("\n".join(_htf["log"]), language="text")

                # ── Setup detail ──────────────────────────────────────────────
                with st.expander("Setup Detection Agent — detail", expanded=False):
                    _sd = _mtf_res["agent_outputs"]["setup"]
                    _sc1, _sc2, _sc3 = st.columns(3)
                    _sc1.metric("Setup type",   _sd.get("setup_type", "—"),
                        help="pullback = retrace to EMA in uptrend; breakout = range break + volume; reversal = support bounce.")
                    _sc2.metric("RSI(14)",       f"{_sd.get('rsi', 0):.1f}",
                        help="Daily RSI. 40-50 in uptrend = healthy pullback. <30 = oversold reversal candidate.")
                    _sc3.metric("Volume ratio", f"{_sd.get('volume_ratio', 1):.2f}x",
                        help="Current volume ÷ 20-day avg. ≥1.5x on breakout = confirmed. <0.8x on pullback = dry.")
                    _tz = _sd.get("trigger_zone", {})
                    st.info(f"Trigger zone: ₹{_tz.get('low', 0):.2f} — ₹{_tz.get('high', 0):.2f} | Candle: {_sd.get('candle_pattern', '—')}")
                    if _sd.get("log"):
                        st.code("\n".join(_sd["log"]), language="text")

                # ── LTF detail ────────────────────────────────────────────────
                with st.expander("LTF Entry Agent — detail", expanded=False):
                    _ltf = _mtf_res["agent_outputs"]["ltf"]
                    _lc1, _lc2, _lc3 = st.columns(3)
                    _lc1.metric("LTF RSI",    f"{_ltf.get('ltf_rsi', 0):.1f}",
                        help="RSI on LTF chart. BUY: RSI crossing >40. SELL: RSI crossing <60.")
                    _lc2.metric("Candle",     _ltf.get("ltf_candle", "—"),
                        help="Last LTF candle pattern. bullish_engulfing/hammer = long confirmation.")
                    _lc3.metric("Micro BO",   str(_ltf.get("micro_breakout", False)),
                        help="True if price broke above the 5-bar high (LONG) or below 5-bar low (SHORT).")
                    st.metric("Invalidation level", f"₹{_ltf.get('invalid_if', 0):.2f}",
                        help="If price reaches this level before entry fills, the setup is invalidated.")
                    if _ltf.get("log"):
                        st.code("\n".join(_ltf["log"]), language="text")

                # ── Risk detail ───────────────────────────────────────────────
                with st.expander("Risk + Leverage Agent — detail", expanded=False):
                    _rk = _mtf_res["agent_outputs"]["risk"]
                    _rc1, _rc2, _rc3, _rc4 = st.columns(4)
                    _rc1.metric("Alloc %",   f"{_rk.get('capital_allocation_pct', 0):.1f}%",
                        help="% of capital to deploy. Higher confidence → more capital allocated.")
                    _rc2.metric("Leverage",  f"{_rk.get('leverage', 1):.2f}x",
                        help="MTF leverage multiplier. Regime-capped at 2x if Nifty Metal is weak.")
                    _rc3.metric("Risk ₹",    f"₹{_rk.get('risk_per_trade_inr', 0):,.0f}",
                        help="Absolute ₹ at risk = position size × risk per share.")
                    _rc4.metric("Regime cap", str(_rk.get("regime_capped", False)),
                        help="True when Nifty Metal sector is misaligned → leverage capped at 2x.")
                    if _rk.get("log"):
                        st.code("\n".join(_rk["log"]), language="text")

                # ── Adaptive feedback loop ────────────────────────────────────
                st.markdown("#### Record trade outcome (adaptive feedback)")
                st.caption("Mark this trade result to update indicator weights for future signals.")
                _fb_col1, _fb_col2, _fb_col3 = st.columns([1, 1, 2])
                _fb_result = _fb_col1.selectbox("Outcome", ["—", "win", "loss", "breakeven"], key="mtf_fb_result")
                if _fb_col2.button("Submit feedback", key="mtf_fb_submit"):
                    if _fb_result in ("win", "loss"):
                        _pillar_scores = {
                            "trend" : _mtf_res["agent_outputs"]["htf"].get("strength", 50) / 100,
                            "setup" : _mtf_res["agent_outputs"]["setup"].get("setup_quality", 50) / 100,
                            "conf"  : _mtf_res["agent_outputs"]["ltf"].get("confirmation_strength", 50) / 100,
                            "volume": min(1.0, _mtf_res["agent_outputs"]["setup"].get("volume_ratio", 1.0) / 2.0),
                        }
                        _new_aw = st.session_state.mtf_adaptive_weights.update(
                            _fb_result, _pillar_scores
                        )
                        st.session_state.mtf_adaptive_weights = _new_aw
                        _log_entry = {
                            "result"       : _fb_result,
                            "trend_score"  : _mtf_res["agent_outputs"]["htf"].get("strength", 50),
                            "setup_score"  : _mtf_res["agent_outputs"]["setup"].get("setup_quality", 50),
                            "conf_score"   : _mtf_res["agent_outputs"]["ltf"].get("confirmation_strength", 50),
                            "volume_ratio" : _mtf_res["agent_outputs"]["setup"].get("volume_ratio", 1.0),
                            "confidence"   : _mtf_res["confidence_score"],
                            "action"       : _mtf_res["action"],
                        }
                        st.session_state.mtf_trade_log.append(_log_entry)
                        st.toast(f"Feedback recorded ({_fb_result}). Weights updated.", icon="🧠")
                        st.rerun()
                    else:
                        st.warning("Select 'win' or 'loss' to update weights.")

                # ── MTF adaptive feedback trade history (compact) ─────────────
                if st.session_state.mtf_trade_log:
                    with st.expander(f"Adaptive feedback log ({len(st.session_state.mtf_trade_log)} entries)", expanded=False):
                        _mtf_log_df = pd.DataFrame(st.session_state.mtf_trade_log)
                        st.dataframe(_mtf_log_df, use_container_width=True, hide_index=True)
                        _wins  = sum(1 for t in st.session_state.mtf_trade_log if t.get("result") == "win")
                        _total = len(st.session_state.mtf_trade_log)
                        if _total:
                            st.metric("Feedback win rate", f"{_wins}/{_total} = {_wins/_total*100:.1f}%")
                        if st.button("Clear adaptive log", key="mtf_clear_log"):
                            st.session_state.mtf_trade_log = []
                            st.session_state.mtf_adaptive_weights = AdaptiveWeights()
                            st.rerun()

            # ════════════════════════════════════════════════════════════════
            # SECTION A — MTF PAPER TRADING
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📋 MTF Paper Trading")
            st.caption(
                "Paper-trade MTF engine signals. "
                "Add manually from the current analysis result above, "
                "or they are added automatically when you click Auto-execute."
            )

            _pt_log = st.session_state.mtf_paper_log

            # Manual add from last MTF result
            _cur_res = st.session_state.get("mtf_last_result")
            if _cur_res and _cur_res.get("action") != "wait":
                _pt_add_col1, _pt_add_col2 = st.columns([3, 1])
                _pt_add_col1.caption(
                    f"Current signal: **{_cur_res['action'].upper()}** @ "
                    f"₹{_cur_res['entry']:.2f}  |  Conf {_cur_res['confidence_score']:.1f}/100  |  "
                    f"SL ₹{_cur_res['stop_loss']:.2f}  |  T1 ₹{_cur_res['targets'][0]:.2f}"
                )
                _pt_track = _pt_add_col2.selectbox("Track", ["manual", "auto"], key="pt_track_sel")
                if st.button("➕ Add to paper trades", key="mtf_pt_add"):
                    _lev_pt = _cur_res["leverage"]
                    _tt_pt  = "intraday" if _lev_pt <= 1.0 else "margin"
                    _new_pt = MTFPaperTrade(
                        logged_at        = str(pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M IST")),
                        trade_type       = _tt_pt,
                        action           = _cur_res["action"],
                        bias             = _cur_res["bias"],
                        confidence_score = _cur_res["confidence_score"],
                        decision         = _cur_res["decision"],
                        leverage         = _lev_pt,
                        entry            = _cur_res["entry"],
                        stop             = _cur_res["stop_loss"],
                        target1          = _cur_res["targets"][0],
                        target2          = _cur_res["targets"][1],
                        position_size    = _cur_res["position_size"],
                        risk_per_trade   = _cur_res["risk_per_trade"],
                        setup_type       = _cur_res["agent_outputs"]["setup"].get("setup_type", "none"),
                        trend_score      = _cur_res["agent_outputs"]["htf"].get("strength", 50),
                        setup_score      = _cur_res["agent_outputs"]["setup"].get("setup_quality", 50),
                        conf_score       = _cur_res["agent_outputs"]["ltf"].get("confirmation_strength", 50),
                        volume_ratio     = _cur_res["agent_outputs"]["setup"].get("volume_ratio", 1.0),
                        track            = _pt_track,
                        notes            = _cur_res["notes"],
                    )
                    st.session_state.mtf_paper_log.append(_new_pt)
                    st.success("Added to MTF paper log.")
                    st.rerun()

            # Resolve open trades against price panel
            _open_trades = [t for t in _pt_log if t.status == "open"]
            if _open_trades and not prices.empty and "MOIL" in prices.columns:
                if st.button("🔄 Resolve open trades against price panel", key="mtf_pt_resolve"):
                    _moil_fwd = prices[["MOIL"]].dropna().copy()
                    _moil_fwd.columns = ["close"]
                    _moil_fwd["open"]  = _moil_fwd["close"]
                    _moil_fwd["high"]  = _moil_fwd["close"] * 1.005
                    _moil_fwd["low"]   = _moil_fwd["close"] * 0.995
                    for _ot in _open_trades:
                        resolve_mtf_paper_trade(_ot, _moil_fwd, margin_bars=10)
                    st.success(f"Resolved {len(_open_trades)} open trade(s).")
                    st.rerun()

            if not _pt_log:
                st.info("No paper trades yet. Run the engine and click Auto-execute, or add manually above.")
            else:
                _pt_df = mtf_paper_log_to_df(_pt_log)

                # ── Summary metrics ───────────────────────────────────────────
                _pa_auto   = summarise_mtf_track(_pt_df, "auto")
                _pa_manual = summarise_mtf_track(_pt_df, "manual")
                _ps1, _ps2 = st.columns(2)
                with _ps1:
                    st.markdown(
                        "<div style='background:#1a3a2a;border:1px solid #26a69a;"
                        "border-radius:8px;padding:14px 18px;'>"
                        "<span style='color:#26a69a;font-weight:700;font-size:0.9rem;'>🤖 AUTO TRACK</span>",
                        unsafe_allow_html=True,
                    )
                    _pa1, _pa2, _pa3 = st.columns(3)
                    _pa1.metric("Trades",    _pa_auto["trades"])
                    _pa2.metric("Win rate",  f"{_pa_auto['win_rate']:.1%}")
                    _pa3.metric("Avg R",     f"{_pa_auto['avg_r']:.2f}R")
                    st.metric("Total PnL (pts)", f"{_pa_auto['total_pnl']:+.2f}")
                    st.metric("Avg Confidence",  f"{_pa_auto['avg_conf']:.1f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with _ps2:
                    st.markdown(
                        "<div style='background:#2a2a1a;border:1px solid #f7c948;"
                        "border-radius:8px;padding:14px 18px;'>"
                        "<span style='color:#f7c948;font-weight:700;font-size:0.9rem;'>👤 MANUAL TRACK</span>",
                        unsafe_allow_html=True,
                    )
                    _pm1, _pm2, _pm3 = st.columns(3)
                    _pm1.metric("Trades",    _pa_manual["trades"])
                    _pm2.metric("Win rate",  f"{_pa_manual['win_rate']:.1%}")
                    _pm3.metric("Avg R",     f"{_pa_manual['avg_r']:.2f}R")
                    st.metric("Total PnL (pts)", f"{_pa_manual['total_pnl']:+.2f}")
                    st.metric("Avg Confidence",  f"{_pa_manual['avg_conf']:.1f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

                # ── Full paper trade log table ─────────────────────────────────
                _disp_pt_cols = [c for c in [
                    "id", "logged_at", "trade_type", "action", "confidence_score",
                    "decision", "leverage", "entry", "stop", "target1", "target2",
                    "position_size", "track", "status",
                    "exit_price_t1", "exit_price_t2", "exit_price_sl",
                    "exit_reason", "pnl_points", "r_multiple",
                ] if c in _pt_df.columns]
                _pt_disp = _pt_df[_disp_pt_cols].copy()
                for _fc in ["entry", "stop", "target1", "target2", "exit_price_t1", "exit_price_t2", "exit_price_sl"]:
                    if _fc in _pt_disp.columns:
                        _pt_disp[_fc] = pd.to_numeric(_pt_disp[_fc], errors="coerce").map(
                            lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
                if "pnl_points" in _pt_disp.columns:
                    _pt_disp["pnl_points"] = pd.to_numeric(_pt_disp["pnl_points"], errors="coerce").map(
                        lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
                if "r_multiple" in _pt_disp.columns:
                    _pt_disp["r_multiple"] = pd.to_numeric(_pt_disp["r_multiple"], errors="coerce").map(
                        lambda x: f"{x:+.2f}R" if pd.notna(x) else "—")

                st.dataframe(_pt_disp, use_container_width=True, hide_index=True)

                _ptbuf = io.StringIO(); _pt_df.to_csv(_ptbuf, index=False)
                _ptc1, _ptc2 = st.columns([1, 4])
                _ptc1.download_button("⬇ Paper log CSV", _ptbuf.getvalue(),
                                      file_name="mtf_paper_trades.csv", mime="text/csv")
                if _ptc2.button("🗑️ Clear paper log", key="mtf_pt_clear"):
                    st.session_state.mtf_paper_log = []
                    st.rerun()

            # ════════════════════════════════════════════════════════════════
            # SECTION B — MTF BACKTEST
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 🔬 MTF Walk-Forward Backtest")
            st.caption(
                "Runs the MTF engine on every daily bar (rolling window), "
                "simulates multi-target exits, and reports cumulative performance."
            )

            _bt_c1, _bt_c2, _bt_c3, _bt_c4 = st.columns(4)
            _bt_lookback   = _bt_c1.number_input("Lookback bars", 60, 500, 120, 10,
                help="Minimum history bars before the engine starts generating signals.")
            _bt_fwd_bars   = _bt_c2.number_input("Forward bars (hold)", 1, 30, 5, 1,
                help="How many bars ahead to simulate exit (T1/T2/SL/EOD).")
            _bt_min_conf   = _bt_c3.slider("Min confidence to trade", 50.0, 90.0, 60.0, 5.0,
                help="Skip signals below this score.")
            _bt_adaptive   = _bt_c4.checkbox("Adaptive weights", value=True,
                help="Update weights after each resolved trade (reinforcement learning).")
            # Fix #11: cost parameters
            with st.expander("💸 Transaction cost settings", expanded=False):
                _btcost_c1, _btcost_c2 = st.columns(2)
                _bt_total_cost = _btcost_c1.number_input(
                    "Total cost % per side (brokerage + STT + slippage)",
                    min_value=0.0, max_value=2.0, value=0.08, step=0.01, format="%.2f",
                    help="All-in cost on each trade leg as % of traded amount.\n\n"
                         "Example: buy 10 shares @ ₹300 = ₹3,000 + sell @ ₹310 = ₹3,100\n"
                         "At 0.08%: cost = 3000×0.08% + 3100×0.08% = ₹2.40 + ₹2.48 = ₹4.88 total\n\n"
                         "Typical breakdown per side:\n"
                         "  • Brokerage: ~0.03%\n"
                         "  • STT (delivery buy+sell): ~0.1% / (intraday sell): ~0.025%\n"
                         "  • Market impact / spread: ~0.02%\n\n"
                         "For positional / MTF trades use 0.15–0.20% (higher STT applies).")
                _bt_mtf_rate   = _btcost_c2.number_input(
                    "MTF / leveraged interest rate % p.a.",
                    min_value=0.0, max_value=36.0, value=18.0, step=0.5, format="%.1f",
                    help="Annual interest charged on the leveraged portion of MTF positions.\n\n"
                         "Formula applied per trade:\n"
                         "  Interest = Trade value × (rate% / 365) × days held\n\n"
                         "Example: ₹3,000 trade held 5 days @ 18% p.a.\n"
                         "  = 3000 × (18/365/100) × 5 = ₹7.40\n\n"
                         "Typical India broker MTF rate: 18–24% p.a.")
                _bt_brokerage = _bt_total_cost   # alias for downstream compatibility
                _bt_slippage  = 0.0              # already included in total cost above

            if st.button("▶️ Run MTF Backtest", key="mtf_run_bt", type="primary"):
                # Fetch 5y daily OHLCV: TradingView → Dhan → Zerodha → yfinance
                with st.spinner("Fetching MOIL.NS daily OHLCV (TradingView → Dhan → Zerodha → yfinance)…"):
                    _moil_bt_df, _bt_src, _bt_err = fetch_daily("MOIL.NS", years=5)
                if not _moil_bt_df.empty:
                    st.caption(f"Backtest data: {source_badge(_bt_src)} — {len(_moil_bt_df)} bars")
                else:
                    if _bt_err:
                        st.warning(f"All feeds failed: {_bt_err}. Falling back to price panel.")
                    if prices.empty or "MOIL" not in prices.columns:
                        st.error("Load market data first (MOIL daily prices required).")
                        _moil_bt_df = None
                    else:
                        st.warning("Using close-only fallback — ATR/candle signals will be inaccurate.")
                        _moil_bt_close = prices["MOIL"].dropna()
                        _moil_bt_df = pd.DataFrame({
                            "open": _moil_bt_close, "high": _moil_bt_close,
                            "low":  _moil_bt_close, "close": _moil_bt_close,
                            "volume": float("nan"),
                        })

                if _moil_bt_df is not None and not _moil_bt_df.empty:
                    _sec_bt = None
                    if "NIFTY Metal" in prices.columns:
                        _sec_bt = prices[["NIFTY Metal"]].rename(columns={"NIFTY Metal": "close"})
                    with st.spinner(f"Running backtest on {len(_moil_bt_df)} bars…"):
                        _bt_res = run_mtf_backtest(
                            daily_df       = _moil_bt_df,
                            sector_df      = _sec_bt,
                            capital_inr    = float(_mtf_capital),
                            max_risk_pct   = float(_mtf_risk_pct),
                            lookback       = int(_bt_lookback),
                            forward_bars   = int(_bt_fwd_bars),
                            min_confidence = float(_bt_min_conf),
                            adaptive       = bool(_bt_adaptive),
                            brokerage_pct  = float(_bt_brokerage),
                            slippage_pct   = float(_bt_slippage),
                            mtf_rate_pa    = float(_bt_mtf_rate),
                        )
                    st.session_state.mtf_backtest_result = _bt_res
                    if "error" in _bt_res:
                        st.error(_bt_res["error"])
                    else:
                        st.success(f"Backtest complete: {_bt_res['metrics']['trades_taken']} trades taken.")

            _bt = st.session_state.mtf_backtest_result
            if _bt and "metrics" in _bt and not _bt.get("error"):
                _bm = _bt["metrics"]

                # ── Key metrics ───────────────────────────────────────────────
                st.markdown("#### Backtest results")
                _bm1, _bm2, _bm3, _bm4, _bm5, _bm6 = st.columns(6)
                _bm1.metric("Trades",        _bm["trades_closed"],
                    help="Total closed trades (reached T1/T2/SL/EOD).")
                _bm2.metric("Win rate",       f"{_bm['win_rate']:.1%}",
                    help="% of closed trades with positive PnL.")
                _bm3.metric("Total PnL (pts)", f"{_bm['total_pnl_pts']:+.2f}",
                    help="Sum of all PnL in price points per share.")
                _bm4.metric("Avg R",          f"{_bm['avg_r']:.2f}R",
                    help="Average R-multiple per trade. >1.0 means avg winner > 1× risk.")
                _bm5.metric("Profit factor",  str(_bm["profit_factor"]),
                    help="Gross profit ÷ gross loss. >1.5 = good, >2.0 = excellent.")
                _bm6.metric("Max drawdown",   f"{_bm['max_dd_pts']:.2f} pts",
                    help="Worst peak-to-trough decline in cumulative PnL.")

                _ba1, _ba2 = st.columns(2)
                _ba1.metric("Avg confidence", f"{_bm['avg_confidence']:.1f}",
                    help="Mean confidence score of traded signals.")
                _ba2.metric("Total signals scanned", _bm["total_signals"],
                    help="Number of bars the engine evaluated (includes skipped low-confidence).")

                # ── Equity curve ──────────────────────────────────────────────
                _eq = _bt["equity_curve"]
                if not _eq.empty:
                    import plotly.graph_objects as _go_bt
                    _eq_fig = _go_bt.Figure()
                    _eq_fig.add_trace(_go_bt.Scatter(
                        x=list(range(len(_eq))), y=_eq.values,
                        mode="lines", name="Cumulative PnL",
                        line=dict(color="#26a69a", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(38,166,154,0.08)",
                    ))
                    _eq_fig.add_hline(y=0, line_dash="dot", line_color="#ef5350", line_width=1)
                    _eq_fig.update_layout(
                        title="MTF Engine — Equity Curve (pts/share)",
                        xaxis_title="Trade #",
                        yaxis_title="Cumulative PnL (pts)",
                        template="plotly_dark",
                        height=320,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(_eq_fig, use_container_width=True)

                # ── Trade table ───────────────────────────────────────────────
                with st.expander("Backtest trade log", expanded=False):
                    _bt_disp_cols = [c for c in [
                        "logged_at", "trade_type", "action", "confidence_score",
                        "leverage", "entry", "stop", "target1", "target2",
                        "status", "exit_reason", "pnl_points", "r_multiple",
                    ] if c in _bt["trades_df"].columns]
                    _bt_disp = _bt["trades_df"][_bt_disp_cols].copy()
                    for _fc in ["entry", "stop", "target1", "target2"]:
                        if _fc in _bt_disp.columns:
                            _bt_disp[_fc] = pd.to_numeric(_bt_disp[_fc], errors="coerce").map(
                                lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
                    if "pnl_points" in _bt_disp.columns:
                        _bt_disp["pnl_points"] = pd.to_numeric(_bt_disp["pnl_points"], errors="coerce").map(
                            lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
                    if "r_multiple" in _bt_disp.columns:
                        _bt_disp["r_multiple"] = pd.to_numeric(_bt_disp["r_multiple"], errors="coerce").map(
                            lambda x: f"{x:+.2f}R" if pd.notna(x) else "—")
                    st.dataframe(_bt_disp, use_container_width=True, hide_index=True)
                    _bt_buf = io.StringIO(); _bt["trades_df"].to_csv(_bt_buf, index=False)
                    st.download_button("⬇ Backtest CSV", _bt_buf.getvalue(),
                                       file_name="mtf_backtest.csv", mime="text/csv")

                # ── Final adaptive weights after backtest ─────────────────────
                with st.expander("Final adaptive weights after backtest", expanded=False):
                    _fw = _bm.get("final_weights", {})
                    _fw1, _fw2, _fw3, _fw4 = st.columns(4)
                    _fw1.metric("Trend",  f"{_fw.get('trend_weight',  0):.3f}")
                    _fw2.metric("Setup",  f"{_fw.get('setup_weight',  0):.3f}")
                    _fw3.metric("Conf",   f"{_fw.get('conf_weight',   0):.3f}")
                    _fw4.metric("Volume", f"{_fw.get('volume_weight', 0):.3f}")
                    if st.button("Apply backtest weights to live engine", key="mtf_apply_bt_weights"):
                        from intraday_vwap_orb.mtf.adaptive_layer import AdaptiveWeights as _AW
                        st.session_state.mtf_adaptive_weights = _AW.from_dict(_fw)
                        st.toast("Backtest-calibrated weights applied to live engine.", icon="✅")

            # ════════════════════════════════════════════════════════════════
            # SECTION C — MTF VALIDATION TRACKER
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### ✅ MTF Validation Tracker")
            st.caption(
                "Breaks down performance by confidence band — "
                "shows whether higher-confidence signals actually deliver better outcomes."
            )

            # Source: prefer backtest data; fall back to paper log
            _val_source_df = None
            if _bt and "trades_df" in _bt and not _bt["trades_df"].empty:
                _val_source_df = _bt["trades_df"]
                st.caption("Using backtest data. Run backtest above to refresh.")
            elif _pt_log:
                _val_source_df = mtf_paper_log_to_df(_pt_log)
                st.caption("Using live paper trade log (no backtest run yet).")

            if _val_source_df is not None and not _val_source_df.empty:
                _vt = build_validation_table(_val_source_df)
                if not _vt.empty:
                    st.dataframe(_vt, use_container_width=True, hide_index=True)

                    # Bar chart: win rate by confidence band
                    _vt_chart = _vt[_vt["Trades"] > 0].copy()
                    _vt_chart["Win rate %"] = _vt_chart["Win rate"].str.replace("%", "").apply(
                        pd.to_numeric, errors="coerce")
                    if not _vt_chart.empty and _vt_chart["Win rate %"].notna().any():
                        import plotly.express as _px_vt
                        _wr_fig = _px_vt.bar(
                            _vt_chart, x="Conf band", y="Win rate %",
                            color="Win rate %",
                            color_continuous_scale=["#ef5350", "#f7c948", "#26a69a"],
                            title="Win Rate by Confidence Band",
                            text="Win rate %",
                        )
                        _wr_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        _wr_fig.update_layout(template="plotly_dark", height=300,
                                              margin=dict(l=40, r=20, t=40, b=40),
                                              showlegend=False)
                        st.plotly_chart(_wr_fig, use_container_width=True)

                # Breakdown by trade type
                st.markdown("##### By trade type (Intraday vs Margin)")
                _vt_tt_rows = []
                for _tt_v in ["intraday", "margin"]:
                    _tt_sub = _val_source_df[_val_source_df["trade_type"] == _tt_v].copy() \
                        if "trade_type" in _val_source_df.columns else pd.DataFrame()
                    if _tt_sub.empty:
                        continue
                    _s = summarise_mtf_track(_tt_sub.assign(track="x"), "x") if not _tt_sub.empty else {}
                    _vt_tt_rows.append({
                        "Type"       : _tt_v.title(),
                        "Trades"     : _s.get("trades", 0),
                        "Win rate"   : f"{_s.get('win_rate', 0):.1%}",
                        "Avg R"      : f"{_s.get('avg_r', 0):.2f}R",
                        "Total PnL"  : f"{_s.get('total_pnl', 0):+.2f} pts",
                        "Avg conf"   : f"{_s.get('avg_conf', 0):.1f}",
                    })
                if _vt_tt_rows:
                    st.dataframe(pd.DataFrame(_vt_tt_rows), use_container_width=True, hide_index=True)

                # Setup type breakdown
                st.markdown("##### By setup type")
                if "setup_type" in _val_source_df.columns and "pnl_points" in _val_source_df.columns:
                    _setup_grp = _val_source_df[_val_source_df["status"] == "closed"].copy()
                    _setup_grp["pnl_points"] = pd.to_numeric(_setup_grp["pnl_points"], errors="coerce")
                    _setup_grp["r_multiple"] = pd.to_numeric(_setup_grp["r_multiple"], errors="coerce")
                    _setup_summary = (
                        _setup_grp.groupby("setup_type")
                        .agg(
                            trades    = ("pnl_points", "count"),
                            total_pnl = ("pnl_points", "sum"),
                            avg_r     = ("r_multiple", "mean"),
                            win_rate  = ("pnl_points", lambda x: (x > 0).mean()),
                        )
                        .reset_index()
                        .rename(columns={"setup_type": "Setup"})
                    )
                    for _col_fmt in ["total_pnl", "avg_r", "win_rate"]:
                        if _col_fmt == "total_pnl":
                            _setup_summary[_col_fmt] = _setup_summary[_col_fmt].map(lambda x: f"{x:+.2f}")
                        elif _col_fmt == "avg_r":
                            _setup_summary[_col_fmt] = _setup_summary[_col_fmt].map(lambda x: f"{x:.2f}R")
                        elif _col_fmt == "win_rate":
                            _setup_summary[_col_fmt] = _setup_summary[_col_fmt].map(lambda x: f"{x:.1%}")
                    st.dataframe(_setup_summary, use_container_width=True, hide_index=True)

                st.markdown("##### Suggested calibration actions")
                st.write(
                    "- **If win rate < 40% at 60–70 band** → raise `Min confidence` slider to 70+\n"
                    "- **If leverage trades underperform** → reduce max leverage or tighten regime cap\n"
                    "- **If pullback > breakout** → check that volume filter is set ≥ 1.5×\n"
                    "- **Run backtest monthly** and apply final weights to live engine"
                )
            else:
                st.info("Run the backtest or add paper trades first to see validation stats.")

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 MODEL EVIDENCE TAB
# ═══════════════════════════════════════════════════════════════════════════════
_t_ev = _tab("📊 Model Evidence")
if _t_ev is not None:
    with _t_ev:
        st.subheader("📊 Model Evidence Dashboard")
        st.caption(
            "All metrics are derived from paper trades and backtests logged in this session. "
            "Click **Refresh** to recompute from the latest data."
        )

        _ev_col_r, _ev_col_b = st.columns([5, 1])
        with _ev_col_b:
            _ev_refresh = st.button("🔄 Refresh metrics", key="ev_refresh", type="primary")

        _ev_intra_tab, _ev_mtf_tab = st.tabs([
            "📊 Intraday ORB Engine", "🧠 MTF Adaptive Engine"
        ])

        # ── helper to colour-code metric delta ───────────────────────────────
        def _pct_fmt(v: float) -> str:
            return f"{v:.1%}"

        def _colour_wr(wr: float) -> str:
            if wr >= 0.60:  return "🟢"
            if wr >= 0.50:  return "🟡"
            return "🔴"

        # ── INTRADAY sub-tab ─────────────────────────────────────────────────
        with _ev_intra_tab:
            _intra_ev = compute_intraday_evidence(st.session_state.paper_log)

            if _intra_ev["error"]:
                st.info(_intra_ev["error"])
                st.caption(
                    "To populate this tab: go to **📈 Intraday → Intraday ORB** and "
                    "execute paper trades (auto or manual) against historical days."
                )
            else:
                _is = _intra_ev["summary"]
                _wr = _is["win_rate"]

                # ── Top-line KPI row ─────────────────────────────────────────
                st.markdown("### Key Performance Indicators")
                _ki1, _ki2, _ki3, _ki4, _ki5, _ki6 = st.columns(6)
                _ki1.metric("Total trades",     str(_is["total_trades"]))
                _ki2.metric("Win rate",         _pct_fmt(_wr),
                            delta=f"{_wr - 0.50:+.1%} vs 50%",
                            delta_color="normal")
                _ki3.metric("Avg R-multiple",   f"{_is['avg_r']:.2f}R",
                            delta=f"{_is['avg_r'] - 1.0:+.2f}R vs 1R",
                            delta_color="normal")
                _ki4.metric("Profit factor",    str(_is["profit_factor"]),
                            delta="≥1.5 target",
                            delta_color="off")
                _ki5.metric("Expectancy",       f"{_is['expectancy_pts']:+.3f} pts/trade")
                _ki6.metric("Max drawdown",     f"{_is['max_dd']:.3f} pts")

                _ki7, _ki8, _ki9, _ki10 = st.columns(4)
                _ki7.metric("Total PnL (pts)",  f"{_is['total_pnl_pts']:+.3f}")
                _ki8.metric("Wins",             str(_is["wins"]))
                _ki9.metric("Losses",           str(_is["losses"]))
                _ki10.metric("Max consec. loss", str(_is["max_consec_loss"]))

                st.markdown(f"> **Edge status: {_colour_wr(_wr)} {'Positive edge' if _wr >= 0.55 else 'Marginal' if _wr >= 0.50 else 'No edge detected'}** — based on {_is['total_trades']} closed trades")

                st.markdown("---")

                # ── Auto vs Manual comparison ────────────────────────────────
                st.markdown("### 🤖 Auto-approved vs 👤 Manual-approved")
                _bt = _intra_ev["by_track"]
                _tm1, _tm2 = st.columns(2)
                for _trk_label, _col in [("🤖 Auto", _tm1), ("👤 Manual", _tm2)]:
                    _trk = "auto" if "Auto" in _trk_label else "manual"
                    _ts  = _bt.get(_trk, {})
                    with _col:
                        st.markdown(f"**{_trk_label}**")
                        if _ts.get("trades", 0) == 0:
                            st.caption("No trades in this track.")
                        else:
                            _col.metric("Trades", _ts["trades"])
                            _col.metric("Win rate", _pct_fmt(_ts["win_rate"]),
                                        delta=f"{_ts['win_rate']-0.50:+.1%} vs 50%",
                                        delta_color="normal")
                            _col.metric("Profit factor", str(_ts["profit_factor"]))
                            _col.metric("Total PnL", f"{_ts['total_pnl']:+.3f} pts")

                st.markdown("---")

                # ── Rolling win rate chart ───────────────────────────────────
                st.markdown("### 📈 Rolling 10-trade Win Rate")
                _rwr = _intra_ev["rolling_wr"]
                if not _rwr.empty:
                    import plotly.graph_objects as _go_ev
                    _rwr_fig = _go_ev.Figure()
                    _rwr_fig.add_trace(_go_ev.Scatter(
                        y=_rwr.values, mode="lines+markers",
                        line=dict(color="#26a69a", width=2),
                        marker=dict(size=5),
                        name="Rolling win rate",
                        fill="tozeroy", fillcolor="rgba(38,166,154,0.12)",
                    ))
                    _rwr_fig.add_hline(y=0.5, line_dash="dash", line_color="#ef5350",
                                       annotation_text="50% breakeven")
                    _rwr_fig.add_hline(y=0.6, line_dash="dot", line_color="#26a69a",
                                       annotation_text="60% target")
                    _rwr_fig.update_layout(
                        template="plotly_dark", height=280,
                        paper_bgcolor="#131722", plot_bgcolor="#131722",
                        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#2a2e39"),
                        xaxis=dict(title="Trade #", gridcolor="#2a2e39"),
                        title="Win rate stability (rolling 10 trades)",
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    st.plotly_chart(_rwr_fig, use_container_width=True)

                # ── Confidence calibration ───────────────────────────────────
                st.markdown("### 🎯 Confidence Score Calibration")
                st.caption("Does a higher confidence score actually produce more wins? This table proves it.")
                _cal = _intra_ev["confidence_cal"]
                if not _cal.empty:
                    st.dataframe(
                        _cal,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Edge": st.column_config.TextColumn("Edge", width="small"),
                            "Win %": st.column_config.TextColumn("Win %", width="small"),
                        }
                    )
                    _cal_fig = _go_ev.Figure()
                    _win_pcts = []
                    for _, row in _cal.iterrows():
                        try:
                            _win_pcts.append(float(str(row["Win %"]).replace("%", "")) / 100)
                        except Exception:
                            _win_pcts.append(0.0)
                    _cal_fig.add_trace(_go_ev.Bar(
                        x=_cal["Conf band"].tolist(),
                        y=_win_pcts,
                        marker_color=["#ef5350" if w < 0.5 else "#ffa726" if w < 0.6 else "#26a69a" for w in _win_pcts],
                        text=[f"{w:.1%}" for w in _win_pcts],
                        textposition="outside",
                    ))
                    _cal_fig.add_hline(y=0.5, line_dash="dash", line_color="#ef5350",
                                       annotation_text="50%")
                    _cal_fig.update_layout(
                        template="plotly_dark", height=260,
                        paper_bgcolor="#131722", plot_bgcolor="#131722",
                        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#2a2e39"),
                        xaxis=dict(gridcolor="#2a2e39"),
                        title="Win rate by confidence band",
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    st.plotly_chart(_cal_fig, use_container_width=True)

                st.markdown("---")

                # ── Signal funnel ────────────────────────────────────────────
                st.markdown("### 🔽 Signal Quality Funnel")
                _fn = _intra_ev["signal_funnel"]
                _fc1, _fc2, _fc3, _fc4 = st.columns(4)
                _fc1.metric("Signals generated",  _fn.get("signals_generated", 0))
                _fc2.metric("Trades closed",       _fn.get("trades_closed", 0))
                _fc3.metric("Wins",                _fn.get("wins", 0))
                _fc4.metric("Open / Pending",      _fn.get("open_or_pending", 0))

                st.markdown("---")

                # ── By side and by time ──────────────────────────────────────
                _bs_col, _bt_col = st.columns(2)
                with _bs_col:
                    st.markdown("### ↕️ By Side")
                    if not _intra_ev["by_side"].empty:
                        st.dataframe(_intra_ev["by_side"], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Not enough data.")
                with _bt_col:
                    st.markdown("### 🕐 By Time of Day")
                    if not _intra_ev["by_time"].empty:
                        st.dataframe(_intra_ev["by_time"], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Not enough data.")

                st.markdown("---")

                # ── Raw trades table ─────────────────────────────────────────
                with st.expander("📋 Raw closed trades log", expanded=False):
                    _raw_df = _intra_ev["trades_df"]
                    _disp_cols = [c for c in ["trade_date","signal_time","side","entry_price",
                                              "stop","target","exit_price","exit_reason",
                                              "pnl","r_multiple","confidence_score",
                                              "volume_ratio","or_range_pct","track"] if c in _raw_df.columns]
                    st.dataframe(_raw_df[_disp_cols], use_container_width=True)
                    _intra_csv = _raw_df[_disp_cols].to_csv(index=False)
                    st.download_button("⬇️ Download intraday evidence CSV",
                                       _intra_csv,
                                       file_name="moil_intraday_evidence.csv",
                                       mime="text/csv",
                                       key="ev_intra_csv")

        # ── MTF sub-tab ──────────────────────────────────────────────────────
        with _ev_mtf_tab:
            _aw_dict = st.session_state.mtf_adaptive_weights.get_weights() \
                       if hasattr(st.session_state.mtf_adaptive_weights, "get_weights") else {}
            _mtf_ev = compute_mtf_evidence(
                mtf_paper_log=st.session_state.mtf_paper_log,
                mtf_backtest_result=st.session_state.mtf_backtest_result,
                adaptive_weights=st.session_state.mtf_adaptive_weights,
            )

            if _mtf_ev["error"]:
                st.info(_mtf_ev["error"])
                st.caption(
                    "To populate this tab: go to **📈 Intraday → MTF Engine**, run the engine "
                    "and click **Auto-execute**, or run the **Walk-Forward Backtest**."
                )
            else:
                _ms = _mtf_ev["summary"]
                _mwr = _ms["win_rate"]

                st.caption(f"Data source: **{_ms['source']}** — {_ms['total_trades']} closed trades")

                # ── Top-line KPIs ────────────────────────────────────────────
                st.markdown("### Key Performance Indicators")
                _mk1, _mk2, _mk3, _mk4, _mk5, _mk6 = st.columns(6)
                _mk1.metric("Total trades",    str(_ms["total_trades"]))
                _mk2.metric("Win rate",        _pct_fmt(_mwr),
                            delta=f"{_mwr - 0.50:+.1%} vs 50%",
                            delta_color="normal")
                _mk3.metric("Avg R-multiple",  f"{_ms['avg_r']:.2f}R",
                            delta=f"{_ms['avg_r'] - 1.0:+.2f}R vs 1R",
                            delta_color="normal")
                _mk4.metric("Profit factor",   str(_ms["profit_factor"]),
                            delta="≥1.5 target", delta_color="off")
                _mk5.metric("Expectancy",      f"{_ms['expectancy_pts']:+.3f} pts/trade")
                _mk6.metric("Max drawdown",    f"{_ms['max_dd']:.3f} pts")

                _mk7, _mk8, _mk9, _mk10 = st.columns(4)
                _mk7.metric("Total PnL (pts)", f"{_ms['total_pnl_pts']:+.3f}")
                _mk8.metric("Wins",            str(_ms["wins"]))
                _mk9.metric("Losses",          str(_ms["losses"]))
                _mk10.metric("Max consec. loss", str(_ms["max_consec_loss"]))

                st.markdown(f"> **Edge status: {_colour_wr(_mwr)} {'Positive edge' if _mwr >= 0.55 else 'Marginal' if _mwr >= 0.50 else 'No edge detected'}** — {_ms['total_trades']} closed trades (net of costs)")

                st.markdown("---")

                # ── Backtest metrics panel ───────────────────────────────────
                _bm = _mtf_ev["backtest_metrics"]
                if _bm:
                    st.markdown("### 🔬 Backtest Metrics (walk-forward, with costs)")
                    _bk1, _bk2, _bk3, _bk4, _bk5 = st.columns(5)
                    _bk1.metric("Trades taken",   str(_bm.get("trades_taken", "—")))
                    _bk2.metric("Win rate",       _pct_fmt(_bm.get("win_rate", 0)))
                    _bk3.metric("Profit factor",  str(_bm.get("profit_factor", "—")))
                    _bk4.metric("Total PnL",      f"{_bm.get('total_pnl', 0):+.3f} pts")
                    _bk5.metric("Max drawdown",   f"{_bm.get('max_drawdown', 0):.3f} pts")
                    st.markdown("---")

                # ── Rolling win rate ─────────────────────────────────────────
                st.markdown("### 📈 Rolling 10-trade Win Rate")
                _mrwr = _mtf_ev["rolling_wr"]
                if not _mrwr.empty:
                    import plotly.graph_objects as _go_mev
                    _mrwr_fig = _go_mev.Figure()
                    _mrwr_fig.add_trace(_go_mev.Scatter(
                        y=_mrwr.values, mode="lines+markers",
                        line=dict(color="#7b61ff", width=2),
                        marker=dict(size=5),
                        name="Rolling win rate",
                        fill="tozeroy", fillcolor="rgba(123,97,255,0.12)",
                    ))
                    _mrwr_fig.add_hline(y=0.5, line_dash="dash", line_color="#ef5350",
                                        annotation_text="50% breakeven")
                    _mrwr_fig.add_hline(y=0.6, line_dash="dot", line_color="#26a69a",
                                        annotation_text="60% target")
                    _mrwr_fig.update_layout(
                        template="plotly_dark", height=280,
                        paper_bgcolor="#131722", plot_bgcolor="#131722",
                        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#2a2e39"),
                        xaxis=dict(title="Trade #", gridcolor="#2a2e39"),
                        title="MTF win rate stability (rolling 10 trades)",
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    st.plotly_chart(_mrwr_fig, use_container_width=True)

                # ── Confidence calibration ───────────────────────────────────
                st.markdown("### 🎯 Confidence Score Calibration")
                st.caption("Proves the scoring formula separates good setups from bad ones.")
                _mcal = _mtf_ev["confidence_cal"]
                if not _mcal.empty:
                    st.dataframe(_mcal, use_container_width=True, hide_index=True,
                                 column_config={
                                     "Edge": st.column_config.TextColumn("Edge", width="small"),
                                 })
                    _mcal_fig = _go_mev.Figure()
                    _mwin_pcts = []
                    for _, row in _mcal.iterrows():
                        try:
                            _mwin_pcts.append(float(str(row["Win %"]).replace("%", "")) / 100)
                        except Exception:
                            _mwin_pcts.append(0.0)
                    _mcal_fig.add_trace(_go_mev.Bar(
                        x=_mcal["Conf band"].tolist(),
                        y=_mwin_pcts,
                        marker_color=["#ef5350" if w < 0.5 else "#ffa726" if w < 0.6 else "#26a69a" for w in _mwin_pcts],
                        text=[f"{w:.1%}" for w in _mwin_pcts],
                        textposition="outside",
                    ))
                    _mcal_fig.add_hline(y=0.5, line_dash="dash", line_color="#ef5350",
                                        annotation_text="50%")
                    _mcal_fig.update_layout(
                        template="plotly_dark", height=260,
                        paper_bgcolor="#131722", plot_bgcolor="#131722",
                        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#2a2e39"),
                        xaxis=dict(gridcolor="#2a2e39"),
                        title="MTF win rate by confidence band",
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    st.plotly_chart(_mcal_fig, use_container_width=True)

                st.markdown("---")

                # ── By type + by setup ───────────────────────────────────────
                _mtype_col, _msetup_col = st.columns(2)
                with _mtype_col:
                    st.markdown("### 📂 By Trade Type")
                    if not _mtf_ev["by_type"].empty:
                        st.dataframe(_mtf_ev["by_type"], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Not enough data.")
                with _msetup_col:
                    st.markdown("### 🔧 By Setup Type")
                    if not _mtf_ev["by_setup"].empty:
                        st.dataframe(_mtf_ev["by_setup"], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Not enough data.")

                st.markdown("---")

                # ── Auto vs Manual ───────────────────────────────────────────
                st.markdown("### 🤖 Auto-executed vs 👤 Manual Trades")
                _mbt = _mtf_ev["by_track"]
                _mam1, _mam2 = st.columns(2)
                for _trk_lbl, _mcol in [("🤖 Auto", _mam1), ("👤 Manual", _mam2)]:
                    _trk  = "auto" if "Auto" in _trk_lbl else "manual"
                    _mts  = _mbt.get(_trk, {})
                    with _mcol:
                        st.markdown(f"**{_trk_lbl}**")
                        if _mts.get("trades", 0) == 0:
                            st.caption("No trades in this track.")
                        else:
                            _mcol.metric("Trades",        _mts["trades"])
                            _mcol.metric("Win rate",      _pct_fmt(_mts["win_rate"]),
                                         delta=f"{_mts['win_rate']-0.50:+.1%} vs 50%",
                                         delta_color="normal")
                            _mcol.metric("Profit factor", str(_mts["profit_factor"]))
                            _mcol.metric("Total PnL",     f"{_mts['total_pnl']:+.3f} pts")

                st.markdown("---")

                # ── Adaptive weight drift ────────────────────────────────────
                st.markdown("### ⚖️ Adaptive Weight Drift")
                st.caption("Shows how the engine re-calibrated its factor weights from live trade feedback.")
                _wd = _mtf_ev["weight_drift"]
                if _wd:
                    _wd_rows = []
                    for _factor, _vals in _wd.items():
                        _wd_rows.append({
                            "Factor":   _factor.title(),
                            "Default":  f"{_vals['default']:.2%}",
                            "Current":  f"{_vals['current']:.2%}",
                            "Δ":        f"{_vals['delta']:+.2%}",
                            "Shifted":  "↑ More weight" if _vals["delta"] > 0.01
                                        else "↓ Less weight" if _vals["delta"] < -0.01
                                        else "≈ Stable",
                        })
                    _wd_df = pd.DataFrame(_wd_rows)
                    st.dataframe(_wd_df, use_container_width=True, hide_index=True)

                    _wd_fig = _go_mev.Figure()
                    _fnames  = [r["Factor"] for r in _wd_rows]
                    _def_wts = [_wd[k.lower()]["default"]  for k in _wd]
                    _cur_wts = [_wd[k.lower()]["current"] for k in _wd]
                    _wd_fig.add_trace(_go_mev.Bar(name="Default", x=_fnames, y=_def_wts,
                                                  marker_color="#546e7a"))
                    _wd_fig.add_trace(_go_mev.Bar(name="Current", x=_fnames, y=_cur_wts,
                                                  marker_color="#7b61ff"))
                    _wd_fig.update_layout(
                        barmode="group", template="plotly_dark", height=260,
                        paper_bgcolor="#131722", plot_bgcolor="#131722",
                        yaxis=dict(tickformat=".0%", gridcolor="#2a2e39"),
                        xaxis=dict(gridcolor="#2a2e39"),
                        title="Factor weights: default vs current (adaptive)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=40, r=20, t=50, b=30),
                    )
                    st.plotly_chart(_wd_fig, use_container_width=True)
                else:
                    st.caption("Adaptive weight data not available — run the engine first.")

                st.markdown("---")

                # ── Raw trades download ──────────────────────────────────────
                with st.expander("📋 Raw closed MTF trades log", expanded=False):
                    _mraw = _mtf_ev["trades_df"]
                    _mdisp = [c for c in [
                        "logged_at","action","bias","trade_type","confidence_score",
                        "leverage","entry","stop","target1","target2",
                        "exit_price","exit_reason","pnl_points","r_multiple",
                        "setup_type","track"
                    ] if c in _mraw.columns]
                    st.dataframe(_mraw[_mdisp], use_container_width=True)
                    _mtf_csv = _mraw[_mdisp].to_csv(index=False)
                    st.download_button("⬇️ Download MTF evidence CSV",
                                       _mtf_csv,
                                       file_name="moil_mtf_evidence.csv",
                                       mime="text/csv",
                                       key="ev_mtf_csv")

st.markdown("---")
st.caption("Research dashboard only. No buy/sell advice. Paid feeds such as Reuters, SteelMint or exchange data should be connected through licensed APIs or approved internal data pipelines.")

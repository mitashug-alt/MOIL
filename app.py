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
from data_layer.evidence_store import evidence_to_macro_rows, load_evidence_from_file
from data_layer.validators import validate_evidence_dataframe
from telegram_alert import send_telegram_alert


st.set_page_config(
    page_title="MOIL Macro Radar",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern UI theming (trading-desk inspired)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: radial-gradient(circle at 10% 10%, #0b1220 0, #0a0f1a 40%, #080c15 100%); color: #e5e7eb; }
    .block-container { padding-top: 1.25rem; padding-bottom: 1.75rem; }
    /* Cards / metrics */
    .css-1v3fvcr, .stMetric, .stDataFrame, .stPlotlyChart, .stTable { background: #0f172a; border-radius: 12px; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 12px 28px rgba(0,0,0,0.35); }
    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #cbd5f5; font-weight: 600; background: transparent; border: none; }
    .stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p { margin: 0; }
    .stTabs [aria-selected="true"] { color: #ffffff; border-bottom: 3px solid #38bdf8; }
    /* Buttons */
    .stButton>button { background: linear-gradient(90deg, #2563eb, #0ea5e9); color: #fff; border: none; border-radius: 10px; padding: 0.55rem 0.9rem; font-weight: 600; box-shadow: 0 10px 24px rgba(37,99,235,0.35); }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(14,165,233,0.35); }
    /* Tables */
    .stDataFrame thead tr th { background: #0b1629 !important; color: #cbd5f5 !important; }
    .stDataFrame tbody tr { background: #0f172a; }
    /* Inputs */
    .stTextInput>div>div>input, .stSelectbox [data-baseweb="select"]>div { background: #0f172a; color: #e5e7eb; border: 1px solid rgba(255,255,255,0.06); }
    /* Markdown tweaks */
    h1, h2, h3, h4, h5, h6 { color: #f8fafc; }
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


# Sidebar -----------------------------------------------------------------
st.sidebar.title("MOIL Macro Radar")
st.sidebar.caption("Regime intelligence for MOIL, manganese and Indian metals")

period = st.sidebar.selectbox("Market history", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Sampling interval", ["1d", "1wk"], index=0)
correlation_lookback = st.sidebar.slider("Correlation window", 30, 252, 90, 5)
anomaly_window = st.sidebar.slider("Anomaly window", 20, 180, 60, 5)
use_demo_data = st.sidebar.checkbox("Use synthetic demo data", value=False)

with st.sidebar.expander("Ticker map", expanded=False):
    edited_tickers: Dict[str, str] = {}
    for name, ticker in DEFAULT_MARKET_TICKERS.items():
        edited_tickers[name] = st.text_input(name, value=ticker, key=f"ticker_{name}")

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

# Data load ----------------------------------------------------------------
st.title("MOIL Macro Radar")
st.caption("From reactive ticker watching to predictive regime analysis for the Indian manganese/steel ecosystem.")

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

# Tabs ---------------------------------------------------------------------
tabs = st.tabs(
    [
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
)

with tabs[0]:
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

        st.markdown("#### Intraday (5m) MOIL")
        intraday_data, intraday_err = cached_intraday_data(tuple(DEFAULT_MARKET_TICKERS.items()), demo=use_demo_data)
        if intraday_err:
            st.caption("; ".join(f"{k}: {v}" for k, v in intraday_err.items()))
        if intraday_data.get("MOIL") is not None and not intraday_data["MOIL"].empty:
            intraday_df = localize_to_ist(intraday_data["MOIL"].tail(200))
            intraday_fig = go.Figure(
                go.Candlestick(
                    x=intraday_df.index,
                    open=intraday_df["Open"],
                    high=intraday_df["High"],
                    low=intraday_df["Low"],
                    close=intraday_df["Close"],
                    name="MOIL 5m",
                )
            )
            intraday_fig.update_layout(title="MOIL intraday (5m)", xaxis_rangeslider_visible=False, yaxis_title="Price")
            st.plotly_chart(intraday_fig, use_container_width=True)
            last_row = intraday_df.iloc[-1]
            first_row = intraday_df.iloc[0]
            intraday_return = (last_row["Close"] / first_row["Open"] - 1) * 100
            session_high = intraday_df["High"].max()
            session_low = intraday_df["Low"].min()
            pos = None
            try:
                if session_high > session_low:
                    pos = (last_row["Close"] - session_low) / (session_high - session_low)
            except Exception:
                pos = None
            last_ts = intraday_df.index[-1]
            st.caption(
                f"Intraday change from session start: {intraday_return:.2f}% | Last close: {last_row['Close']:.2f} | "
                f"Session high/low: {session_high:.2f} / {session_low:.2f} | Position in range: {pos*100:.1f}%" if pos is not None else ""
            )
            st.caption(f"Last bar: {last_ts}")
            csv_buffer = io.StringIO()
            intraday_df.to_csv(csv_buffer)
            st.download_button(
                "Download intraday OHLCV (5m)",
                csv_buffer.getvalue(),
                file_name="moil_intraday_5m.csv",
                mime="text/csv",
            )
        else:
            st.info("Intraday feed unavailable (yfinance blocked or MOIL not returned).")

        st.markdown("#### Intraday research (VWAP + ORB)")
        st.caption("Lightweight, research-only; uses sample data by default. Upload your own 5m CSV/Parquet to backtest.")
        sample_intraday_path = Path("data/sample/intraday_sample_moil.csv")
        upload = st.file_uploader("Upload intraday OHLCV (datetime, open, high, low, close, volume)", type=["csv", "parquet", "pq"], accept_multiple_files=False)
        with st.expander("Advanced settings", expanded=False):
            vol_mult = st.slider("Volume ratio threshold (x prev 5 bars)", 1.0, 4.0, 2.0, 0.1)
            rr = st.slider("Risk-reward", 1.0, 3.0, 2.0, 0.1)
            one_trade = st.checkbox("One trade per day", value=True)
            ema_filter = st.checkbox("Require EMA20 filter", value=False)
            slippage_bps = st.slider("Slippage + costs (bps total)", 0, 50, 7, 1)
        run_col, dl_col = st.columns([1, 1])
        with run_col:
            if st.button("Run intraday backtest", use_container_width=True):
                with st.spinner("Running VWAP + ORB backtest..."):
                    try:
                        if upload is not None:
                            if upload.name.lower().endswith((".parquet", ".pq")):
                                df_raw = pd.read_parquet(upload)
                            else:
                                df_raw = pd.read_csv(upload)
                            if "symbol" not in df_raw.columns:
                                df_raw["symbol"] = "MOIL"
                        else:
                            df_raw = load_intraday(sample_intraday_path).df
                        cfg = StrategyConfig(
                            volume_multiplier=vol_mult,
                            risk_reward=rr,
                            one_trade_per_day=one_trade,
                            require_ema_filter=ema_filter,
                            slippage_bps=slippage_bps,
                        )
                        prepared = prepare_intraday(df_raw, cfg)
                        trades, daily, bt_summary, stats = run_backtest(prepared, cfg)
                        feats = daily_features(trades, stats, prepared)
                        st.session_state["intraday_trades"] = trades
                        st.session_state["intraday_daily"] = daily
                        st.session_state["intraday_feats"] = feats
                        st.session_state["intraday_summary"] = bt_summary
                        st.success(f"Backtest complete: {bt_summary['total_trades']} trades")
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
        with dl_col:
            if "intraday_summary" in st.session_state and st.session_state.get("intraday_trades") is not None:
                s = st.session_state["intraday_summary"]
                trades = st.session_state["intraday_trades"]
                daily = st.session_state.get("intraday_daily", pd.DataFrame())
                feats = st.session_state.get("intraday_feats", pd.DataFrame())
                c1, c2, c3 = st.columns(3)
                c1.metric("Trades", s.get("total_trades", 0))
                c2.metric("Win rate", f"{s.get('win_rate', 0):.1%}")
                c3.metric("Net PnL", f"{s.get('net_pnl', 0):.2f}")
                st.dataframe(trades.head(15), use_container_width=True)
                st.dataframe(daily.head(10), use_container_width=True)
                if not feats.empty:
                    st.dataframe(feats.head(10), use_container_width=True)
                tbuf = io.StringIO(); trades.to_csv(tbuf, index=False)
                dbuf = io.StringIO(); daily.to_csv(dbuf, index=False)
                fbuf = io.StringIO(); feats.to_csv(fbuf, index=False)
                st.download_button("Download trades CSV", tbuf.getvalue(), file_name="intraday_trades.csv", mime="text/csv")
                st.download_button("Download daily PnL CSV", dbuf.getvalue(), file_name="intraday_daily_pnl.csv", mime="text/csv")
                st.download_button("Download features CSV", fbuf.getvalue(), file_name="intraday_features.csv", mime="text/csv")
    with right:
        st.markdown("#### Latest snapshot")
        st.dataframe(format_snapshot(snapshot), use_container_width=True, hide_index=True)
        st.markdown("#### MOIL technical levels")
        st.json(technical_levels or {"status": "MOIL OHLCV unavailable"})

with tabs[1]:
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

with tabs[2]:
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

with tabs[3]:
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

with tabs[4]:
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

with tabs[5]:
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

with tabs[6]:
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

with tabs[7]:
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

with tabs[8]:
    st.subheader("Data Quality Command Center")
    # On-demand macro cache refresh (uses existing fetch_source_aware_macro_feed fallbacks)
    if st.button("Refresh macro cache now"):
        with st.spinner("Refreshing macro cache..."):
            rows, diag = fetch_source_aware_macro_feed(max_results_per_query=3)
            cache_dir = Path("data/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "latest_macro_data.json").write_text(rows.to_json(orient="records"), encoding="utf-8")
        st.success("Macro cache refreshed to data/cache/latest_macro_data.json")
        if diag:
            st.caption("; ".join(diag))
        st.experimental_rerun()

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

st.markdown("---")
st.caption("Research dashboard only. No buy/sell advice. Paid feeds such as Reuters, SteelMint or exchange data should be connected through licensed APIs or approved internal data pipelines.")

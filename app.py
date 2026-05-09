"""MOIL Macro Radar - Streamlit application.

Institutional regime-intelligence dashboard for MOIL Ltd, manganese,
ferroalloys, Indian steel-cycle proxies, China export pressure, energy/FX
stress, institutional flow monitoring and Groq/Llama 3.3 AI scoring.
"""

from __future__ import annotations

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
    correlation_matrix,
    detect_anomalies,
    fetch_market_data,
    fetch_nse_deals,
    generate_demo_market_data,
    generate_rule_based_commentary,
    metric_snapshot,
    normalize_manual_macro,
    normalize_to_100,
    read_manual_macro_csv,
    read_news_tracker_csv,
    run_market_validation,
)
from telegram_alert import send_telegram_alert


st.set_page_config(
    page_title="MOIL Macro Radar",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
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
if "groq_scores" not in st.session_state:
    st.session_state.groq_scores = None
if "groq_scores_applied" not in st.session_state:
    st.session_state.groq_scores_applied = None
if "groq_commentary" not in st.session_state:
    st.session_state.groq_commentary = None

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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Groq configured", "Yes" if is_groq_configured() else "No")
    c2.metric("Model", config["model"])
    c3.metric("Key name", config["key_name"])
    c4.metric("Project", config["project"])
    st.caption("The API key is never displayed. Configure GROQ_API_KEY in Streamlit secrets, GitHub Codespaces secrets, or environment variables.")

    if not is_groq_configured():
        st.warning("Groq is not configured. Manual macro scoring and rule-based commentary remain active.")
    run_col, apply_col, clear_col = st.columns(3)
    with run_col:
        if st.button("Run Groq macro scoring", disabled=not (enable_groq_scoring and is_groq_configured())):
            with st.spinner("Groq/Llama 3.3 is scoring macro indicators..."):
                regime_context = {
                    "label": summary.label,
                    "normalized_score": summary.normalized_score,
                    "confidence_score": summary.confidence_score,
                    "data_quality_score": summary.data_quality_score,
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
            st.success("Groq scores applied to current regime calculation.")
            st.rerun()
    with clear_col:
        if st.button("Clear Groq scores"):
            st.session_state.groq_scores = None
            st.session_state.groq_scores_applied = None
            st.session_state.groq_commentary = None
            st.rerun()

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
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("NSE Sync: Bulk/Block Deals"):
            with st.spinner("Fetching NSE deal data..."):
                deals = fetch_nse_deals("MOIL")
                if deals is not None and not deals.empty:
                    updates = []
                    for _, row in deals.head(10).iterrows():
                        details = ", ".join(f"{k}: {v}" for k, v in row.to_dict().items() if pd.notna(v))[:500]
                        updates.append(
                            {
                                "date": str(row.get("date", "")),
                                "category": "Institutional",
                                "item": str(row.get("clientName", row.get("name", "NSE deal"))),
                                "value": str(row.get("type", "deal")),
                                "impact": 1.0 if "BUY" in str(row.get("type", "")).upper() else -1.0 if "SELL" in str(row.get("type", "")).upper() else 0.0,
                                "confidence": 0.70,
                                "details": details,
                                "source": "NSE bulk/block sync",
                            }
                        )
                    st.session_state.news_tracker = pd.concat([pd.DataFrame(updates), st.session_state.news_tracker], ignore_index=True)
                    st.success(f"Added {len(updates)} NSE tracker rows.")
                else:
                    st.info("No MOIL NSE deals returned or endpoint blocked in this environment.")
    with n2:
        if st.button("Save news tracker"):
            DATA_DIR.mkdir(exist_ok=True)
            st.session_state.news_tracker.to_csv(NEWS_TRACKER_PATH, index=False)
            st.success(f"Saved to {NEWS_TRACKER_PATH}")
    with n3:
        buffer = io.StringIO()
        st.session_state.news_tracker.to_csv(buffer, index=False)
        st.download_button("Download tracker CSV", buffer.getvalue(), file_name="news_tracker.csv", mime="text/csv")

    edited_news = st.data_editor(
        st.session_state.news_tracker,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=360,
        column_config={
            "impact": st.column_config.NumberColumn("impact", min_value=-2.0, max_value=2.0, step=0.5),
            "confidence": st.column_config.NumberColumn("confidence", min_value=0.0, max_value=1.0, step=0.05),
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
    st.subheader("Data quality")
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

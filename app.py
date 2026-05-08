from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from macro_radar import (
    DEFAULT_MARKET_TICKERS,
    MANUAL_INDICATOR_TEMPLATE,
    build_price_panel,
    compute_regime_score,
    correlation_matrix,
    detect_return_anomalies,
    fetch_market_data,
    format_alert_message,
    generate_demo_market_data,
    generate_openai_commentary,
    generate_rule_based_commentary,
    metric_snapshot,
    moil_correlation_table,
    normalize_manual_macro,
    normalize_to_100,
    read_manual_macro_csv,
    rolling_correlation,
)
from telegram_alert import send_telegram_alert


st.set_page_config(
    page_title="MOIL Macro Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
MANUAL_TEMPLATE_PATH = DATA_DIR / "manual_macro_template.csv"


def _safe_secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def load_live_data(tickers: Dict[str, str], period: str, interval: str):
    return fetch_market_data(tickers, period=period, interval=interval)


def make_metric_card(label: str, snapshot: pd.DataFrame) -> None:
    row = snapshot[snapshot["asset"] == label]
    if row.empty:
        st.metric(label, "n/a", "feed unavailable")
        return
    record = row.iloc[0]
    latest = record["latest"]
    daily = record["1D %"]
    latest_text = "n/a" if pd.isna(latest) else f"{latest:,.2f}"
    daily_text = "n/a" if pd.isna(daily) else f"{daily:+.2%}"
    st.metric(label, latest_text, daily_text)


def line_chart_from_panel(panel: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    for col in panel.columns:
        fig.add_trace(go.Scatter(x=panel.index, y=panel[col], mode="lines", name=col))
    fig.update_layout(title=title, yaxis_title=y_title, hovermode="x unified", legend_title_text="")
    return fig


def pct_cols_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    pct_cols = [col for col in display.columns if "%" in str(col) or "vol" in str(col).lower()]
    for col in pct_cols:
        display[col] = display[col].apply(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return display


# ----------------------------- Sidebar controls -----------------------------
st.sidebar.title("MOIL Macro Radar")
st.sidebar.caption("Live market proxies + manual physical-market tracker")

period = st.sidebar.selectbox("Market history", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Sampling interval", ["1d", "1wk"], index=0)
correlation_lookback = st.sidebar.slider("Correlation lookback sessions", 30, 252, 90, 5)
rolling_corr_window = st.sidebar.slider("Rolling correlation window", 20, 180, 60, 5)
anomaly_threshold = st.sidebar.slider("Anomaly z-score threshold", 1.5, 4.0, 2.5, 0.1)
use_demo_data = st.sidebar.checkbox("Use synthetic demo data", value=False)

with st.sidebar.expander("Ticker map", expanded=False):
    st.caption("Edit symbols when a provider changes coverage or you prefer a different proxy.")
    edited_tickers: Dict[str, str] = {}
    for name, ticker in DEFAULT_MARKET_TICKERS.items():
        edited_tickers[name] = st.text_input(name, value=ticker, key=f"ticker_{name}")

with st.sidebar.expander("AI commentary", expanded=False):
    openai_api_key = st.text_input(
        "OpenAI API key",
        value=_safe_secret("OPENAI_API_KEY"),
        type="password",
        help="Optional. Leave blank to use deterministic rule-based commentary.",
    )
    openai_model = st.text_input(
        "Model",
        value=_safe_secret("OPENAI_MODEL", "gpt-4o-mini"),
    )
    use_openai_commentary = st.checkbox("Generate commentary with OpenAI", value=False)

with st.sidebar.expander("Telegram alerts", expanded=False):
    telegram_token = st.text_input(
        "Bot token",
        value=_safe_secret("TELEGRAM_BOT_TOKEN"),
        type="password",
    )
    telegram_chat_id = st.text_input("Chat ID", value=_safe_secret("TELEGRAM_CHAT_ID"))

# -------------------------------- Data load ---------------------------------
st.title("MOIL Macro Radar")
st.caption(
    "Industrial commodity intelligence dashboard for MOIL, Indian steel-cycle proxies, energy stress, FX pressure and China-linked supply risk."
)

manual_upload = st.sidebar.file_uploader(
    "Upload manual macro CSV",
    type=["csv"],
    help="Use the included template for silico-manganese, India steel, China exports and power-stress rows.",
)

if manual_upload is not None:
    manual_macro = normalize_manual_macro(pd.read_csv(manual_upload))
else:
    manual_macro = read_manual_macro_csv(MANUAL_TEMPLATE_PATH)

with st.sidebar.expander("Manual macro scoring", expanded=False):
    st.caption("Scores here immediately feed the composite regime model. Use -2 to +2.")
    manual_macro = st.data_editor(
        manual_macro,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=260,
        key="sidebar_manual_macro_editor",
        column_config={
            "score": st.column_config.NumberColumn("score", min_value=-2, max_value=2, step=0.5),
            "status": st.column_config.SelectboxColumn(
                "status", options=["bullish", "constructive", "neutral", "watch", "bearish"]
            ),
        },
    )
    manual_macro = normalize_manual_macro(manual_macro)

with st.spinner("Loading market data..."):
    if use_demo_data:
        data = generate_demo_market_data(edited_tickers.keys())
        errors = {}
        st.info("Synthetic demo data is active. Disable it to use live yfinance feeds.")
    else:
        data, errors = load_live_data(edited_tickers, period, interval)
        if not data:
            data = generate_demo_market_data(edited_tickers.keys())
            st.warning("Live market data did not return usable prices, so the dashboard is displaying synthetic demo data.")

prices = build_price_panel(data)
snapshot = metric_snapshot(prices)
scorecard, summary = compute_regime_score(prices, manual_macro)
corr = correlation_matrix(prices, lookback=correlation_lookback)
moil_corr = moil_correlation_table(corr, base="MOIL")
anomalies = detect_return_anomalies(prices, window=rolling_corr_window, threshold=anomaly_threshold)

if errors:
    with st.expander("Feed warnings", expanded=False):
        st.write(pd.DataFrame([{"asset": k, "warning": v} for k, v in errors.items()]))

# ------------------------------- Hero metrics -------------------------------
hero_cols = st.columns(5)
with hero_cols[0]:
    make_metric_card("MOIL", snapshot)
with hero_cols[1]:
    make_metric_card("NIFTY Metal", snapshot)
with hero_cols[2]:
    make_metric_card("Brent Crude", snapshot)
with hero_cols[3]:
    make_metric_card("USDINR", snapshot)
with hero_cols[4]:
    st.metric("Regime Score", f"{summary.normalized_score:.1f}/100", summary.label)

st.markdown("---")

# -------------------------------- Main tabs ---------------------------------
tab_overview, tab_correlations, tab_regime, tab_macro, tab_alerts, tab_data = st.tabs(
    [
        "Market radar",
        "Correlation engine",
        "Regime score",
        "Physical macro tracker",
        "Alerts & commentary",
        "Data table",
    ]
)

with tab_overview:
    left, right = st.columns([2, 1])
    with left:
        default_assets = [asset for asset in ["MOIL", "NIFTY Metal", "JSW Steel", "Tata Steel", "Brent Crude", "USDINR"] if asset in prices.columns]
        selected_assets = st.multiselect(
            "Assets to plot",
            options=list(prices.columns),
            default=default_assets or list(prices.columns[: min(4, len(prices.columns))]),
        )
        if selected_assets:
            normalized = normalize_to_100(prices[selected_assets].dropna(how="all"))
            st.plotly_chart(
                line_chart_from_panel(normalized, "Normalized performance, indexed to 100", "Index"),
                use_container_width=True,
            )
        else:
            st.info("Select at least one asset to plot.")

    with right:
        st.subheader("Latest snapshot")
        display_snapshot = snapshot[snapshot["asset"].isin(selected_assets if selected_assets else snapshot["asset"])]
        st.dataframe(pct_cols_for_display(display_snapshot), use_container_width=True, hide_index=True)

    if "MOIL" in data and not data["MOIL"].empty:
        moil_df = data["MOIL"].dropna(subset=["Close"])
        candle = go.Figure(
            data=[
                go.Candlestick(
                    x=moil_df.index,
                    open=moil_df["Open"],
                    high=moil_df["High"],
                    low=moil_df["Low"],
                    close=moil_df["Close"],
                    name="MOIL",
                )
            ]
        )
        candle.update_layout(title="MOIL price structure", xaxis_rangeslider_visible=False, yaxis_title="Price")
        st.plotly_chart(candle, use_container_width=True)

with tab_correlations:
    st.subheader("Correlation engine")
    if corr.empty:
        st.info("Not enough return history for correlation analysis.")
    else:
        heatmap = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title=f"{correlation_lookback}-session return correlation matrix",
        )
        st.plotly_chart(heatmap, use_container_width=True)

        corr_col1, corr_col2 = st.columns([1, 2])
        with corr_col1:
            st.markdown("#### MOIL correlation rank")
            st.dataframe(moil_corr, use_container_width=True, hide_index=True)
        with corr_col2:
            peer_options = [asset for asset in prices.columns if asset != "MOIL"]
            default_peer_index = peer_options.index("NIFTY Metal") if "NIFTY Metal" in peer_options else 0
            selected_peer = st.selectbox("Rolling MOIL correlation versus", peer_options, index=default_peer_index)
            rolling = rolling_correlation(prices, "MOIL", selected_peer, window=rolling_corr_window)
            rolling_fig = go.Figure()
            rolling_fig.add_trace(go.Scatter(x=rolling.index, y=rolling, mode="lines", name=f"MOIL vs {selected_peer}"))
            rolling_fig.update_layout(
                title=f"Rolling {rolling_corr_window}-session correlation",
                yaxis_title="Correlation",
                yaxis_range=[-1, 1],
                hovermode="x unified",
            )
            st.plotly_chart(rolling_fig, use_container_width=True)

with tab_regime:
    st.subheader("Composite macro regime score")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=summary.normalized_score,
            number={"suffix": "/100"},
            delta={"reference": 50},
            title={"text": summary.label},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.22},
                "steps": [
                    {"range": [0, 30], "color": "#f5f5f5"},
                    {"range": [30, 45], "color": "#eeeeee"},
                    {"range": [45, 55], "color": "#e0e0e0"},
                    {"range": [55, 70], "color": "#eeeeee"},
                    {"range": [70, 100], "color": "#f5f5f5"},
                ],
            },
        )
    )
    st.plotly_chart(gauge, use_container_width=True)

    st.caption(
        f"Raw score {summary.score:+.2f}, range {summary.min_score:+.2f} to {summary.max_score:+.2f}. Updated {summary.updated_at}."
    )
    st.dataframe(scorecard, use_container_width=True, hide_index=True)

    st.markdown("#### Return anomaly monitor")
    if anomalies.empty:
        st.info("Not enough data for anomaly detection.")
    else:
        display_anomalies = anomalies.copy()
        display_anomalies["latest_return_%"] = display_anomalies["latest_return_%"].map(lambda x: f"{x:.2%}")
        display_anomalies["z_score"] = display_anomalies["z_score"].map(lambda x: f"{x:+.2f}")
        st.dataframe(display_anomalies, use_container_width=True, hide_index=True)

with tab_macro:
    st.subheader("Manual physical macro tracker")
    st.write(
        "These indicators usually require paid feeds or desk updates: silico-manganese, India crude steel production, China exports and China power/industrial stress. Edit scores in the sidebar; the current scored table is shown below."
    )
    st.dataframe(manual_macro, use_container_width=True, hide_index=True)
    csv_buffer = StringIO()
    manual_macro.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download updated manual macro CSV",
        csv_buffer.getvalue(),
        file_name="manual_macro_template.csv",
        mime="text/csv",
    )
    st.caption("Place the downloaded file at `data/manual_macro_template.csv` to load it automatically on the next run.")

with tab_alerts:
    st.subheader("AI-generated macro commentary")
    if use_openai_commentary and openai_api_key:
        commentary = generate_openai_commentary(
            openai_api_key,
            openai_model,
            summary,
            scorecard,
            moil_corr,
            anomalies,
        )
        if not commentary:
            commentary = generate_rule_based_commentary(summary, scorecard, moil_corr, anomalies)
    else:
        commentary = generate_rule_based_commentary(summary, scorecard, moil_corr, anomalies)
    st.markdown(commentary)

    st.markdown("#### Telegram snapshot")
    alert_message = format_alert_message(summary, scorecard, anomalies)
    st.text_area("Alert preview", alert_message, height=180)
    if st.button("Send Telegram alert"):
        ok, detail = send_telegram_alert(alert_message, telegram_token, telegram_chat_id)
        if ok:
            st.success(detail)
        else:
            st.error(detail)

    st.download_button(
        "Download scorecard CSV",
        scorecard.to_csv(index=False),
        file_name="moil_regime_scorecard.csv",
        mime="text/csv",
    )

with tab_data:
    st.subheader("Raw market data")
    st.dataframe(pct_cols_for_display(snapshot), use_container_width=True, hide_index=True)
    if not prices.empty:
        st.download_button(
            "Download price panel CSV",
            prices.to_csv(index=True),
            file_name="moil_price_panel.csv",
            mime="text/csv",
        )
        with st.expander("Price panel"):
            st.dataframe(prices.tail(500), use_container_width=True)

st.markdown("---")
st.caption(
    "Research dashboard only. Paid data connectors such as Reuters and SteelMint should be wired through licensed APIs or approved internal data pipelines."
)

"""MOIL Macro Radar Dashboard - Complete Streamlit Application

Institutional-grade commodity intelligence dashboard focused on MOIL Ltd and macro factors
affecting manganese, silico-manganese, ferroalloys, Indian steel demand, China steel exports,
energy stress, freight, USDINR, and commodity-cycle regime.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gemini_macro_scorer import (
    get_gemini_config,
    is_gemini_configured,
    score_macro_with_gemini,
    generate_gemini_commentary,
)
from macro_radar import (
    DEFAULT_MARKET_TICKERS,
    build_price_panel,
    compute_regime_score,
    correlation_matrix,
    detect_anomalies,
    fetch_market_data,
    generate_gemini_commentary_wrapper,
    generate_rule_based_commentary,
    metric_snapshot,
    normalize_manual_macro,
    read_manual_macro_csv,
    rolling_correlation,
)
from telegram_alert import send_telegram_alert

# Page configuration
st.set_page_config(
    page_title="MOIL Macro Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
DATA_DIR = Path("data")
MANUAL_TEMPLATE_PATH = DATA_DIR / "manual_macro_template.csv"

# Sidebar controls
st.sidebar.title("MOIL Macro Radar")
st.sidebar.caption("Live market proxies + manual physical-market tracker")

# Market data controls
period = st.sidebar.selectbox("Market history", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Sampling interval", ["1d", "1wk"], index=0)
correlation_lookback = st.sidebar.slider("Correlation lookback sessions", 30, 252, 90, 5)
anomaly_window = st.sidebar.slider("Anomaly detection window", 20, 180, 60, 5)
use_demo_data = st.sidebar.checkbox("Use synthetic demo data", value=False)

# Ticker map editor
with st.sidebar.expander("Ticker map", expanded=False):
    st.caption("Edit symbols when a provider changes coverage or you prefer a different proxy.")
    edited_tickers: Dict[str, str] = {}
    for name, ticker in DEFAULT_MARKET_TICKERS.items():
        edited_tickers[name] = st.text_input(name, value=ticker, key=f"ticker_{name}")

# Manual macro controls
with st.sidebar.expander("Manual macro scoring", expanded=False):
    st.caption("Scores here immediately feed the composite regime model. Use -2 to +2.")
    manual_upload = st.sidebar.file_uploader(
        "Upload manual macro CSV",
        type=["csv"],
        help="Use the included template for silico-manganese, India steel, China exports and power-stress rows.",
    )

# Gemini controls
enable_gemini_scoring = st.sidebar.checkbox("Enable Gemini auto scoring", value=False)
enable_gemini_commentary = st.sidebar.checkbox("Enable Gemini commentary", value=False)

# Telegram controls
enable_telegram = st.sidebar.checkbox("Enable Telegram alerts", value=False)

# Main content
st.title("MOIL Macro Radar")
st.caption(
    "Industrial commodity intelligence dashboard for MOIL, Indian steel-cycle proxies, energy stress, FX pressure and China-linked supply risk."
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.gemini_scores = None
    st.session_state.gemini_commentary = None

# Load market data
if not st.session_state.data_loaded:
    with st.spinner("Loading market data..."):
        try:
            if use_demo_data:
                from macro_radar import generate_demo_market_data
                data = generate_demo_market_data(edited_tickers.keys())
                errors = {}
                st.info("Synthetic demo data is active. Disable it to use live yfinance feeds.")
            else:
                data, errors = fetch_market_data(edited_tickers, period=period, interval=interval)
                if not data:
                    st.warning("Live market data did not return usable prices, switching to demo data.")
                    from macro_radar import generate_demo_market_data
                    data = generate_demo_market_data(edited_tickers.keys())
                    errors = {}

            prices = build_price_panel(data)
            snapshot = metric_snapshot(prices)

            # Load manual macro
            if manual_upload is not None:
                manual_macro = normalize_manual_macro(pd.read_csv(manual_upload))
            else:
                manual_macro = read_manual_macro_csv(MANUAL_TEMPLATE_PATH)

            # Compute analytics
            corr = correlation_matrix(prices, lookback=correlation_lookback)
            moil_corr = corr.loc["MOIL"].drop("MOIL").sort_values(ascending=False).reset_index()
            moil_corr.columns = ["asset", "correlation_to_moil"]
            anomalies = detect_anomalies(prices, window=anomaly_window, threshold=2.5)

            # Compute regime score (without Gemini initially)
            scorecard, summary = compute_regime_score(prices, manual_macro)

            # Store in session state
            st.session_state.prices = prices
            st.session_state.snapshot = snapshot
            st.session_state.manual_macro = manual_macro
            st.session_state.corr = corr
            st.session_state.moil_corr = moil_corr
            st.session_state.anomalies = anomalies
            st.session_state.scorecard = scorecard
            st.session_state.summary = summary
            st.session_state.errors = errors
            st.session_state.data_loaded = True

            st.success("Market data loaded successfully!")

        except Exception as exc:
            st.error(f"Error loading market data: {exc}")
            st.stop()

# Tabs
tab_overview, tab_correlations, tab_regime, tab_macro, tab_gemini, tab_alerts, tab_data = st.tabs(
    [
        "Market radar",
        "Correlation engine",
        "Regime score",
        "Manual macro tracker",
        "Gemini auto scoring",
        "Alerts & commentary",
        "Data quality",
    ]
)

# Market radar tab
with tab_overview:
    left, right = st.columns([2, 1])

    with left:
        default_assets = [asset for asset in ["MOIL", "NIFTY Metal", "JSW Steel", "Brent Crude", "USDINR"]
                         if asset in st.session_state.prices.columns]
        selected_assets = st.multiselect(
            "Assets to plot",
            options=list(st.session_state.prices.columns),
            default=default_assets[:4] if default_assets else list(st.session_state.prices.columns[:4]),
        )

        if selected_assets:
            normalized = st.session_state.prices[selected_assets].dropna(how="all")
            if not normalized.empty:
                # Normalize to 100
                normalized = (normalized / normalized.iloc[0] * 100)
                fig = go.Figure()
                for col in normalized.columns:
                    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
                fig.update_layout(
                    title="Normalized performance, indexed to 100",
                    yaxis_title="Index",
                    hovermode="x unified",
                    legend_title_text=""
                )
                st.plotly_chart(fig, use_container_width=True)

        # MOIL candlestick if available
        if "MOIL" in data and not data["MOIL"].empty:
            moil_df = data["MOIL"].dropna(subset=["Close"])
            if len(moil_df) > 0:
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
                candle.update_layout(
                    title="MOIL price structure",
                    xaxis_rangeslider_visible=False,
                    yaxis_title="Price"
                )
                st.plotly_chart(candle, use_container_width=True)

    with right:
        st.subheader("Latest snapshot")
        display_snapshot = st.session_state.snapshot.copy()
        display_snapshot["1D %"] = display_snapshot["1D %"].apply(lambda x: "" if pd.isna(x) else f"{x:.2%}")
        st.dataframe(display_snapshot, use_container_width=True, hide_index=True)

# Correlation engine tab
with tab_correlations:
    st.subheader("Correlation engine")

    if st.session_state.corr.empty:
        st.info("Not enough return history for correlation analysis.")
    else:
        heatmap = px.imshow(
            st.session_state.corr,
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
            st.dataframe(st.session_state.moil_corr, use_container_width=True, hide_index=True)

        with corr_col2:
            peer_options = [asset for asset in st.session_state.prices.columns if asset != "MOIL"]
            default_peer = peer_options[0] if peer_options else None
            selected_peer = st.selectbox("Rolling MOIL correlation versus", peer_options, index=0)

            if selected_peer:
                rolling = rolling_correlation(
                    st.session_state.prices, "MOIL", selected_peer, window=anomaly_window
                )
                if not rolling.empty:
                    rolling_fig = go.Figure()
                    rolling_fig.add_trace(go.Scatter(x=rolling.index, y=rolling, mode="lines", name=f"MOIL vs {selected_peer}"))
                    rolling_fig.update_layout(
                        title=f"Rolling {anomaly_window}-session correlation",
                        yaxis_title="Correlation",
                        yaxis_range=[-1, 1],
                        hovermode="x unified",
                    )
                    st.plotly_chart(rolling_fig, use_container_width=True)

# Regime score tab
with tab_regime:
    st.subheader("Composite macro regime score")

    # Gauge chart
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=st.session_state.summary.normalized_score,
            number={"suffix": "/100"},
            delta={"reference": 50},
            title={"text": st.session_state.summary.label},
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
        f"Raw score {st.session_state.summary.score:+.2f}, range {st.session_state.summary.min_score:+.2f} to {st.session_state.summary.max_score:+.2f}. Updated {st.session_state.summary.updated_at}."
    )
    st.dataframe(st.session_state.scorecard, use_container_width=True, hide_index=True)

    st.markdown("#### Return anomaly monitor")
    if st.session_state.anomalies.empty:
        st.info("Not enough data for anomaly detection.")
    else:
        display_anomalies = st.session_state.anomalies.copy()
        display_anomalies["latest_return_%"] = display_anomalies["latest_return_%"].apply(lambda x: f"{x:.2%}")
        display_anomalies["z_score"] = display_anomalies["z_score"].apply(lambda x: f"{x:+.2f}")
        st.dataframe(display_anomalies, use_container_width=True, hide_index=True)

# Manual macro tracker tab
with tab_macro:
    st.subheader("Manual physical macro tracker")
    st.write(
        "These indicators usually require paid feeds or desk updates: silico-manganese, India crude steel production, China exports and China power-stress. Edit scores in the sidebar; the current scored table is shown below."
    )

    # Editable dataframe
    edited_manual_macro = st.data_editor(
        st.session_state.manual_macro,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=300,
        key="manual_macro_editor",
        column_config={
            "score": st.column_config.NumberColumn("score", min_value=-2, max_value=2, step=0.5),
            "status": st.column_config.SelectboxColumn(
                "status", options=["bullish", "constructive", "neutral", "watch", "bearish"]
            ),
        },
    )

    # Update session state if edited
    if not edited_manual_macro.equals(st.session_state.manual_macro):
        st.session_state.manual_macro = normalize_manual_macro(edited_manual_macro)
        # Recompute regime score
        st.session_state.scorecard, st.session_state.summary = compute_regime_score(
            st.session_state.prices, st.session_state.manual_macro, st.session_state.gemini_scores
        )
        st.rerun()

    # Download button
    csv_buffer = pd.io.common.StringIO()
    st.session_state.manual_macro.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download updated manual macro CSV",
        csv_buffer.getvalue(),
        file_name="manual_macro_template.csv",
        mime="text/csv",
    )
    st.caption("Place the downloaded file at `data/manual_macro_template.csv` to load it automatically on the next run.")

# Gemini auto scoring tab
with tab_gemini:
    st.subheader("Gemini auto scoring")

    # Configuration status
    config = get_gemini_config()
    st.markdown("#### Configuration Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gemini configured", "Yes" if is_gemini_configured() else "No")
    with col2:
        st.metric("Model", config["model"])
    with col3:
        st.metric("Key name", config["key_name"])
    with col4:
        st.metric("Project", config["project"])

    if not is_gemini_configured():
        st.warning("Gemini is not configured. Set GEMINI_API_KEY in secrets or environment variables.")
        st.stop()

    # Run Gemini scoring
    if st.button("Run Gemini macro scoring", disabled=not enable_gemini_scoring):
        with st.spinner("Generating Gemini macro scores..."):
            try:
                market_snapshot = st.session_state.snapshot
                regime_context = {
                    "label": st.session_state.summary.label,
                    "normalized_score": st.session_state.summary.normalized_score,
                }

                gemini_result = score_macro_with_gemini(
                    st.session_state.manual_macro,
                    market_snapshot,
                    regime_context
                )

                if gemini_result:
                    st.session_state.gemini_scores = gemini_result

                    # Recompute regime score with Gemini scores
                    st.session_state.scorecard, st.session_state.summary = compute_regime_score(
                        st.session_state.prices, st.session_state.manual_macro, st.session_state.gemini_scores
                    )

                    st.success("Gemini macro scoring completed!")

                    # Display results
                    st.markdown("#### Gemini Macro Scores")
                    gemini_df = pd.DataFrame(gemini_result["macro_scores"])
                    st.dataframe(gemini_df, use_container_width=True, hide_index=True)

                    st.markdown("#### Overall Assessment")
                    st.metric("Overall manual macro score", f"{gemini_result['overall_manual_macro_score']:+.2f}")
                    st.write(f"**Regime commentary:** {gemini_result['regime_commentary']}")

                    if gemini_result.get("watch_items"):
                        st.markdown("**Watch items:**")
                        for item in gemini_result["watch_items"]:
                            st.write(f"- {item}")

                    if gemini_result.get("risks"):
                        st.markdown("**Risks:**")
                        for risk in gemini_result["risks"]:
                            st.write(f"- {risk}")

                    # Apply to current session
                    if st.button("Apply Gemini scores to regime calculation"):
                        st.rerun()

                else:
                    st.error("Gemini scoring failed. Check API key and configuration.")

            except Exception as exc:
                st.error(f"Gemini scoring error: {exc}")

# Alerts and commentary tab
with tab_alerts:
    st.subheader("AI-generated macro commentary")

    # Generate commentary
    if enable_gemini_commentary and is_gemini_configured():
        if st.button("Generate Gemini commentary"):
            with st.spinner("Generating Gemini commentary..."):
                try:
                    context = {
                        "regime_label": st.session_state.summary.label,
                        "regime_score": st.session_state.summary.normalized_score,
                        "manual_macro_score": st.session_state.gemini_scores.get("overall_manual_macro_score", 0) if st.session_state.gemini_scores else 0,
                        "top_correlations": st.session_state.moil_corr.head(3).to_dict('records') if not st.session_state.moil_corr.empty else [],
                        "anomalies": st.session_state.anomalies.head(5).to_dict('records') if not st.session_state.anomalies.empty else [],
                    }

                    commentary = generate_gemini_commentary(context)
                    if commentary:
                        st.session_state.gemini_commentary = commentary
                        st.markdown(commentary)
                    else:
                        st.error("Gemini commentary generation failed.")

                except Exception as exc:
                    st.error(f"Gemini commentary error: {exc}")
    else:
        # Use rule-based commentary
        commentary = generate_rule_based_commentary(
            st.session_state.summary,
            st.session_state.scorecard,
            st.session_state.moil_corr,
            st.session_state.anomalies
        )
        st.markdown(commentary)

    st.markdown("#### Telegram snapshot")
    from macro_radar import build_telegram_alert_text
    alert_message = build_telegram_alert_text(
        st.session_state.summary,
        st.session_state.scorecard,
        st.session_state.anomalies
    )
    st.text_area("Alert preview", alert_message, height=180)

    if enable_telegram and st.button("Send Telegram alert"):
        ok, detail = send_telegram_alert(alert_message)
        if ok:
            st.success(detail)
        else:
            st.error(detail)

    # Download buttons
    st.download_button(
        "Download scorecard CSV",
        st.session_state.scorecard.to_csv(index=False),
        file_name="moil_regime_scorecard.csv",
        mime="text/csv",
    )

# Data quality tab
with tab_data:
    st.subheader("Raw market data")

    # Data quality metrics
    st.markdown("#### Data Quality")
    quality_data = []
    for name in edited_tickers.keys():
        if name in data:
            df = data[name]
            row_count = len(df)
            date_range = f"{df.index.min().date() if not df.empty else 'N/A'} to {df.index.max().date() if not df.empty else 'N/A'}"
            last_price = df['Close'].iloc[-1] if not df.empty and 'Close' in df.columns else 'N/A'
            quality_data.append({
                "Asset": name,
                "Ticker": edited_tickers[name],
                "Rows": row_count,
                "Date Range": date_range,
                "Last Price": last_price,
            })
        else:
            quality_data.append({
                "Asset": name,
                "Ticker": edited_tickers[name],
                "Rows": 0,
                "Date Range": "No data",
                "Last Price": "N/A",
            })

    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    # Feed warnings
    if st.session_state.errors:
        with st.expander("Feed warnings", expanded=False):
            for asset, error in st.session_state.errors.items():
                st.write(f"**{asset}:** {error}")

    # Raw data download
    st.markdown("#### Raw Data")
    st.dataframe(st.session_state.snapshot, use_container_width=True, hide_index=True)

    if not st.session_state.prices.empty:
        st.download_button(
            "Download price panel CSV",
            st.session_state.prices.to_csv(index=True),
            file_name="moil_price_panel.csv",
            mime="text/csv",
        )

        with st.expander("Price panel preview"):
            st.dataframe(st.session_state.prices.tail(100), use_container_width=True)

st.markdown("---")
st.caption(
    "Research dashboard only. Paid data connectors such as Reuters and SteelMint should be wired through licensed APIs or approved internal data pipelines."
)

"""Telegram alert helper for MOIL Macro Radar.

Credentials are read from Streamlit secrets or environment variables:
TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import requests


def _secret(name: str) -> str:
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return str(os.getenv(name, "") or "").strip()


def send_telegram_alert(
    message: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    timeout: int = 15,
) -> Tuple[bool, str]:
    token = (bot_token or _secret("TELEGRAM_BOT_TOKEN")).strip()
    target = (chat_id or _secret("TELEGRAM_CHAT_ID")).strip()
    clean_message = (message or "").strip()

    if not clean_message:
        return False, "Telegram message is empty."
    if not token or not target:
        return False, "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": target,
            "text": clean_message[:4096],
            "disable_web_page_preview": True,
        }
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            return False, f"Telegram rejected message: {body}"
        return True, "Telegram alert sent."
    except Exception as exc:
        return False, f"Telegram send failed: {exc}"


def build_trade_alert_message(
    symbol: str,
    side: str,
    entry_price: float,
    stop: float,
    target: float,
    confidence_score: int,
    confidence_label: str,
    volume_ratio: float,
    or_high: float,
    or_low: float,
    or_range_pct: float,
    vwap: float,
    signal_time: str,
    trade_date: str,
    track: str,
    approved_by: str,
    risk_reward: float = 2.0,
) -> str:
    """Build a fully-detailed trade alert message for Telegram."""
    risk_per_share = abs(entry_price - stop)
    reward = abs(target - entry_price)
    rr_actual = reward / risk_per_share if risk_per_share > 0 else 0.0
    side_emoji = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
    track_label = "🤖 AUTO" if track == "auto" else "👤 MANUAL-APPROVED"
    conf_bar = "█" * (confidence_score // 10) + "░" * (10 - confidence_score // 10)

    msg = (
        f"⚡ MOIL TRADING ALERT — {symbol}\n"
        f"{'═' * 32}\n"
        f"{side_emoji}  |  {track_label}\n"
        f"📅 {trade_date}  🕐 Signal @ {signal_time} IST\n"
        f"\n"
        f"━━━ EXECUTION LEVELS ━━━\n"
        f"  Entry  : ₹{entry_price:,.2f}\n"
        f"  Stop   : ₹{stop:,.2f}  (risk ₹{risk_per_share:.2f}/share)\n"
        f"  Target : ₹{target:,.2f}  (R:R = 1:{rr_actual:.1f})\n"
        f"\n"
        f"━━━ SIGNAL CONTEXT ━━━\n"
        f"  OR High / Low : ₹{or_high:,.2f} / ₹{or_low:,.2f}\n"
        f"  OR Range      : {or_range_pct:.2f}%\n"
        f"  VWAP at sig.  : ₹{vwap:,.2f}\n"
        f"  Volume ratio  : {volume_ratio:.2f}x avg\n"
        f"\n"
        f"━━━ CONFIDENCE ━━━\n"
        f"  Score  : {confidence_score}/100  ({confidence_label})\n"
        f"  [{conf_bar}]\n"
        f"  Approved by: {approved_by}\n"
        f"\n"
        f"━━━ EXECUTION CHECKLIST ━━━\n"
        f"  1. Open your broker app (Zerodha/Upstox/Angel)\n"
        f"  2. Search: {symbol}.NS\n"
        f"  3. Place {'BUY' if side == 'LONG' else 'SELL'} order @ ~₹{entry_price:,.2f}\n"
        f"  4. Set SL @ ₹{stop:,.2f} (bracket/cover order)\n"
        f"  5. Target: ₹{target:,.2f}\n"
        f"\n"
        f"⚠️ Research dashboard only. Not SEBI-registered advice.\n"
        f"Verify levels before placing any order."
    )
    return msg


def build_mtf_alert_message(
    symbol: str,
    trade_type: str,
    action: str,
    bias: str,
    confidence_score: float,
    decision: str,
    entry: float,
    stop_loss: float,
    targets: list,
    leverage: float,
    position_size: int,
    risk_per_trade: float,
    capital_inr: float,
    notes: str,
    auto_executed: bool,
    ema20: float = 0.0,
    ema50: float = 0.0,
    setup_type: str = "—",
    setup_quality: float = 0.0,
    vol_ratio: float = 1.0,
    regime_capped: bool = False,
) -> str:
    """Build a full MTF Engine trade alert for Telegram."""
    risk_per_share = abs(entry - stop_loss)
    rr1 = abs(targets[0] - entry) / risk_per_share if risk_per_share > 0 else 0
    rr2 = abs(targets[1] - entry) / risk_per_share if risk_per_share > 0 else 0
    conf_bar = "█" * (int(confidence_score) // 10) + "░" * (10 - int(confidence_score) // 10)
    action_emoji = "🟢 BUY" if action == "buy" else "🔴 SELL"
    bias_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(bias, "")
    exec_tag = "🤖 AUTO-EXECUTED" if auto_executed else "📋 RECOMMENDED"
    trade_tag = "⚡ INTRADAY" if trade_type == "intraday" else "🏦 MARGIN (MTF)"
    dec_map = {"strong_trade": "🔥 STRONG", "moderate_trade": "⚡ MODERATE", "avoid": "⛔ AVOID"}
    dec_label = dec_map.get(decision, decision)

    msg = (
        f"🧠 MTF ENGINE ALERT — {symbol}\n"
        f"{'═' * 32}\n"
        f"{trade_tag}  |  {exec_tag}\n"
        f"{action_emoji}  |  Bias: {bias_emoji} {bias.upper()}\n"
        f"Decision: {dec_label}\n"
        f"\n"
        f"━━━ CONFIDENCE ━━━\n"
        f"  Score  : {confidence_score:.1f}/100\n"
        f"  [{conf_bar}]\n"
        f"  Setup  : {setup_type.title()} (quality {setup_quality:.0f}/100)\n"
        f"  Volume : {vol_ratio:.2f}x avg\n"
        f"  Regime cap: {'YES — max 2x leverage' if regime_capped else 'No'}\n"
        f"\n"
        f"━━━ EXECUTION LEVELS ━━━\n"
        f"  Entry    : ₹{entry:,.2f}\n"
        f"  Stop-loss: ₹{stop_loss:,.2f}  (₹{risk_per_share:.2f}/share)\n"
        f"  Target 1 : ₹{targets[0]:,.2f}  (R:R 1:{rr1:.1f}) ← 50% exit\n"
        f"  Target 2 : ₹{targets[1]:,.2f}  (R:R 1:{rr2:.1f}) ← trail rest\n"
        f"\n"
        f"━━━ POSITION / LEVERAGE ━━━\n"
        f"  Leverage : {leverage:.1f}x\n"
        f"  Qty      : {position_size} shares\n"
        f"  Risk ₹   : ₹{risk_per_trade:,.0f}  ({risk_per_trade/capital_inr*100:.2f}% capital)\n"
        f"\n"
        f"━━━ HTF CONTEXT ━━━\n"
        f"  EMA20 : ₹{ema20:,.2f}  |  EMA50: ₹{ema50:,.2f}\n"
        f"\n"
        f"━━━ EXECUTION CHECKLIST ━━━\n"
        f"  1. Open broker (Zerodha/Upstox/Angel)\n"
        f"  2. Search: {symbol}.NS\n"
        f"  3. {'BUY' if action == 'buy' else 'SELL'} @ ~₹{entry:,.2f}\n"
        f"  4. SL @ ₹{stop_loss:,.2f} (bracket/cover order)\n"
        f"  5. T1 ₹{targets[0]:,.2f} | T2 ₹{targets[1]:,.2f}\n"
        f"  6. {'Enable MTF margin in broker app' if trade_type == 'margin' else 'Use CNC/MIS as per holding'}\n"
        f"\n"
        f"📝 {notes}\n"
        f"\n"
        f"⚠️ Research only. Not SEBI-registered advice.\n"
        f"Verify all levels before placing any order."
    )
    return msg


if __name__ == "__main__":
    ok, detail = send_telegram_alert("MOIL Macro Radar test alert")
    print(detail)
    raise SystemExit(0 if ok else 1)

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


if __name__ == "__main__":
    ok, detail = send_telegram_alert("MOIL Macro Radar test alert")
    print(detail)
    raise SystemExit(0 if ok else 1)

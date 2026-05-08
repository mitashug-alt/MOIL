"""Telegram integration for MOIL Macro Radar.

Use environment variables or Streamlit secrets instead of hardcoding credentials:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import requests


TELEGRAM_API_BASE = "https://api.telegram.org"


def send_telegram_alert(
    message: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: Optional[str] = None,
    timeout: int = 15,
) -> Tuple[bool, str]:
    """Send a Telegram message through the Bot API.

    Returns (success, detail). The function never raises for missing credentials or
    Telegram API errors, which makes it safe to call from a dashboard button.
    """
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat:
        return False, "Telegram token/chat_id missing. Set secrets or environment variables."
    if not message.strip():
        return False, "Message is empty."

    payload = {"chat_id": chat, "text": message[:3900]}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        response = requests.post(
            f"{TELEGRAM_API_BASE}/bot{token}/sendMessage",
            json=payload,
            timeout=timeout,
        )
        if response.ok:
            return True, "Telegram alert sent."
        return False, f"Telegram API error {response.status_code}: {response.text[:300]}"
    except requests.RequestException as exc:
        return False, f"Telegram request failed: {exc}"


if __name__ == "__main__":
    ok, detail = send_telegram_alert(
        "MOIL Macro Radar test alert. Configure this script with environment variables."
    )
    print(detail)
    raise SystemExit(0 if ok else 1)

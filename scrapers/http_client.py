from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Mapping

import requests

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123 Safari/537.36",
]


@dataclass
class FetchResult:
    ok: bool
    url: str
    status_code: int | None
    text: str
    content: bytes
    error: str


class ResilientHttpClient:
    """Small polite HTTP client for scheduled scraping.

    This client uses normal headers, retries, timeouts and delays. It does not
    bypass CAPTCHA, login walls, paywalls or bot-detection systems.
    """

    def __init__(self, timeout: int = 25, max_retries: int = 3, delay_seconds: float = 1.5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.session = requests.Session()

    def get(self, url: str, accept: str = "*/*", extra_headers: Mapping[str, str] | None = None) -> FetchResult:
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": accept,
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }
            if extra_headers:
                headers.update(dict(extra_headers))
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                if response.status_code < 400:
                    return FetchResult(True, url, response.status_code, response.text or "", response.content or b"", "")
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            time.sleep(self.delay_seconds * attempt)
        return FetchResult(False, url, None, "", b"", last_error)

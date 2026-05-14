from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


DEFAULT_OUTSTANDING_SHARES = 203_485_211
DEFAULT_SYMBOL = "MOIL"
DEFAULT_SHAREHOLDING_REF_DATE = date(2026, 3, 31)


@dataclass(slots=True)
class TrackerConfig:
    symbol: str = DEFAULT_SYMBOL
    exchange_symbol: str = DEFAULT_SYMBOL
    company_name: str = "MOIL Ltd"
    outstanding_shares: int = DEFAULT_OUTSTANDING_SHARES
    bulk_threshold_shares: int = 1_017_427  # ceil(0.5% of outstanding)
    named_holder_threshold_shares: int = 2_034_853  # ceil(1% of outstanding)
    sast_five_percent_shares: int = 10_174_261
    sast_two_percent_shares: int = 4_069_705
    shareholding_reference_date: date = DEFAULT_SHAREHOLDING_REF_DATE
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    reports_dir: str = "reports"
    request_timeout: float = 20.0
    max_retries: int = 2
    user_agent: str = "MOIL-Institutional-Tracker/1.0"
    accept_language: str = "en-US,en;q=0.9"
    enable_network: bool = True
    verbose: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()

    @property
    def bulk_threshold_pct(self) -> float:
        if self.outstanding_shares == 0:
            return 0.0
        return (self.bulk_threshold_shares / self.outstanding_shares) * 100

    @property
    def named_holder_threshold_pct(self) -> float:
        if self.outstanding_shares == 0:
            return 0.0
        return (self.named_holder_threshold_shares / self.outstanding_shares) * 100

    @property
    def sast_five_percent_pct(self) -> float:
        if self.outstanding_shares == 0:
            return 0.0
        return (self.sast_five_percent_shares / self.outstanding_shares) * 100

    @property
    def sast_two_percent_pct(self) -> float:
        if self.outstanding_shares == 0:
            return 0.0
        return (self.sast_two_percent_shares / self.outstanding_shares) * 100


def tracker_config(overrides: Optional[dict] = None) -> TrackerConfig:
    cfg = TrackerConfig()
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg

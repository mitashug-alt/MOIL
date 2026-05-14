"""
Adaptive Layer
--------------
Tracks past MTF engine outputs and trade results.
Dynamically adjusts indicator weights based on which pillars
correlated with winning vs losing trades.

Architecture:
  - Stateless per execution (weights stored externally as a dict)
  - All state passed in / returned; no hidden globals
  - Fully JSON-serialisable

Weight update rule (simplified reinforcement):
  win  → increase weight of pillars that scored high on that trade
  loss → decrease weight of pillars that scored high on that trade
  Weights are clipped to [MIN_WEIGHT, MAX_WEIGHT] and re-normalised to sum=1.

Default weights match composite_engine spec:
  trend  : 0.35
  setup  : 0.30
  conf   : 0.20
  volume : 0.15
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.60
LEARNING_RATE = 0.02    # shift per trade result


@dataclass
class AdaptiveWeights:
    trend_weight : float = 0.35
    setup_weight : float = 0.30
    conf_weight  : float = 0.20
    volume_weight: float = 0.15

    def get_weights(self) -> dict:
        return {
            "trend_weight" : self.trend_weight,
            "setup_weight" : self.setup_weight,
            "conf_weight"  : self.conf_weight,
            "volume_weight": self.volume_weight,
        }

    def _normalise(self) -> None:
        total = self.trend_weight + self.setup_weight + self.conf_weight + self.volume_weight
        if total <= 0:
            self.trend_weight  = 0.35
            self.setup_weight  = 0.30
            self.conf_weight   = 0.20
            self.volume_weight = 0.15
            return
        self.trend_weight  = round(self.trend_weight  / total, 4)
        self.setup_weight  = round(self.setup_weight  / total, 4)
        self.conf_weight   = round(self.conf_weight   / total, 4)
        self.volume_weight = round(self.volume_weight / total, 4)

    def _clip(self) -> None:
        self.trend_weight  = max(MIN_WEIGHT, min(MAX_WEIGHT, self.trend_weight))
        self.setup_weight  = max(MIN_WEIGHT, min(MAX_WEIGHT, self.setup_weight))
        self.conf_weight   = max(MIN_WEIGHT, min(MAX_WEIGHT, self.conf_weight))
        self.volume_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, self.volume_weight))

    def update(
        self,
        trade_result: str,                  # "win" | "loss" | "breakeven"
        pillar_scores: dict,                # {"trend":float, "setup":float, "conf":float, "volume":float}
        lr: float = LEARNING_RATE,
    ) -> "AdaptiveWeights":
        """
        Return updated AdaptiveWeights (immutable style — original unchanged).
        trade_result : "win" or "loss"
        pillar_scores: normalised 0-1 scores per pillar
        """
        if trade_result not in ("win", "loss"):
            return self

        sign = +1.0 if trade_result == "win" else -1.0
        new = AdaptiveWeights(
            trend_weight  = self.trend_weight  + sign * lr * pillar_scores.get("trend",  0.5),
            setup_weight  = self.setup_weight  + sign * lr * pillar_scores.get("setup",  0.5),
            conf_weight   = self.conf_weight   + sign * lr * pillar_scores.get("conf",   0.5),
            volume_weight = self.volume_weight + sign * lr * pillar_scores.get("volume", 0.5),
        )
        new._clip()
        new._normalise()
        return new

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveWeights":
        return cls(
            trend_weight  = float(d.get("trend_weight",  0.35)),
            setup_weight  = float(d.get("setup_weight",  0.30)),
            conf_weight   = float(d.get("conf_weight",   0.20)),
            volume_weight = float(d.get("volume_weight", 0.15)),
        )

    @classmethod
    def from_json(cls, s: str) -> "AdaptiveWeights":
        return cls.from_dict(json.loads(s))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def update_weights_from_trade_log(
    weights: AdaptiveWeights,
    trade_log: list[dict],
    max_trades: int = 50,
) -> AdaptiveWeights:
    """
    Replay the last `max_trades` closed trades to re-calibrate weights.

    Each entry in trade_log must have:
      {
        "result"        : "win" | "loss",
        "trend_score"   : 0-100,
        "setup_score"   : 0-100,
        "conf_score"    : 0-100,
        "volume_ratio"  : float,
      }
    """
    recent = trade_log[-max_trades:]
    aw = AdaptiveWeights(
        trend_weight  = weights.trend_weight,
        setup_weight  = weights.setup_weight,
        conf_weight   = weights.conf_weight,
        volume_weight = weights.volume_weight,
    )
    for t in recent:
        result = t.get("result", "breakeven")
        pillar_scores = {
            "trend" : t.get("trend_score",  50) / 100.0,
            "setup" : t.get("setup_score",  50) / 100.0,
            "conf"  : t.get("conf_score",   50) / 100.0,
            "volume": min(1.0, t.get("volume_ratio", 1.0) / 2.0),
        }
        aw = aw.update(result, pillar_scores)
    return aw

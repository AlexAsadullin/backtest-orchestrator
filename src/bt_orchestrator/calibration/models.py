"""Serialisable calibrated fill models (JSON, no pickles, so versions can be diffed and audited).

* :class:`FillProbModel` - logistic regression for P(limit order fills within ``horizon_ns``) on the
  distance to the touch (bps), ``log1p`` of the queue ahead, and volatility (bps per sqrt-second).
* :class:`SlippageModel` - least squares for taker slippage versus the arrival mid (bps) on the half
  spread, ``sqrt(qty / ADV)`` (the square-root impact law) and volatility.
* :class:`LatencyModel` - empirical quantiles of submit-to-fill latency for taker orders.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

FILL_PROB_FEATURES: tuple[str, ...] = ("distance_to_touch_bps", "log1p_queue", "vol_bps_sqrt_s")
SLIPPAGE_FEATURES: tuple[str, ...] = ("half_spread_bps", "sqrt_participation", "vol_bps_sqrt_s")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FillProbModel(_Model):
    features: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    horizon_ns: int

    def predict(self, distance_bps: float, queue_ahead: float, vol_bps_sqrt_s: float) -> float:
        x = np.array([distance_bps, math.log1p(max(queue_ahead, 0.0)), vol_bps_sqrt_s])
        z = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        logit = float(self.intercept + np.dot(np.asarray(self.coef), z))
        return 1.0 / (1.0 + math.exp(-logit))


class SlippageModel(_Model):
    features: list[str]
    intercept: float
    coef: list[float]
    resid_std_bps: float

    def predict_bps(
        self, half_spread_bps: float, participation: float, vol_bps_sqrt_s: float
    ) -> float:
        """Expected slippage versus the mid, including the half spread."""
        return half_spread_bps * self.coef[0] + self.impact_bps(participation, vol_bps_sqrt_s)

    def impact_bps(self, participation: float, vol_bps_sqrt_s: float) -> float:
        """The part of the slippage not explained by crossing the spread."""
        return (
            self.intercept
            + self.coef[1] * math.sqrt(max(participation, 0.0))
            + self.coef[2] * vol_bps_sqrt_s
        )


class LatencyModel(_Model):
    quantile_levels: list[float]
    quantiles_ns: list[float]
    mean_ns: float

    def quantile(self, q: float) -> int:
        return int(np.interp(q, self.quantile_levels, self.quantiles_ns))


class FillModelBundle(_Model):
    """One calibrated, versioned set of fill models for a venue and symbol."""

    schema_version: Literal["1"] = "1"
    version: int
    venue: str
    symbol: str
    fitted_at: datetime
    sample_start: datetime
    sample_end: datetime
    data_fingerprint: str
    label_basis: Literal["orders+fills", "fills_only"]
    n_limit_orders: int
    n_taker_fills: int
    reference_queue_ahead: float
    reference_half_spread_bps: float
    reference_vol_bps_sqrt_s: float
    fill_prob: FillProbModel
    slippage: SlippageModel
    latency: LatencyModel
    evaluation: dict[str, float]

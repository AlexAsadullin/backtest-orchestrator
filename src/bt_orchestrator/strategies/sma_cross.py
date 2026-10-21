"""Simple moving-average crossover: long ``qty`` when the fast SMA is above the slow SMA, else short."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from bt_orchestrator.domain import InstrumentSpec
from bt_orchestrator.strategies.base import Strategy, StrategyParams, VectorSignal


class SmaCrossParams(StrategyParams):
    fast: int = Field(ge=1)
    slow: int = Field(ge=2)
    qty: float = Field(gt=0)

    @model_validator(mode="after")
    def _fast_below_slow(self) -> SmaCrossParams:
        if self.fast >= self.slow:
            raise ValueError("fast must be smaller than slow")
        return self


class SmaCross(Strategy):
    name = "sma_cross"
    params_model = SmaCrossParams

    def vectorised(
        self, bars: pd.DataFrame, params: StrategyParams, instrument: InstrumentSpec
    ) -> VectorSignal:
        p = SmaCrossParams.model_validate(params.model_dump())
        close = bars["close"].astype("float64")
        fast = close.rolling(p.fast, min_periods=p.fast).mean().to_numpy()
        slow = close.rolling(p.slow, min_periods=p.slow).mean().to_numpy()
        # Flat until the slow average exists; sign(0) keeps the position flat on exact ties.
        direction = np.where(np.isfinite(slow), np.sign(fast - slow), 0.0)
        return VectorSignal(target=direction * p.qty)

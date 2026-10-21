"""Limit-order mean reversion on the z-score of the close.

Rules (identical in the vectorised and event forms, so the fidelity ladder measures fill semantics,
not logic drift):

1. At the close of bar ``i`` compute ``z = (close_i - mean) / std`` over the last ``lookback`` closes
   (sample standard deviation).
2. Flat with no working order: if ``z < -entry_z`` post a limit buy at ``close * (1 - offset_bps/1e4)``
   rounded down to the tick; if ``z > entry_z`` post a limit sell, rounded up.
3. A working entry order lives for ``order_ttl_bars`` bars and is then cancelled.
4. In a position, exit with a market order once ``z`` has reverted past ``exit_z`` (``z >= -exit_z``
   when long, ``z <= exit_z`` when short). The exit is not checked on the bar the entry filled, so every
   bar carries at most one execution (vectorbt places one order per bar).

The vectorised form assumes a limit buy fills during a later bar whose low reaches the limit, at the
better of the limit and the bar open. This is the optimistic "touch fills" screen; the event and tick
engines replace it with probabilistic and queue-based fills.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field, model_validator

from bt_orchestrator.domain import InstrumentSpec
from bt_orchestrator.strategies.base import (
    BarView,
    EventHandler,
    FillView,
    Side,
    Strategy,
    StrategyContext,
    StrategyParams,
    VectorSignal,
)

BPS = 1e4


class LimitMeanReversionParams(StrategyParams):
    lookback: int = Field(ge=3)
    entry_z: float = Field(gt=0)
    exit_z: float = Field(ge=0)
    offset_bps: float = Field(ge=0)
    qty: float = Field(gt=0)
    order_ttl_bars: int = Field(ge=1)

    @model_validator(mode="after")
    def _exit_inside_entry(self) -> LimitMeanReversionParams:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be smaller than entry_z")
        return self


def zscore_last(window: NDArray[np.float64]) -> float:
    """z-score of the last element of ``window`` against the window's mean and sample std."""
    std = float(np.std(window, ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return 0.0
    return (float(window[-1]) - float(np.mean(window))) / std


def entry_side(z: float, p: LimitMeanReversionParams) -> Side | None:
    if z < -p.entry_z:
        return Side.BUY
    if z > p.entry_z:
        return Side.SELL
    return None


def should_exit(position: float, z: float, p: LimitMeanReversionParams) -> bool:
    return (position > 0 and z >= -p.exit_z) or (position < 0 and z <= p.exit_z)


def entry_price(close: float, side: Side, p: LimitMeanReversionParams, inst: InstrumentSpec) -> float:
    """Passive limit price, rounded away from the market to a whole tick."""
    raw = close * (1.0 - side * p.offset_bps / BPS)
    ticks = math.floor(raw / inst.tick_size) if side is Side.BUY else math.ceil(raw / inst.tick_size)
    return round(ticks * inst.tick_size, inst.price_precision)


class LimitMeanReversion(Strategy):
    name = "limit_mean_reversion"
    params_model = LimitMeanReversionParams

    @staticmethod
    def _params(params: StrategyParams) -> LimitMeanReversionParams:
        return LimitMeanReversionParams.model_validate(params.model_dump())

    def vectorised(
        self, bars: pd.DataFrame, params: StrategyParams, instrument: InstrumentSpec
    ) -> VectorSignal:
        """Simulate the rules over the bar arrays with touch fills (see module docstring)."""
        p = self._params(params)
        n = len(bars)
        o = bars["open"].to_numpy(dtype="float64")
        lo = bars["low"].to_numpy(dtype="float64")
        hi = bars["high"].to_numpy(dtype="float64")
        c = bars["close"].to_numpy(dtype="float64")
        target = np.full(n, np.nan)
        price = np.full(n, np.nan)
        maker = np.zeros(n, dtype=bool)

        position = 0.0
        pending: tuple[Side, float, int] | None = None  # side, limit price, bar placed
        filled_bar = -1
        for i in range(n):
            if pending is not None:
                side, limit, placed = pending
                touched = lo[i] <= limit if side is Side.BUY else hi[i] >= limit
                if touched:
                    fill = min(o[i], limit) if side is Side.BUY else max(o[i], limit)
                    position = side * p.qty
                    target[i], price[i], maker[i] = position, fill, True
                    pending, filled_bar = None, i
                elif i - placed >= p.order_ttl_bars:
                    pending = None
            if i + 1 < p.lookback:
                continue
            z = zscore_last(c[i + 1 - p.lookback : i + 1])
            if position != 0.0:
                if i > filled_bar and should_exit(position, z, p):
                    position = 0.0
                    target[i], price[i], maker[i] = 0.0, c[i], False
            elif pending is None:
                side_opt = entry_side(z, p)
                if side_opt is not None:
                    pending = (side_opt, entry_price(c[i], side_opt, p, instrument), i)
        return VectorSignal(target=target, price=price, maker=maker)

    def event_handler(
        self, bars: pd.DataFrame, params: StrategyParams, instrument: InstrumentSpec
    ) -> EventHandler:
        return _MeanReversionHandler(self._params(params))


class _MeanReversionHandler(EventHandler):
    def __init__(self, p: LimitMeanReversionParams) -> None:
        self._p = p
        self._closes: list[float] = []
        self._placed_at: int | None = None
        self._last_bar = -1
        self._filled_bar = -1

    def on_fill(self, ctx: StrategyContext, fill: FillView) -> None:
        if self._placed_at is not None:
            # Entry fills arrive while the engine replays the bar after the last one delivered.
            self._filled_bar = self._last_bar + 1
            if ctx.open_order_count() == 0:
                self._placed_at = None

    def on_bar(self, ctx: StrategyContext, bar: BarView) -> None:
        p = self._p
        self._last_bar = bar.index
        self._closes.append(bar.close)
        if len(self._closes) > p.lookback:
            del self._closes[0]

        if self._placed_at is not None and ctx.open_order_count() == 0:
            self._placed_at = None
        if self._placed_at is not None and bar.index - self._placed_at >= p.order_ttl_bars:
            ctx.cancel_all()
            self._placed_at = None
        if len(self._closes) < p.lookback:
            return
        z = zscore_last(np.asarray(self._closes, dtype="float64"))
        position = ctx.position
        if abs(position) >= ctx.instrument.lot_size / 2:
            if bar.index > self._filled_bar and should_exit(position, z, p):
                ctx.cancel_all()
                ctx.submit_market(Side.SELL if position > 0 else Side.BUY, abs(position))
            return
        if self._placed_at is None and ctx.open_order_count() == 0:
            side = entry_side(z, p)
            if side is not None:
                ctx.submit_limit(side, p.qty, entry_price(bar.close, side, p, ctx.instrument))
                self._placed_at = bar.index

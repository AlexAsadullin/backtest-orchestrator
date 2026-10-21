"""Engine-agnostic strategy API.

A strategy declares its parameters as a pydantic model and implements one or both of

* :meth:`Strategy.vectorised` - a causal target-position function over the whole bar frame (used by the
  vectorised screen, and replayed bar by bar on the event engines by :class:`TargetFollower`);
* :meth:`Strategy.event_handler` - bar/fill callbacks that place market and limit orders through a
  :class:`StrategyContext`, which every event or tick engine adapter implements.

Engines never see strategy internals; strategies never see engine objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from bt_orchestrator.domain import InstrumentSpec, Liquidity


class Side(IntEnum):
    BUY = 1
    SELL = -1


class StrategyParams(BaseModel):
    """Base for strategy parameter declarations; concrete models list typed, bounded fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class BarView:
    """A completed bar as delivered to a strategy at its close time."""

    index: int
    ts: int
    ts_close: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FillView:
    ts: int
    side: Side
    qty: float
    price: float
    liquidity: Liquidity
    order_ref: str


class StrategyContext(Protocol):
    """What an engine exposes to a running strategy."""

    @property
    def instrument(self) -> InstrumentSpec: ...

    @property
    def now_ns(self) -> int: ...

    @property
    def position(self) -> float:
        """Filled position in base units (signed)."""
        ...

    @property
    def working_qty(self) -> float:
        """Signed unfilled quantity of open orders (orders in flight count as working)."""
        ...

    def open_order_count(self) -> int:
        """Open orders, not counting those with a cancel already requested."""
        ...

    def submit_market(self, side: Side, qty: float) -> str: ...

    def submit_limit(self, side: Side, qty: float, price: float) -> str: ...

    def cancel_all(self) -> None: ...


class EventHandler(ABC):
    """Bar and fill callbacks of an event-driven strategy."""

    @abstractmethod
    def on_bar(self, ctx: StrategyContext, bar: BarView) -> None: ...

    def on_fill(self, ctx: StrategyContext, fill: FillView) -> None:  # noqa: B027 - optional hook
        """Called for every (partial) fill before the next bar is delivered."""


@dataclass(frozen=True)
class VectorSignal:
    """Output of a vectorised strategy, aligned with the input bars.

    ``target[i]`` is the position to hold after bar ``i`` (NaN: place no order on that bar).
    ``price[i]`` is the execution price of that order (NaN or ``None``: the bar close).
    ``maker[i]`` marks passive (maker-fee) executions; ``None`` means all taker.
    """

    target: NDArray[np.float64]
    price: NDArray[np.float64] | None = None
    maker: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        n = len(self.target)
        for name in ("price", "maker"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(f"VectorSignal.{name} length {len(arr)} != target length {n}")


class Strategy(ABC):
    """A named strategy with declared parameters."""

    name: ClassVar[str]
    params_model: ClassVar[type[StrategyParams]]

    def parse_params(
        self, raw: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
    ) -> StrategyParams:
        """Merge request params over configured defaults and validate them."""
        merged = {**(defaults or {}), **raw}
        return self.params_model.model_validate(merged)

    @property
    def supports_vectorised(self) -> bool:
        return type(self).vectorised is not Strategy.vectorised

    def vectorised(
        self, bars: pd.DataFrame, params: StrategyParams, instrument: InstrumentSpec
    ) -> VectorSignal:
        """Vectorised target positions; strategies without one raise ``NotImplementedError``."""
        raise NotImplementedError(f"strategy {self.name!r} has no vectorised form")

    def event_handler(
        self, bars: pd.DataFrame, params: StrategyParams, instrument: InstrumentSpec
    ) -> EventHandler:
        """Event callbacks for the bars an engine is about to replay.

        The default replays the vectorised target positions with market orders, so a strategy that only
        implements :meth:`vectorised` still runs on every event and tick engine.
        """
        signal = self.vectorised(bars, params, instrument)
        return TargetFollower(bars["ts"].to_numpy(), signal.target)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params_schema": self.params_model.model_json_schema(),
            "vectorised": self.supports_vectorised,
        }


class TargetFollower(EventHandler):
    """Moves the position to precomputed per-bar targets with market orders.

    Targets are keyed by bar open time, so engines that build their own bars (from trades or a
    simulator tape) stay aligned with the vectorised signal.
    """

    def __init__(self, bar_ts: NDArray[np.int64], target: NDArray[np.float64]) -> None:
        if len(bar_ts) != len(target):
            raise ValueError("bar_ts and target must have the same length")
        self._targets = {
            int(ts): float(t) for ts, t in zip(bar_ts, target, strict=True) if np.isfinite(t)
        }

    def on_bar(self, ctx: StrategyContext, bar: BarView) -> None:
        target = self._targets.get(bar.ts)
        if target is None:
            return
        delta = ctx.instrument.round_qty(target - ctx.position - ctx.working_qty)
        if abs(delta) < ctx.instrument.lot_size / 2:
            return
        ctx.submit_market(Side.BUY if delta > 0 else Side.SELL, abs(delta))

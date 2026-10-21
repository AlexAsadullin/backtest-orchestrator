"""The engine protocol and the common result format every adapter returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from bt_orchestrator.calibration.models import FillModelBundle
from bt_orchestrator.domain import CostSpec, Fidelity, InstrumentSpec, Tape
from bt_orchestrator.strategies.base import Strategy, StrategyParams


@dataclass(frozen=True)
class EngineJob:
    """One strategy on one symbol, with everything the engine needs."""

    strategy: Strategy
    params: StrategyParams
    instrument: InstrumentSpec
    bars: pd.DataFrame
    bar_interval_ns: int
    costs: CostSpec
    initial_cash: float
    seed: int
    tape: Tape | None = None
    fill_models: FillModelBundle | None = None


@dataclass
class EngineResult:
    """Fills and a per-bar mark-to-market ledger on the engine's own bar grid.

    ``ledger`` columns: ``ts`` (bar open), ``ts_close``, ``close``, ``position``, ``cash``, ``equity``,
    ``ret``. ``native`` holds what the engine itself reports (final PnL, fees, fill count) so parity
    checks can compare engines on their own accounting as well as on the common ledger.
    """

    engine: str
    fidelity: Fidelity
    symbol: str
    fills: pd.DataFrame
    ledger: pd.DataFrame
    native: dict[str, float]
    wall_time_s: float
    initial_cash: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return pd.Series(self.ledger["ret"].to_numpy(), index=self.ledger["ts_close"].to_numpy())

    @property
    def final_pnl(self) -> float:
        return float(self.ledger["equity"].iloc[-1] - self.initial_cash)


class Engine(Protocol):
    name: str
    fidelity: Fidelity

    def run(self, job: EngineJob) -> EngineResult: ...


def mark_to_market(
    fills: pd.DataFrame, bars: pd.DataFrame, bar_interval_ns: int, initial_cash: float
) -> pd.DataFrame:
    """Per-bar position, cash and equity from fills, marked at each bar close.

    A fill at time ``t`` belongs to the first bar whose close time is ``>= t``; fills after the last
    close are booked on the last bar. ``ret`` is the simple equity return per bar (the first bar against
    the initial cash).
    """
    ts_close = bars["ts"].to_numpy(dtype="int64") + bar_interval_ns
    n = len(bars)
    d_pos = np.zeros(n)
    d_cash = np.zeros(n)
    if not fills.empty:
        k = np.minimum(np.searchsorted(ts_close, fills["ts"].to_numpy(), side="left"), n - 1)
        signed = fills["side"].to_numpy() * fills["qty"].to_numpy()
        np.add.at(d_pos, k, signed)
        np.add.at(d_cash, k, -signed * fills["price"].to_numpy() - fills["fee"].to_numpy())
    position = np.cumsum(d_pos)
    cash = initial_cash + np.cumsum(d_cash)
    close = bars["close"].to_numpy(dtype="float64")
    equity = cash + position * close
    prev = np.r_[initial_cash, equity[:-1]]
    return pd.DataFrame(
        {
            "ts": bars["ts"].to_numpy(dtype="int64"),
            "ts_close": ts_close,
            "close": close,
            "position": position,
            "cash": cash,
            "equity": equity,
            "ret": equity / prev - 1.0,
        }
    )

"""Core value types shared by strategies, engines and analytics.

Conventions
-----------
* Bars are a DataFrame with columns ``ts`` (bar *open* time, int64 ns), ``open``, ``high``, ``low``,
  ``close``, ``volume``. A bar is knowable at ``ts + interval`` (its close time).
* Fills are a DataFrame with columns listed in :data:`FILL_COLUMNS`. ``side`` is +1 (buy) or -1 (sell);
  ``qty`` is always positive; ``fee`` is in quote currency.
* Money uses float64: this package does analytics, not account keeping, so binary floating point is
  adequate and keeps all engines on one numeric type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class Fidelity(StrEnum):
    """Rungs of the fidelity ladder, cheapest first (contract 6.5 names)."""

    VECTORISED = "vectorised"
    EVENT = "event"
    TICK = "tick"
    REACTIVE_SIM = "reactive_sim"


class Liquidity(StrEnum):
    MAKER = "maker"
    TAKER = "taker"


BAR_COLUMNS: tuple[str, ...] = ("ts", "open", "high", "low", "close", "volume")
FILL_COLUMNS: tuple[str, ...] = ("ts", "side", "qty", "price", "fee", "liquidity", "order_ref")


@dataclass(frozen=True)
class InstrumentSpec:
    """Tradable instrument as the engines need it."""

    symbol: str
    venue: str
    base: str
    quote: str
    tick_size: float
    lot_size: float
    price_precision: int
    size_precision: int

    def round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, self.price_precision)

    def round_qty(self, qty: float) -> float:
        return round(round(qty / self.lot_size) * self.lot_size, self.size_precision)


@dataclass(frozen=True)
class CostSpec:
    """Proportional fees on traded notional (rates, e.g. 0.0005 = 5 bps)."""

    maker_fee: float
    taker_fee: float

    def fee(self, notional: float, liquidity: Liquidity) -> float:
        rate = self.maker_fee if liquidity is Liquidity.MAKER else self.taker_fee
        return abs(notional) * rate


@dataclass(frozen=True)
class Tape:
    """Tick data for one symbol in the I1 tape schema (contract 6.2): trades and book deltas."""

    symbol: str
    trades: pd.DataFrame
    book: pd.DataFrame


@dataclass
class DataBundle:
    """Everything an engine run needs for one symbol, plus the fingerprints that produced it."""

    symbol: str
    bars: pd.DataFrame
    tape: Tape | None = None
    fingerprints: dict[str, str] = field(default_factory=dict)


def empty_fills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="int64"),
            "side": pd.Series(dtype="int64"),
            "qty": pd.Series(dtype="float64"),
            "price": pd.Series(dtype="float64"),
            "fee": pd.Series(dtype="float64"),
            "liquidity": pd.Series(dtype="object"),
            "order_ref": pd.Series(dtype="object"),
        }
    )


def make_fills(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a fills frame with the canonical columns and dtypes, sorted by time."""
    if not rows:
        return empty_fills()
    frame = pd.DataFrame(rows, columns=list(FILL_COLUMNS))
    frame = frame.astype(
        {"ts": "int64", "side": "int64", "qty": "float64", "price": "float64", "fee": "float64"}
    )
    return frame.sort_values("ts", kind="stable").reset_index(drop=True)


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Check the bar schema and ordering; return bars with a clean RangeIndex."""
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars are missing columns {missing}")
    if bars.empty:
        raise ValueError("bars are empty")
    ts = bars["ts"].to_numpy()
    if not np.all(np.diff(ts) > 0):
        raise ValueError("bar timestamps must be strictly increasing")
    return bars.loc[:, list(BAR_COLUMNS)].reset_index(drop=True)

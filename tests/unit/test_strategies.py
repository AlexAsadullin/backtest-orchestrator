from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from bt_orchestrator.domain import InstrumentSpec, Liquidity
from bt_orchestrator.orchestrator import LookAheadError, check_causality
from bt_orchestrator.strategies.base import (
    BarView,
    FillView,
    Side,
    Strategy,
    StrategyParams,
    TargetFollower,
    VectorSignal,
)
from bt_orchestrator.strategies.limit_mean_reversion import (
    LimitMeanReversion,
    LimitMeanReversionParams,
    entry_price,
    entry_side,
    should_exit,
    zscore_last,
)
from bt_orchestrator.strategies.registry import default_registry
from bt_orchestrator.strategies.sma_cross import SmaCross, SmaCrossParams

INST = InstrumentSpec("BTC-USDT-PERP", "SYNTH", "BTC", "USDT", 0.1, 0.001, 1, 3)
LMR = {"lookback": 5, "entry_z": 1.0, "exit_z": 0.25, "offset_bps": 1.0, "qty": 1.0, "order_ttl_bars": 1}


def bars_from_close(close: list[float] | np.ndarray, spread: float = 0.5) -> pd.DataFrame:
    c = np.asarray(close, dtype="float64")
    return pd.DataFrame(
        {
            "ts": np.arange(len(c), dtype="int64") * 60_000_000_000,
            "open": c,
            "high": c + spread,
            "low": c - spread,
            "close": c,
            "volume": 1.0,
        }
    )


@dataclass
class FakeContext:
    """A minimal StrategyContext that fills market orders immediately and keeps limits working."""

    position: float = 0.0
    working_qty: float = 0.0
    orders: list[tuple[str, int, float, float | None]] = field(default_factory=list)
    cancels: int = 0
    open_orders: int = 0
    now_ns: int = 0

    @property
    def instrument(self) -> InstrumentSpec:
        return INST

    def open_order_count(self) -> int:
        return self.open_orders

    def submit_market(self, side: Side, qty: float) -> str:
        self.orders.append(("market", int(side), qty, None))
        self.position += side * qty
        return "m"

    def submit_limit(self, side: Side, qty: float, price: float) -> str:
        self.orders.append(("limit", int(side), qty, price))
        self.open_orders += 1
        return "l"

    def cancel_all(self) -> None:
        self.cancels += 1
        self.open_orders = 0


def bar(i: int, close: float) -> BarView:
    return BarView(i, i * 60, (i + 1) * 60, close, close + 0.5, close - 0.5, close, 1.0)


def test_registry_lookup() -> None:
    reg = default_registry()
    assert reg.names() == ["limit_mean_reversion", "sma_cross"]
    assert "sma_cross" in reg
    with pytest.raises(KeyError, match="known"):
        reg.get("nope")
    desc = reg.get("sma_cross").describe()
    assert desc["vectorised"] is True
    assert "properties" in desc["params_schema"]


def test_sma_params_validation() -> None:
    with pytest.raises(ValidationError, match="fast must be smaller"):
        SmaCrossParams(fast=10, slow=5, qty=1.0)
    with pytest.raises(ValidationError):
        SmaCross().parse_params({"fast": 2}, {"slow": 5, "qty": 1.0, "extra": 1})


def test_sma_cross_targets_follow_the_trend() -> None:
    close = np.r_[np.linspace(100, 110, 20), np.linspace(110, 95, 20)]
    sig = SmaCross().vectorised(bars_from_close(close), SmaCrossParams(fast=2, slow=5, qty=2.0), INST)
    assert np.all(sig.target[:4] == 0.0)
    assert sig.target[10] == 2.0
    assert sig.target[-1] == -2.0
    assert sig.price is None


def test_vector_signal_length_check() -> None:
    with pytest.raises(ValueError, match="length"):
        VectorSignal(target=np.zeros(3), price=np.zeros(2))


def test_target_follower_trades_the_difference() -> None:
    follower = TargetFollower(np.array([0, 60, 120]), np.array([1.0, np.nan, -1.0]))
    ctx = FakeContext()
    follower.on_bar(ctx, bar(0, 100.0))
    follower.on_bar(ctx, bar(1, 100.0))
    follower.on_bar(ctx, bar(2, 100.0))
    assert ctx.orders == [("market", 1, 1.0, None), ("market", -1, 2.0, None)]
    follower.on_bar(ctx, bar(2, 100.0))
    assert len(ctx.orders) == 2
    with pytest.raises(ValueError, match="same length"):
        TargetFollower(np.array([0]), np.array([1.0, 2.0]))


def test_default_event_handler_replays_vectorised_targets() -> None:
    handler = SmaCross().event_handler(
        bars_from_close([1.0, 2.0, 3.0]), SmaCrossParams(fast=1, slow=2, qty=1), INST
    )
    assert isinstance(handler, TargetFollower)


class EventOnly(Strategy):
    name = "event_only"
    params_model = StrategyParams


def test_strategy_without_vectorised_form() -> None:
    s = EventOnly()
    assert not s.supports_vectorised
    with pytest.raises(NotImplementedError):
        s.vectorised(bars_from_close([1.0]), StrategyParams(), INST)
    check_causality(s, bars_from_close([1.0, 2.0]), StrategyParams(), INST, 3)


class Peeking(Strategy):
    name = "peeking"
    params_model = StrategyParams

    def vectorised(
        self, bars: pd.DataFrame, params: StrategyParams, instrument: InstrumentSpec
    ) -> VectorSignal:
        return VectorSignal(target=np.full(len(bars), float(bars["close"].iloc[-1])))


def test_causality_check_catches_look_ahead() -> None:
    bars = bars_from_close(np.linspace(100, 120, 40))
    with pytest.raises(LookAheadError, match="future bars"):
        check_causality(Peeking(), bars, StrategyParams(), INST, 3)
    check_causality(SmaCross(), bars, SmaCrossParams(fast=2, slow=5, qty=1), INST, 3)


def test_mean_reversion_helpers() -> None:
    p = LimitMeanReversionParams(**LMR)
    assert zscore_last(np.ones(5)) == 0.0
    assert zscore_last(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert entry_side(-2.0, p) is Side.BUY
    assert entry_side(2.0, p) is Side.SELL
    assert entry_side(0.0, p) is None
    assert should_exit(1.0, 0.0, p)
    assert should_exit(-1.0, 0.0, p)
    assert not should_exit(1.0, -1.0, p)
    assert entry_price(100.04, Side.BUY, p, INST) == pytest.approx(100.0)
    assert entry_price(100.04, Side.SELL, p, INST) == pytest.approx(100.1)
    with pytest.raises(ValidationError, match="exit_z"):
        LimitMeanReversionParams(**(LMR | {"exit_z": 2.0}))


def test_mean_reversion_vectorised_touch_fills() -> None:
    close = [100, 100.2, 99.8, 100.1, 99.9, 97.0, 96.0, 99.0, 100.0, 100.0, 100.0]
    bars = bars_from_close(close, spread=2.0)
    sig = LimitMeanReversion().vectorised(bars, LimitMeanReversionParams(**LMR), INST)
    entries = np.flatnonzero(sig.target == 1.0)
    assert len(entries) == 1
    i = entries[0]
    assert sig.maker is not None
    assert sig.maker[i]
    assert sig.price is not None
    assert sig.price[i] <= close[i - 1]
    exits = np.flatnonzero(sig.target == 0.0)
    assert len(exits) == 1
    assert exits[0] > i
    assert sig.maker is not None
    assert not sig.maker[exits[0]]


def test_mean_reversion_vectorised_order_expires() -> None:
    close = [100, 100.2, 99.8, 100.1, 99.9, 97.0, 97.0, 97.0]
    sig = LimitMeanReversion().vectorised(
        bars_from_close(close, spread=0.0), LimitMeanReversionParams(**LMR), INST
    )
    assert np.all(np.isnan(sig.target))


def test_mean_reversion_event_handler_life_cycle() -> None:
    p = LimitMeanReversionParams(**LMR)
    handler = LimitMeanReversion().event_handler(bars_from_close([1.0]), p, INST)
    ctx = FakeContext()
    closes = [100, 100.2, 99.8, 100.1, 99.9, 97.0]
    for i, c in enumerate(closes):
        handler.on_bar(ctx, bar(i, c))
    assert ctx.orders[-1][0] == "limit"
    assert ctx.orders[-1][1] == 1
    # The order is still working one bar later, so the time-to-live cancels it.
    handler.on_bar(ctx, bar(6, 97.0))
    assert ctx.cancels == 1
    # A fresh entry, then a fill, then the exit after reversion.
    handler.on_bar(ctx, bar(7, 95.0))
    assert ctx.orders[-1][0] == "limit"
    ctx.position, ctx.open_orders = 1.0, 0
    handler.on_fill(ctx, FillView(0, Side.BUY, 1.0, 95.0, Liquidity.MAKER, "l"))
    handler.on_bar(ctx, bar(8, 95.0))
    handler.on_bar(ctx, bar(9, 101.0))
    assert ctx.orders[-1] == ("market", -1, 1.0, None)
    assert ctx.position == pytest.approx(0.0)

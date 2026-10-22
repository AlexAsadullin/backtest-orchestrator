"""The parity classifier names each kind of disagreement it is meant to find."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bt_orchestrator.domain import Fidelity, InstrumentSpec, make_fills
from bt_orchestrator.engines.base import EngineResult, mark_to_market
from bt_orchestrator.parity import compare_runs, executions
from bt_orchestrator.settings import Settings

INST = InstrumentSpec("BTC-USDT-PERP", "SYNTH", "BTC", "USDT", 0.1, 0.001, 1, 3)
BAR = 60
CASH = 1_000_000.0


def bars(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * BAR,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0 + np.arange(n),
            "volume": 1.0,
        }
    )


def fill(ts: int, side: int, qty: float = 1.0, price: float = 100.0, fee: float = 0.05) -> dict[str, object]:
    return {
        "ts": ts,
        "side": side,
        "qty": qty,
        "price": price,
        "fee": fee,
        "liquidity": "taker",
        "order_ref": "o",
    }


def result(rows: list[dict[str, object]], engine: str = "a") -> EngineResult:
    fills = make_fills(rows)
    return EngineResult(
        engine=engine,
        fidelity=Fidelity.VECTORISED,
        symbol=INST.symbol,
        fills=fills,
        ledger=mark_to_market(fills, bars(), BAR, CASH),
        native={"pnl": 0.0},
        wall_time_s=0.0,
        initial_cash=CASH,
    )


BASE = [fill(BAR, 1), fill(5 * BAR, -1), fill(10 * BAR, 1), fill(15 * BAR, -1)]


def test_identical_runs_pass(settings: Settings) -> None:
    rep = compare_runs(result(BASE), result(BASE, "b"), INST, settings.parity, "s")
    assert rep.passed
    assert rep.n_identical == 4
    assert rep.counts == {}


def test_executions_group_partial_fills() -> None:
    ts_close = bars()["ts"].to_numpy() + BAR
    x = executions(make_fills([fill(BAR, 1, 0.5, 100.0), fill(BAR, 1, 0.5, 102.0)]), ts_close)
    assert len(x) == 1
    assert x["vwap"].iloc[0] == pytest.approx(101.0)
    assert x["n_fills"].iloc[0] == 2
    assert executions(make_fills([]), ts_close).empty


def test_each_kind_is_classified(settings: Settings) -> None:
    b = [
        fill(BAR, 1, 0.5, fee=0.025),
        fill(BAR, 1, 0.5, fee=0.025),  # partial fill
        fill(5 * BAR, -1, price=100.05, fee=0.050025),  # rounding
        fill(10 * BAR, 1, price=101.0, fee=0.0505),  # fill price
        fill(17 * BAR, -1),  # fill timing (two bars late)
    ]
    rep = compare_runs(result(BASE), result(b, "b"), INST, settings.parity, "s")
    assert rep.counts == {"partial_fill": 1, "rounding": 1, "fill_price": 1, "fill_timing": 1}
    assert not rep.passed
    assert rep.material_disagreement_rate == pytest.approx(0.5)


def test_fee_quantity_and_missing(settings: Settings) -> None:
    b = [fill(BAR, 1, fee=0.5), fill(5 * BAR, -1, qty=2.0, fee=0.1), fill(12 * BAR, 1, qty=3.0)]
    rep = compare_runs(result(BASE), result(b, "b"), INST, settings.parity, "s")
    assert rep.counts["fee"] == 1
    assert rep.counts["quantity"] == 1
    assert rep.counts["missing_in_b"] == 2
    assert rep.counts["missing_in_a"] == 1
    assert rep.position_mismatch_bars > 0


def test_different_bar_grids_are_refused(settings: Settings) -> None:
    a = result(BASE)
    b = result(BASE)
    b.ledger = b.ledger.assign(ts=b.ledger["ts"] + 1)
    with pytest.raises(ValueError, match="same bar grid"):
        compare_runs(a, b, INST, settings.parity, "s")

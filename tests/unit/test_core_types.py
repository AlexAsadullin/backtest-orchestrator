from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from bt_orchestrator.domain import (
    CostSpec,
    InstrumentSpec,
    Liquidity,
    empty_fills,
    make_fills,
    validate_bars,
)
from bt_orchestrator.engines.base import mark_to_market
from bt_orchestrator.timeutil import ns_to_datetime, ns_to_iso, parse_iso, to_ns, utc_now

INST = InstrumentSpec("BTC-USDT-PERP", "SYNTH", "BTC", "USDT", 0.1, 0.001, 1, 3)


def test_time_round_trip() -> None:
    ns = to_ns("2026-01-05T00:00:00Z")
    assert ns_to_iso(ns) == "2026-01-05T00:00:00Z"
    assert ns_to_datetime(ns) == datetime(2026, 1, 5, tzinfo=UTC)
    assert parse_iso(datetime(2026, 1, 5, tzinfo=UTC)).tzinfo is UTC
    assert utc_now().tzinfo is UTC


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_iso("2026-01-05T00:00:00")


def test_instrument_rounding() -> None:
    assert INST.round_price(100.04) == pytest.approx(100.0)
    assert INST.round_price(100.06) == pytest.approx(100.1)
    assert INST.round_qty(0.0014) == pytest.approx(0.001)


def test_cost_spec_fee_by_liquidity() -> None:
    costs = CostSpec(maker_fee=0.0002, taker_fee=0.0005)
    assert costs.fee(-1000.0, Liquidity.MAKER) == pytest.approx(0.2)
    assert costs.fee(1000.0, Liquidity.TAKER) == pytest.approx(0.5)


def test_make_fills_sorts_and_types() -> None:
    assert make_fills([]).equals(empty_fills())
    rows = [
        {"ts": 2, "side": 1, "qty": 1, "price": 10, "fee": 0, "liquidity": "taker", "order_ref": "b"},
        {"ts": 1, "side": -1, "qty": 1, "price": 11, "fee": 0, "liquidity": "maker", "order_ref": "a"},
    ]
    fills = make_fills(rows)
    assert fills["order_ref"].tolist() == ["a", "b"]
    assert fills["price"].dtype == np.float64


def _bars(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * 60,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0, 102.0, 101.0, 103.0][:n],
            "volume": 1.0,
        }
    )


def test_validate_bars() -> None:
    assert len(validate_bars(_bars())) == 4
    with pytest.raises(ValueError, match="missing"):
        validate_bars(_bars().drop(columns=["volume"]))
    with pytest.raises(ValueError, match="empty"):
        validate_bars(_bars().iloc[:0])
    with pytest.raises(ValueError, match="increasing"):
        validate_bars(_bars().iloc[::-1])


def test_mark_to_market_books_fills_on_bar_close() -> None:
    common = {"qty": 1.0, "liquidity": "taker"}
    fills = make_fills(
        [
            {"ts": 60, "side": 1, "price": 100.0, "fee": 1.0, "order_ref": "x"} | common,
            {"ts": 999, "side": -1, "price": 103.0, "fee": 0.0, "order_ref": "y"} | common,
        ]
    )
    ledger = mark_to_market(fills, _bars(), 60, 1000.0)
    assert ledger["position"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert ledger["equity"].iloc[0] == pytest.approx(999.0)
    assert ledger["equity"].iloc[-1] == pytest.approx(1002.0)
    assert ledger["ret"].iloc[0] == pytest.approx(-0.001)

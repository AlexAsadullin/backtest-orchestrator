"""MOEX session calendar and the session-mapped synthetic market."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bt_orchestrator.data.session import SessionCalendar
from bt_orchestrator.data.synthetic import MarketSpec, generate_tape
from bt_orchestrator.data.tape import bars_from_trades
from bt_orchestrator.orchestrator import instrument_spec
from bt_orchestrator.settings import Settings
from bt_orchestrator.timeutil import to_ns

MAIN = SessionCalendar.from_strings("07:00", "15:40", [0, 1, 2, 3, 4])
MINUTE = 60_000_000_000


def test_trading_time_skips_nights_and_weekends() -> None:
    friday_open = to_ns("2026-01-09T07:00:00Z")
    length = MAIN.length_ns
    wall = MAIN.to_wall(friday_open, np.array([0, length - MINUTE, length, length + MINUTE]))
    assert [pd.Timestamp(t, unit="ns", tz="UTC").isoformat() for t in wall] == [
        "2026-01-09T07:00:00+00:00",
        "2026-01-09T15:39:00+00:00",
        "2026-01-12T07:00:00+00:00",  # Monday
        "2026-01-12T07:01:00+00:00",
    ]
    (end,) = MAIN.to_wall(friday_open, np.array([length]), closing=True)
    assert end == to_ns("2026-01-09T15:40:00Z")
    assert MAIN.seconds_per_day == 31200.0


def test_in_session_and_validation() -> None:
    ts = np.array(
        [to_ns(t) for t in ("2026-01-05T06:59:00Z", "2026-01-05T07:00:00Z", "2026-01-10T08:00:00Z")]
    )
    assert MAIN.in_session(ts).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="inside a session"):
        MAIN.to_wall(to_ns("2026-01-05T03:00:00Z"), np.array([0]))
    with pytest.raises(ValueError, match="open before"):
        SessionCalendar.from_strings("10:00", "09:00", [0])
    with pytest.raises(ValueError, match="weekdays"):
        SessionCalendar.from_strings("07:00", "15:40", [7])


def test_synthetic_moex_tape_trades_only_in_session(settings: Settings) -> None:
    symbol = "LKOH-RUB-TQBR"
    spec = MarketSpec.from_settings(
        settings.synthetic,
        symbol,
        settings.market.venue,
        settings.bar_interval_ns,
        settings.market.calendar(),
    )
    spec = MarketSpec(**{**spec.__dict__, "n_bars": 600})  # past the first session close
    assert spec.initial_price == 6038.0  # per-symbol override
    tape = generate_tape(spec, instrument_spec(settings, symbol))
    cal = settings.market.calendar()
    assert cal is not None
    assert cal.in_session(tape.trades["ts_exchange"].to_numpy()).all()
    assert (tape.trades["price"] % 0.5 == 0).all()
    bars = bars_from_trades(tape.trades, spec.start_ns, spec.end_ns, settings.bar_interval_ns, cal)
    assert len(bars) == 600
    assert cal.in_session(bars["ts"].to_numpy()).all()

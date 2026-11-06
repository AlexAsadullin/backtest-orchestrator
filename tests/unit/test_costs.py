"""Fee schedules: default rates, the MOEX T-Invest tariff and the zero-fee set of T-Bank's own funds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bt_orchestrator.costs import FeeBook, load_zero_fee_set
from bt_orchestrator.domain import CostSpec, Liquidity
from bt_orchestrator.engines.nautilus_engine import nautilus_venue
from bt_orchestrator.settings import Settings
from bt_orchestrator.wiring import build_orchestrator
from conftest import make_settings

REFERENCE = "market-tape-recorder/reference/tbank_zero_fee_funds.json"


def write_reference(data_dir: Path, funds: list[dict[str, object]]) -> None:
    path = data_dir / REFERENCE
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "source": "test",
        "retrieved_at": "2026-11-05T00:00:00Z",
        "rule": "T-Capital funds",
        "funds": funds,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_config_rates(settings: Settings) -> None:
    c = settings.costs
    assert (c.maker_fee, c.taker_fee) == (0.0002, 0.0005)
    for venue in ("moex-tbank", "moex-algopack", "moex-iss"):
        sched = c.venues[venue]
        assert (sched.maker_fee, sched.taker_fee) == (0.0025, 0.0025)
        assert sched.source == "T-Invest basic tariff per user, 2026-11-05"
        assert sched.zero_fee_funds is not None
        assert (sched.zero_fee_funds.maker_fee, sched.zero_fee_funds.taker_fee) == (0.0, 0.0)


def test_moex_tariff_and_defaults(settings: Settings) -> None:
    book = FeeBook(settings.costs, settings.data_dir)
    assert book.for_instrument("SYNTH", "BTC-USDT-PERP") == CostSpec(0.0002, 0.0005)
    sber = book.for_instrument("moex-tbank", "SBER-RUB-TQBR")
    assert sber == CostSpec(0.0025, 0.0025)
    assert sber.fee(-28_500.0, Liquidity.TAKER) == pytest.approx(71.25)
    # A config-listed T-Capital fund pays nothing even without the reference file.
    assert book.for_instrument("moex-tbank", "TMOS-RUB-TQTF") == CostSpec(0.0, 0.0)


def test_reference_file_extends_the_zero_fee_set(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_reference(
        data, [{"ticker": "TPAY", "figi": "F1", "instrument_uid": "uid-tpay", "class_code": "TQTF"}]
    )
    s = make_settings(data)
    book = FeeBook(s.costs, s.data_dir)
    assert book.for_instrument("moex-tbank", "TPAY-RUB-TQTF") == CostSpec(0.0, 0.0)
    assert book.for_instrument("moex-tbank", "ANY-RUB-TQTF", instrument_uid="uid-tpay") == CostSpec(0.0, 0.0)
    assert book.for_instrument("moex-tbank", "TMOS-RUB-TQTF") == CostSpec(0.0, 0.0)
    assert book.for_instrument("moex-tbank", "GAZP-RUB-TQBR", instrument_uid="other") == CostSpec(
        0.0025, 0.0025
    )


def test_absent_reference_falls_back_and_bad_reference_fails(settings: Settings, tmp_path: Path) -> None:
    funds = settings.costs.venues["moex-tbank"].zero_fee_funds
    assert funds is not None
    zero = load_zero_fee_set(funds, tmp_path / "nowhere")
    assert zero.tickers == frozenset(funds.tickers)
    bad = tmp_path / "bad"
    (bad / REFERENCE).parent.mkdir(parents=True)
    (bad / REFERENCE).write_text(json.dumps({"funds": "TMOS"}), encoding="utf-8")
    with pytest.raises(ValueError, match="funds"):
        load_zero_fee_set(funds, bad)


def test_orchestrator_charges_moex_fees_on_a_moex_venue(tmp_path: Path) -> None:
    s = make_settings(tmp_path / "data", market={"venue": "moex-tbank"})
    orch = build_orchestrator(s)
    assert orch.costs("SBER-RUB-TQBR") == CostSpec(0.0025, 0.0025)
    assert orch.costs("LKOH-RUB-TQBR") == CostSpec(0.0025, 0.0025)
    assert orch.costs("TMOS-RUB-TQBR") == CostSpec(0.0, 0.0)


def test_crypto_profile_restores_the_synthetic_perp_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BT_PROFILE", "crypto")
    s = make_settings(tmp_path / "data")
    assert s.market.venue == "SYNTH"
    assert s.market.session is None
    assert s.market.default_universe == ["BTC-USDT-PERP"]
    assert s.synthetic.initial_price == 60000.0
    assert build_orchestrator(s).costs("BTC-USDT-PERP") == CostSpec(0.0002, 0.0005)
    monkeypatch.setenv("BT_PROFILE", "nope")
    with pytest.raises(FileNotFoundError, match="profile"):
        make_settings(tmp_path / "data")


def test_nautilus_venue_ids_have_no_hyphen() -> None:
    assert str(nautilus_venue("moex-tbank")) == "moex_tbank"
    assert str(nautilus_venue("SYNTH")) == "SYNTH"

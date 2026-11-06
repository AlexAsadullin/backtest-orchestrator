"""Fee schedules: which proportional fee rates a fill pays on a venue and instrument.

``costs.maker_fee`` / ``costs.taker_fee`` are the default rates, used for venues without a schedule (the
synthetic crypto market of the report experiments). ``costs.venues.<venue>`` overrides them per venue;
the MOEX venues use the T-Invest basic tariff (0.25% of trade value per fill). A venue schedule may name
a zero-fee set (T-Bank's own funds, the T-Capital ETFs) by ticker or instrument uid in config, extended
by I1's reference file ``market-tape-recorder/reference/tbank_zero_fee_funds.json`` when it exists
(top-level ``funds``, each with ``ticker`` and ``instrument_uid``). A missing file is not an error: the
config list is used and a warning is logged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from bt_orchestrator.domain import CostSpec
from bt_orchestrator.settings import CostSettings, ZeroFeeFundsSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ZeroFeeSet:
    """Tickers and instrument uids that pay the zero-fee rates."""

    tickers: frozenset[str]
    instrument_uids: frozenset[str]

    def contains(self, ticker: str, instrument_uid: str | None) -> bool:
        return ticker in self.tickers or (
            instrument_uid is not None and instrument_uid in self.instrument_uids
        )


def load_zero_fee_set(cfg: ZeroFeeFundsSettings, data_dir: Path) -> ZeroFeeSet:
    """Config tickers and uids plus those of the reference file (absent file: config only, with a warning)."""
    tickers = set(cfg.tickers)
    uids = set(cfg.instrument_uids)
    if cfg.reference_file is not None:
        path = data_dir / cfg.reference_file
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            funds = doc.get("funds") if isinstance(doc, dict) else None
            if not isinstance(funds, list):
                raise ValueError(f"{path}: expected a top-level 'funds' list")
            for fund in funds:
                if fund.get("ticker"):
                    tickers.add(str(fund["ticker"]))
                if fund.get("instrument_uid"):
                    uids.add(str(fund["instrument_uid"]))
            log.info("zero_fee_reference_loaded", path=str(path), funds=len(funds), source=doc.get("source"))
        else:
            log.warning("zero_fee_reference_missing", path=str(path), fallback_tickers=sorted(cfg.tickers))
    return ZeroFeeSet(frozenset(tickers), frozenset(uids))


class FeeBook:
    """Resolves the :class:`CostSpec` of one instrument on one venue."""

    def __init__(self, cfg: CostSettings, data_dir: Path) -> None:
        self._cfg = cfg
        self._data_dir = data_dir
        self._zero_fee: dict[str, ZeroFeeSet] = {}

    def zero_fee_set(self, venue: str) -> ZeroFeeSet | None:
        schedule = self._cfg.venues.get(venue)
        if schedule is None or schedule.zero_fee_funds is None:
            return None
        if venue not in self._zero_fee:
            self._zero_fee[venue] = load_zero_fee_set(schedule.zero_fee_funds, self._data_dir)
        return self._zero_fee[venue]

    def for_instrument(
        self, venue: str, symbol: str, *, ticker: str | None = None, instrument_uid: str | None = None
    ) -> CostSpec:
        """Rates for ``symbol`` on ``venue``.

        ``ticker`` defaults to the SECID part of a ``<SECID>-RUB-<BOARD>`` symbol.
        """
        schedule = self._cfg.venues.get(venue)
        if schedule is None:
            return CostSpec(self._cfg.maker_fee, self._cfg.taker_fee)
        zero = self.zero_fee_set(venue)
        funds = schedule.zero_fee_funds
        if (
            zero is not None
            and funds is not None
            and zero.contains(ticker or symbol.split("-", 1)[0], instrument_uid)
        ):
            return CostSpec(funds.maker_fee, funds.taker_fee)
        return CostSpec(schedule.maker_fee, schedule.taker_fee)

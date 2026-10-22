"""Loads bars and tick tapes for a backtest through a point-in-time :class:`DataSource`."""

from __future__ import annotations

import hashlib

from bt_orchestrator.contracts.pit import PitQuery
from bt_orchestrator.data.source import DataSource
from bt_orchestrator.data.tape import BOOK_DELTA_COLUMNS, TRADE_COLUMNS
from bt_orchestrator.domain import BAR_COLUMNS, DataBundle, Tape, validate_bars
from bt_orchestrator.settings import PitSettings


def combine_fingerprints(fingerprints: dict[str, str]) -> str | None:
    """One fingerprint per trial: the I2 fingerprint itself, or a hash over several sorted ones."""
    if not fingerprints:
        return None
    values = sorted(set(fingerprints.values()))
    if len(values) == 1:
        return values[0]
    return "sha256:" + hashlib.sha256("|".join(values).encode()).hexdigest()


class MarketData:
    def __init__(self, source: DataSource, cfg: PitSettings) -> None:
        self._source = source
        self._cfg = cfg

    def _query(self, dataset: str, symbol: str, start: str, end: str, known_at: str) -> PitQuery:
        return PitQuery(
            dataset=dataset,
            start=start,
            end=end,
            known_at=known_at,
            filters={"symbol": symbol},
            mode=self._cfg.query_mode,
        )

    def load(self, symbol: str, start: str, end: str, known_at: str, *, with_tape: bool) -> DataBundle:
        ds = self._cfg.datasets
        bars_res = self._source.query(self._query(ds.bars, symbol, start, end, known_at))
        if bars_res.frame.empty:
            raise ValueError(f"no bars for {symbol} in [{start}, {end})")
        bars = validate_bars(bars_res.frame.loc[:, list(BAR_COLUMNS)])
        fingerprints = {"bars": bars_res.fingerprint}
        tape = None
        if with_tape:
            trades = self._source.query(self._query(ds.trades, symbol, start, end, known_at))
            book = self._source.query(self._query(ds.book, symbol, start, end, known_at))
            if trades.frame.empty or book.frame.empty:
                raise ValueError(f"no tick tape for {symbol} in [{start}, {end})")
            tape = Tape(
                symbol=symbol,
                trades=trades.frame.loc[:, list(TRADE_COLUMNS)],
                book=book.frame.loc[:, list(BOOK_DELTA_COLUMNS)],
            )
            fingerprints |= {"trades": trades.fingerprint, "book": book.fingerprint}
        return DataBundle(symbol=symbol, bars=bars, tape=tape, fingerprints=fingerprints)

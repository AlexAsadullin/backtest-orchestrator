"""Loads bars and tick tapes for a backtest through a point-in-time :class:`DataSource`.

Bars come either from a bars dataset or, with ``pit.bars_source: trades``, from I2's ``tape.trades``
aggregated on the configured bar grid (the shipped I2 has no bars dataset). Either way the bundle keeps
I2's own fingerprint of every dataset it read, primary dataset first.
"""

from __future__ import annotations

from bt_orchestrator.contracts.pit import PitFilter, PitQuery
from bt_orchestrator.data.source import DataSource, QueryResult
from bt_orchestrator.data.tape import BOOK_DELTA_COLUMNS, TRADE_COLUMNS, bars_from_trades
from bt_orchestrator.domain import BAR_COLUMNS, DataBundle, Tape, validate_bars
from bt_orchestrator.settings import PitSettings
from bt_orchestrator.timeutil import to_ns


class MarketData:
    def __init__(self, source: DataSource, cfg: PitSettings, *, venue: str, bar_interval_ns: int) -> None:
        self._source = source
        self._cfg = cfg
        self._venue = venue
        self._interval = bar_interval_ns

    def _query(
        self,
        dataset: str,
        symbol: str,
        start: str,
        end: str,
        known_at: str,
        *,
        tape: bool,
        strategy_id: str | None,
    ) -> PitQuery:
        filters = [PitFilter(column="symbol", op="==", value=symbol)]
        if tape:
            filters.append(PitFilter(column="venue", op="==", value=self._venue))
        return PitQuery(
            dataset=dataset,
            start=start,
            end=end,
            known_at=known_at,
            filters=filters,
            mode=self._cfg.query_mode,
            strategy_id=strategy_id,
        )

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        known_at: str,
        *,
        with_tape: bool,
        strategy_id: str | None = None,
    ) -> DataBundle:
        ds = self._cfg.datasets
        results: dict[str, QueryResult] = {}

        def fetch(name: str, dataset: str, tape: bool) -> QueryResult:
            if name not in results:
                q = self._query(dataset, symbol, start, end, known_at, tape=tape, strategy_id=strategy_id)
                results[name] = self._source.query(q)
            return results[name]

        if self._cfg.bars_source == "trades":
            trades = fetch("trades", ds.trades, True)
            if trades.frame.empty:
                raise ValueError(f"no trades for {symbol} in [{start}, {end})")
            bars = bars_from_trades(trades.frame, to_ns(start), to_ns(end), self._interval)
        else:
            raw = fetch("bars", ds.bars, False)
            if raw.frame.empty:
                raise ValueError(f"no bars for {symbol} in [{start}, {end})")
            bars = raw.frame.loc[:, list(BAR_COLUMNS)]
        if bars.empty:
            raise ValueError(f"no bars for {symbol} in [{start}, {end})")
        bars = validate_bars(bars)

        tape_data = None
        if with_tape:
            trades = fetch("trades", ds.trades, True)
            book = fetch("book", ds.book, True)
            if trades.frame.empty or book.frame.empty:
                raise ValueError(f"no tick tape for {symbol} in [{start}, {end})")
            tape_data = Tape(
                symbol=symbol,
                trades=trades.frame.loc[:, list(TRADE_COLUMNS)],
                book=book.frame.loc[:, list(BOOK_DELTA_COLUMNS)],
            )
        flags = tuple(dict.fromkeys(f for r in results.values() for f in r.leakage_flags))
        return DataBundle(
            symbol=symbol,
            bars=bars,
            tape=tape_data,
            fingerprints={name: r.fingerprint for name, r in results.items()},
            leakage_flags=flags,
        )

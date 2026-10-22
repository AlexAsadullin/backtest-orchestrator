"""Helpers for the I1 tape schema (contract 6.2): bar aggregation and the on-disk layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bt_orchestrator.domain import BAR_COLUMNS, Tape

TRADE_COLUMNS: tuple[str, ...] = (
    "venue",
    "symbol",
    "ts_exchange",
    "ts_local",
    "capture_region",
    "connection_id",
    "seq",
    "trade_id",
    "price",
    "size",
    "side",
)
BOOK_DELTA_COLUMNS: tuple[str, ...] = (
    "venue",
    "symbol",
    "ts_exchange",
    "ts_local",
    "capture_region",
    "connection_id",
    "seq",
    "side",
    "price",
    "size",
    "prev_seq",
)


def event_time(frame: pd.DataFrame) -> np.ndarray:
    """Exchange time where present, else local arrival time (int64 ns)."""
    local: np.ndarray = frame["ts_local"].to_numpy(dtype="int64")
    if "ts_exchange" not in frame.columns:
        return local
    exch: np.ndarray = frame["ts_exchange"].to_numpy(dtype="float64")
    return np.where(np.isnan(exch), local, exch).astype("int64")


def bars_from_trades(trades: pd.DataFrame, start_ns: int, end_ns: int, interval_ns: int) -> pd.DataFrame:
    """Aggregate trades into OHLCV bars on the grid ``[start_ns, end_ns)``.

    A bar without trades repeats the previous close with zero volume; bars before the first trade are
    dropped, because no price is known yet.
    """
    if end_ns <= start_ns:
        raise ValueError("end_ns must be after start_ns")
    n_bars = (end_ns - start_ns) // interval_ns
    ts = event_time(trades)
    keep = (ts >= start_ns) & (ts < start_ns + n_bars * interval_ns)
    ts = ts[keep]
    price = trades["price"].to_numpy(dtype="float64")[keep]
    size = trades["size"].to_numpy(dtype="float64")[keep]
    order = np.argsort(ts, kind="stable")
    ts, price, size = ts[order], price[order], size[order]
    bucket = (ts - start_ns) // interval_ns

    frame = pd.DataFrame({"bucket": bucket, "price": price, "size": size})
    grouped = frame.groupby("bucket", sort=True)
    agg = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["size"].sum(),
        }
    ).reindex(np.arange(n_bars))
    agg["close"] = agg["close"].ffill()
    for col in ("open", "high", "low"):
        agg[col] = agg[col].fillna(agg["close"])
    agg["volume"] = agg["volume"].fillna(0.0)
    agg["ts"] = start_ns + agg.index.to_numpy(dtype="int64") * interval_ns
    agg = agg.dropna(subset=["close"])
    return agg.loc[:, list(BAR_COLUMNS)].reset_index(drop=True)


def _partition_dir(root: Path, venue: str, channel: str, symbol: str, date: str) -> Path:
    return root / f"venue={venue}" / f"channel={channel}" / f"symbol={symbol}" / f"date={date}"


def write_i1_tape(tape: Tape, root: Path, venue: str) -> list[Path]:
    """Write a tape in the I1 layout, one Parquet part per channel and UTC date."""
    written: list[Path] = []
    for channel, frame in (("trades", tape.trades), ("book_delta", tape.book)):
        if frame.empty:
            continue
        dates = pd.to_datetime(event_time(frame), unit="ns", utc=True).strftime("%Y-%m-%d")
        for date, part in frame.groupby(np.asarray(dates), sort=True):
            out = _partition_dir(root, venue, channel, tape.symbol, str(date))
            out.mkdir(parents=True, exist_ok=True)
            path = out / "part-0.parquet"
            part.reset_index(drop=True).to_parquet(path, index=False)
            written.append(path)
    return written


def _read_channel(root: Path, channel: str, symbol: str) -> pd.DataFrame:
    parts = sorted(
        p
        for p in root.rglob("*.parquet")
        if f"channel={channel}" in p.parts and f"symbol={symbol}" in p.parts
    )
    if not parts:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return frame.sort_values("ts_local", kind="stable").reset_index(drop=True)


def read_i1_tape(root: Path, symbol: str) -> Tape:
    """Read trades and book deltas for ``symbol`` from any directory holding the I1 layout."""
    trades = _read_channel(root, "trades", symbol)
    book = _read_channel(root, "book_delta", symbol)
    if trades.empty or book.empty:
        raise FileNotFoundError(f"no trades/book_delta for {symbol} under {root}")
    return Tape(symbol=symbol, trades=trades, book=book)

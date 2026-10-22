"""Writes the synthetic market and fills under this project's data subtree.

Market datasets go where the local point-in-time source expects them (``local_source.datasets``), with
``valid_time`` / ``knowledge_time`` columns; fills and orders go to ``<synthetic>/fills`` and
``<synthetic>/orders`` in the I6 export layout (contract 6.4), never into I6's own subtree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from bt_orchestrator.data.synthetic import (
    MarketSpec,
    bars_with_knowledge_times,
    generate_fills,
    generate_tape,
    with_knowledge_times,
    write_date_partitioned,
)
from bt_orchestrator.data.tape import bars_from_trades
from bt_orchestrator.engines.nautilus_engine import average_daily_volume
from bt_orchestrator.orchestrator import instrument_spec
from bt_orchestrator.settings import Settings
from bt_orchestrator.timeutil import ns_to_iso

FILLS_DIR = "fills"
ORDERS_DIR = "orders"


@dataclass(frozen=True)
class SyntheticPaths:
    datasets: dict[str, Path]
    fills_root: Path
    orders_root: Path
    start: str
    end: str
    n_bars: int


def synthetic_fill_roots(settings: Settings) -> tuple[Path, Path]:
    return settings.synthetic_dir / FILLS_DIR, settings.synthetic_dir / ORDERS_DIR


def write_synthetic(settings: Settings, symbols: list[str], n_bars: int | None = None) -> SyntheticPaths:
    """Generate tapes, bars and fills for ``symbols`` and write them (overwriting earlier output)."""
    s = settings
    frames: dict[str, list[pd.DataFrame]] = {"bars": [], "trades": [], "book": []}
    fills, orders = [], []
    spec0: MarketSpec | None = None
    for k, symbol in enumerate(symbols):
        inst = instrument_spec(s, symbol)
        spec = MarketSpec.from_settings(s.synthetic, symbol, s.market.venue, s.bar_interval_ns)
        spec = replace(spec, seed=spec.seed + k, n_bars=n_bars or spec.n_bars)
        spec0 = spec0 or spec
        tape = generate_tape(spec, inst)
        bars = bars_from_trades(tape.trades, spec.start_ns, spec.end_ns, s.bar_interval_ns)
        frames["bars"].append(bars_with_knowledge_times(bars, symbol, s.bar_interval_ns))
        frames["trades"].append(with_knowledge_times(tape.trades, "ts_exchange", "ts_local"))
        frames["book"].append(with_knowledge_times(tape.book, "ts_exchange", "ts_local"))
        synthetic_fills = generate_fills(
            s.synthetic.fills,
            symbol=symbol,
            start_ns=spec.start_ns,
            initial_price=spec.initial_price,
            vol_bps_per_sqrt_s=spec.vol_bps_per_sqrt_s,
            adv=average_daily_volume(bars, s.bar_interval_ns, s.market.seconds_per_day),
            maker_fee=s.costs.maker_fee,
            taker_fee=s.costs.taker_fee,
            seed=spec.seed + 1000,
        )
        fills.append(synthetic_fills.fills)
        orders.append(synthetic_fills.orders)
    if spec0 is None:
        raise ValueError("no symbols to generate")

    ds = s.pit.datasets
    paths: dict[str, Path] = {}
    for key, name in (("bars", ds.bars), ("trades", ds.trades), ("book", ds.book)):
        path = s.project_dir / s.local_source.datasets[name].path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames[key], ignore_index=True).to_parquet(path, index=False)
        paths[name] = path

    fills_root, orders_root = synthetic_fill_roots(s)
    for root in (fills_root, orders_root):
        if root.exists():
            for part in root.glob("date=*/*.parquet"):
                part.unlink()
    write_date_partitioned(pd.concat(fills, ignore_index=True), fills_root, "ts_fill")
    write_date_partitioned(pd.concat(orders, ignore_index=True), orders_root, "ts_submit")
    return SyntheticPaths(
        datasets=paths,
        fills_root=fills_root,
        orders_root=orders_root,
        start=ns_to_iso(spec0.start_ns),
        end=ns_to_iso(spec0.end_ns),
        n_bars=spec0.n_bars,
    )

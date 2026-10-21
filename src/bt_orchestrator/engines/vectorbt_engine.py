"""Vectorised screen on vectorbt: ``Portfolio.from_orders`` with target amounts at bar prices."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import vectorbt as vbt

from bt_orchestrator.domain import Fidelity, Liquidity, make_fills
from bt_orchestrator.engines.base import EngineJob, EngineResult, mark_to_market
from bt_orchestrator.settings import VectorbtSettings

VBT_SELL = 1


class VectorbtEngine:
    """Fills every order at the signal's price (default: the bar close) with proportional fees."""

    name = "vectorbt"
    fidelity = Fidelity.VECTORISED

    def __init__(self, cfg: VectorbtSettings) -> None:
        self._cfg = cfg

    def run(self, job: EngineJob) -> EngineResult:
        started = time.perf_counter()
        bars = job.bars
        signal = job.strategy.vectorised(bars, job.params, job.instrument)
        index = pd.to_datetime(bars["ts"].to_numpy(), unit="ns", utc=True)
        close = bars["close"].to_numpy(dtype="float64")
        price = close if signal.price is None else np.where(np.isnan(signal.price), close, signal.price)
        maker = np.zeros(len(bars), dtype=bool) if signal.maker is None else signal.maker
        fees = np.where(maker, job.costs.maker_fee, job.costs.taker_fee)
        portfolio = vbt.Portfolio.from_orders(
            pd.Series(close, index=index),
            size=pd.Series(signal.target, index=index),
            size_type="targetamount",
            price=pd.Series(price, index=index),
            fees=pd.Series(fees, index=index),
            init_cash=job.initial_cash,
            direction=self._cfg.direction,
            freq=pd.Timedelta(job.bar_interval_ns, unit="ns"),
        )
        records = portfolio.orders.values
        ts_close = bars["ts"].to_numpy(dtype="int64") + job.bar_interval_ns
        rows = [
            {
                "ts": int(ts_close[r["idx"]]),
                "side": -1 if r["side"] == VBT_SELL else 1,
                "qty": float(r["size"]),
                "price": float(r["price"]),
                "fee": float(r["fees"]),
                "liquidity": (Liquidity.MAKER if maker[r["idx"]] else Liquidity.TAKER).value,
                "order_ref": f"vbt-{int(r['id'])}",
            }
            for r in records
        ]
        fills = make_fills(rows)
        ledger = mark_to_market(fills, bars, job.bar_interval_ns, job.initial_cash)
        native = {
            "pnl": float(portfolio.final_value()) - job.initial_cash,
            "fees": float(fills["fee"].sum()),
            "n_fills": float(len(fills)),
        }
        return EngineResult(
            engine=self.name,
            fidelity=self.fidelity,
            symbol=job.instrument.symbol,
            fills=fills,
            ledger=ledger,
            native=native,
            wall_time_s=time.perf_counter() - started,
            initial_cash=job.initial_cash,
        )

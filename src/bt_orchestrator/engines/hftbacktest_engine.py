"""Tick/queue rung on hftbacktest: L2 book replay with queue-position fills and order latency.

The strategy runs at bar boundaries (Python), while hftbacktest replays every book update and trade in
between (Rust), so resting limit orders fill only when the queue ahead of them is consumed.

Calibration is injected in two places: the order entry/response latency is the calibrated latency
quantile, and taker fills pay the calibrated impact term (the slippage not explained by crossing the
spread, which the replayed book already charges). hftbacktest's book does not react to our orders, so
without that overlay a large market order would pay no impact at all.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from hftbacktest import (
    BUY_EVENT,
    DEPTH_EVENT,
    GTC,
    LIMIT,
    MARKET,
    SELL_EVENT,
    TRADE_EVENT,
    BacktestAsset,
    HashMapMarketDepthBacktest,
    event_dtype,
)
from hftbacktest.data.validation import correct_event_order, validate_event_order
from hftbacktest.order import CANCELED, EXPIRED, FILLED, NEW, PARTIALLY_FILLED, REJECTED
from numpy.typing import NDArray

from bt_orchestrator.calibration.models import FillModelBundle
from bt_orchestrator.data.tape import event_time
from bt_orchestrator.domain import CostSpec, Fidelity, InstrumentSpec, Liquidity, Tape, make_fills
from bt_orchestrator.engines.base import EngineJob, EngineResult, mark_to_market
from bt_orchestrator.engines.nautilus_engine import average_daily_volume, bar_volatility_bps_sqrt_s
from bt_orchestrator.settings import HftBacktestSettings
from bt_orchestrator.strategies.base import BarView, EventHandler, FillView, Side

BPS = 1e4
END_OF_DATA = 1
# hftbacktest reports this timestamp before the first elapse; time then starts at the first event.
NOT_STARTED = np.iinfo(np.int64).max
ASSET = 0
_TERMINAL = {FILLED, CANCELED, EXPIRED, REJECTED}
_FILLABLE = {FILLED, PARTIALLY_FILLED, CANCELED, EXPIRED}


def tape_to_events(tape: Tape) -> NDArray[np.void]:
    """Convert I1 trades + book deltas into an hftbacktest event array with valid event order."""
    book, trades = tape.book, tape.trades
    n_b, n_t = len(book), len(trades)
    raw = np.zeros(n_b + n_t, dtype=event_dtype)
    side_b = np.where(book["side"].to_numpy() == "bid", BUY_EVENT, SELL_EVENT)
    side_t = np.where(trades["side"].to_numpy() == "buy", BUY_EVENT, SELL_EVENT)
    raw["ev"] = np.r_[DEPTH_EVENT | side_b, TRADE_EVENT | side_t].astype("uint64")
    exch = np.r_[event_time(book), event_time(trades)]
    local = np.r_[book["ts_local"].to_numpy(dtype="int64"), trades["ts_local"].to_numpy(dtype="int64")]
    raw["exch_ts"] = exch
    raw["local_ts"] = np.maximum(local, exch)
    raw["px"] = np.r_[book["price"].to_numpy(dtype="float64"), trades["price"].to_numpy(dtype="float64")]
    raw["qty"] = np.r_[book["size"].to_numpy(dtype="float64"), trades["size"].to_numpy(dtype="float64")]
    by_exch = np.argsort(raw["exch_ts"], kind="stable")
    by_local = np.argsort(raw["local_ts"], kind="stable")
    events: NDArray[np.void] = correct_event_order(raw, by_exch, by_local)
    validate_event_order(events)
    return events


@dataclass
class _Tracked:
    side: Side
    qty: float
    filled: float = 0.0
    cancel_requested: bool = False
    done: bool = False


class _HbtContext:
    """:class:`StrategyContext` over an hftbacktest instance."""

    def __init__(
        self,
        hbt: object,
        spec: InstrumentSpec,
        costs: CostSpec,
        impact_bps: Callable[[float], float],
    ) -> None:
        self._hbt = hbt
        self._spec = spec
        self._costs = costs
        self._impact = impact_bps
        self._orders: dict[int, _Tracked] = {}
        self._next_id = 1
        self.fills: list[dict[str, object]] = []

    @property
    def instrument(self) -> InstrumentSpec:
        return self._spec

    @property
    def now_ns(self) -> int:
        return int(self._hbt.current_timestamp)  # type: ignore[attr-defined]

    @property
    def position(self) -> float:
        return float(self._hbt.position(ASSET))  # type: ignore[attr-defined]

    def _live(self) -> list[_Tracked]:
        return [o for o in self._orders.values() if not o.done and not o.cancel_requested]

    @property
    def working_qty(self) -> float:
        return float(sum(o.side * (o.qty - o.filled) for o in self._live()))

    def open_order_count(self) -> int:
        return len(self._live())

    def _submit(self, side: Side, qty: float, price: float, order_type: int) -> str:
        oid = self._next_id
        self._next_id += 1
        qty = self._spec.round_qty(qty)
        submit = self._hbt.submit_buy_order if side is Side.BUY else self._hbt.submit_sell_order  # type: ignore[attr-defined]
        rc = submit(ASSET, oid, price, qty, GTC, order_type, False)
        if rc != 0:
            raise RuntimeError(f"hftbacktest rejected order submission (code {rc})")
        self._orders[oid] = _Tracked(side=side, qty=qty)
        return f"hbt-{oid}"

    def submit_market(self, side: Side, qty: float) -> str:
        depth = self._hbt.depth(ASSET)  # type: ignore[attr-defined]
        price = depth.best_ask if side is Side.BUY else depth.best_bid
        if not math.isfinite(price) or price <= 0:
            raise RuntimeError("cannot submit a market order before the book has a touch")
        return self._submit(side, qty, float(price), MARKET)

    def submit_limit(self, side: Side, qty: float, price: float) -> str:
        return self._submit(side, qty, self._spec.round_price(price), LIMIT)

    def cancel_all(self) -> None:
        for o in self._live():
            o.cancel_requested = True
        self.retry_cancels()

    def retry_cancels(self) -> None:
        """Send cancels for orders whose previous request was still in flight."""
        orders = self._hbt.orders(ASSET)  # type: ignore[attr-defined]
        for oid, tracked in self._orders.items():
            if not tracked.cancel_requested or tracked.done:
                continue
            order = orders.get(oid)
            if order is not None and order.cancellable:
                self._hbt.cancel(ASSET, oid, False)  # type: ignore[attr-defined]

    def collect_fills(self) -> list[FillView]:
        """Turn executed quantity changes since the last call into fills."""
        orders = self._hbt.orders(ASSET)  # type: ignore[attr-defined]
        new: list[FillView] = []
        for oid, tracked in self._orders.items():
            if tracked.done:
                continue
            order = orders.get(oid)
            if order is None:
                continue
            status = int(order.status)
            if status in _FILLABLE:
                total = order.qty - order.leaves_qty
                delta = self._spec.round_qty(total - tracked.filled)
                if delta > 0:
                    maker = bool(order.arr[0]["maker"])
                    price = float(order.exec_price)
                    if not maker:
                        price *= 1.0 + tracked.side * self._impact(delta) / BPS
                    fill = FillView(
                        ts=int(order.exch_timestamp),
                        side=tracked.side,
                        qty=delta,
                        price=price,
                        liquidity=Liquidity.MAKER if maker else Liquidity.TAKER,
                        order_ref=f"hbt-{oid}",
                    )
                    tracked.filled += delta
                    new.append(fill)
                    self.fills.append(
                        {
                            "ts": fill.ts,
                            "side": int(fill.side),
                            "qty": delta,
                            "price": price,
                            "fee": self._costs.fee(delta * price, fill.liquidity),
                            "liquidity": fill.liquidity.value,
                            "order_ref": fill.order_ref,
                        }
                    )
            if status in _TERMINAL or (status == NEW and tracked.filled >= tracked.qty):
                tracked.done = True
        return new


class HftBacktestEngine:
    name = "hftbacktest"

    def __init__(
        self,
        cfg: HftBacktestSettings,
        seconds_per_day: float,
        latency_quantile: float,
        fidelity: Fidelity = Fidelity.TICK,
    ) -> None:
        self._cfg = cfg
        self._seconds_per_day = seconds_per_day
        self._latency_quantile = latency_quantile
        self.fidelity = fidelity

    def _latency(self, bundle: FillModelBundle | None) -> tuple[int, int]:
        if bundle is None:
            return self._cfg.entry_latency_ns, self._cfg.response_latency_ns
        lat = bundle.latency.quantile(self._latency_quantile)
        return lat, lat

    def _asset(self, job: EngineJob, events: NDArray[np.void], latency: tuple[int, int]) -> BacktestAsset:
        cfg = self._cfg
        asset = (
            BacktestAsset()
            .data([events])
            .linear_asset(cfg.contract_size)
            .constant_order_latency(latency[0], latency[1])
            .trading_value_fee_model(job.costs.maker_fee, job.costs.taker_fee)
            .tick_size(job.instrument.tick_size)
            .lot_size(job.instrument.lot_size)
        )
        asset = (
            asset.power_prob_queue_model(cfg.queue_power)
            if cfg.queue_model == "power_prob"
            else asset.risk_adverse_queue_model()
        )
        return (
            asset.no_partial_fill_exchange()
            if cfg.exchange == "no_partial_fill"
            else asset.partial_fill_exchange()
        )

    def run(self, job: EngineJob) -> EngineResult:
        if job.tape is None:
            raise ValueError("the tick engine needs a tick tape")
        started = time.perf_counter()
        events = tape_to_events(job.tape)
        latency = self._latency(job.fill_models)
        hbt = HashMapMarketDepthBacktest([self._asset(job, events, latency)])
        impact = self._impact_fn(job)
        ctx = _HbtContext(hbt, job.instrument, job.costs, impact)
        handler = job.strategy.event_handler(job.bars, job.params, job.instrument)
        try:
            self._replay(hbt, ctx, handler, job, int(events["exch_ts"].min()))
            state = hbt.state_values(ASSET)
            mark = float(job.bars["close"].iloc[-1])
            native_pnl = float(state.balance - state.fee + state.position * mark)
            native_fees = float(state.fee)
        finally:
            hbt.close()
        fills = make_fills(ctx.fills)
        ledger = mark_to_market(fills, job.bars, job.bar_interval_ns, job.initial_cash)
        meta: dict[str, object] = {
            "entry_latency_ns": latency[0],
            "response_latency_ns": latency[1],
            "queue_model": self._cfg.queue_model,
            "fill_model": (
                f"calibrated-v{job.fill_models.version}" if job.fill_models is not None else "uncalibrated"
            ),
            "n_events": len(events),
        }
        return EngineResult(
            engine=self.name,
            fidelity=self.fidelity,
            symbol=job.instrument.symbol,
            fills=fills,
            ledger=ledger,
            native={"pnl": native_pnl, "fees": native_fees, "n_fills": float(len(fills))},
            wall_time_s=time.perf_counter() - started,
            initial_cash=job.initial_cash,
            meta=meta,
        )

    def _impact_fn(self, job: EngineJob) -> Callable[[float], float]:
        bundle = job.fill_models
        if bundle is None:
            return lambda qty: 0.0
        adv = average_daily_volume(job.bars, job.bar_interval_ns, self._seconds_per_day)
        vol = bar_volatility_bps_sqrt_s(job.bars, job.bar_interval_ns)

        def impact(qty: float) -> float:
            return max(bundle.slippage.impact_bps(qty / adv, vol), 0.0)

        return impact

    def _replay(
        self, hbt: object, ctx: _HbtContext, handler: EventHandler, job: EngineJob, first_ts: int
    ) -> None:
        ts_open = job.bars["ts"].to_numpy(dtype="int64")
        cols = [job.bars[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close", "volume")]
        for i, ts in enumerate(ts_open):
            ts_close = int(ts) + job.bar_interval_ns
            now = int(hbt.current_timestamp)  # type: ignore[attr-defined]
            gap = ts_close - (first_ts if now == NOT_STARTED else now)
            if gap > 0 and hbt.elapse(gap) == END_OF_DATA:  # type: ignore[attr-defined]
                break
            for fill in ctx.collect_fills():
                handler.on_fill(ctx, fill)
            ctx.retry_cancels()
            bar = BarView(
                index=i,
                ts=int(ts),
                ts_close=ts_close,
                open=float(cols[0][i]),
                high=float(cols[1][i]),
                low=float(cols[2][i]),
                close=float(cols[3][i]),
                volume=float(cols[4][i]),
            )
            handler.on_bar(ctx, bar)

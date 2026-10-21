"""Event-driven rung on NautilusTrader's ``BacktestEngine`` (bars, or trade ticks aggregated to bars).

Calibrated fills are injected through :class:`CalibratedFillModel`, a subclass of Nautilus'
``FillModel``: limit orders resting at the touch fill with the calibrated probability, and market
orders walk a simulated book whose liquidity sits the calibrated slippage away from the quote.
"""

from __future__ import annotations

import math
import time
import warnings
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import Bar, BarSpecification, BarType, BookOrder, TradeTick
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    AggressorSide,
    BarAggregation,
    BookType,
    CurrencyType,
    LiquiditySide,
    OmsType,
    OrderSide,
    OrderType,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TradeId, TraderId, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy as NautilusStrategy

from bt_orchestrator.calibration.models import FillModelBundle
from bt_orchestrator.data.tape import event_time
from bt_orchestrator.domain import Fidelity, InstrumentSpec, Liquidity, make_fills
from bt_orchestrator.engines.base import EngineJob, EngineResult, mark_to_market
from bt_orchestrator.settings import NautilusSettings
from bt_orchestrator.strategies.base import BarView, EventHandler, FillView, Side
from bt_orchestrator.timeutil import NANOS_PER_SECOND

BPS = 1e4
# Depth of the simulated book used for slipped market orders; large enough to absorb any order.
UNLIMITED_SIZE = 1_000_000


def _currency(code: str) -> Currency:
    try:
        return Currency.from_str(code)
    except (ValueError, KeyError):
        return Currency(code, 8, 0, code, CurrencyType.CRYPTO)


def build_instrument(spec: InstrumentSpec, maker_fee: float, taker_fee: float) -> CurrencyPair:
    venue = Venue(spec.venue)
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(spec.symbol), venue),
        raw_symbol=Symbol(spec.symbol),
        base_currency=_currency(spec.base),
        quote_currency=_currency(spec.quote),
        price_precision=spec.price_precision,
        size_precision=spec.size_precision,
        price_increment=Price(spec.tick_size, spec.price_precision),
        size_increment=Quantity(spec.lot_size, spec.size_precision),
        lot_size=None,
        max_quantity=None,
        min_quantity=None,
        max_notional=None,
        min_notional=None,
        max_price=None,
        min_price=None,
        margin_init=Decimal(0),
        margin_maint=Decimal(0),
        maker_fee=Decimal(str(maker_fee)),
        taker_fee=Decimal(str(taker_fee)),
        ts_event=0,
        ts_init=0,
    )


def bar_volatility_bps_sqrt_s(bars: pd.DataFrame, bar_interval_ns: int) -> float:
    """Realised volatility of bar closes in bps per sqrt-second (the calibration feature's units)."""
    log_ret = np.diff(np.log(bars["close"].to_numpy(dtype="float64")))
    if len(log_ret) < 2:
        return 0.0
    return float(np.std(log_ret, ddof=1) * BPS / math.sqrt(bar_interval_ns / NANOS_PER_SECOND))


def average_daily_volume(bars: pd.DataFrame, bar_interval_ns: int, seconds_per_day: float) -> float:
    bars_per_day = seconds_per_day * NANOS_PER_SECOND / bar_interval_ns
    return float(bars["volume"].mean() * bars_per_day)


def _bar_step(interval_ns: int) -> tuple[int, Any]:
    """Express a bar interval in the largest whole unit (Nautilus rejects e.g. 60 SECOND)."""
    seconds = interval_ns // NANOS_PER_SECOND
    for unit_s, aggregation in ((3600, BarAggregation.HOUR), (60, BarAggregation.MINUTE)):
        if seconds % unit_s == 0:
            return seconds // unit_s, aggregation
    return seconds, BarAggregation.SECOND


class CalibratedFillModel(FillModel):  # type: ignore[misc]  # Cython base class is untyped
    """Nautilus fill model driven by a calibrated :class:`FillModelBundle`."""

    def __init__(
        self,
        bundle: FillModelBundle,
        *,
        vol_bps_sqrt_s: float,
        adv: float,
        random_seed: int,
    ) -> None:
        prob = bundle.fill_prob.predict(0.0, bundle.reference_queue_ahead, vol_bps_sqrt_s)
        super().__init__(prob_fill_on_limit=prob, prob_slippage=0.0, random_seed=random_seed)
        self._bundle = bundle
        self._vol = vol_bps_sqrt_s
        self._adv = adv
        self.prob_fill_at_touch = prob

    def slippage_bps(self, qty: float) -> float:
        b = self._bundle
        bps = b.slippage.predict_bps(b.reference_half_spread_bps, qty / self._adv, self._vol)
        return max(bps, 0.0)

    def get_orderbook_for_fill_simulation(
        self, instrument: Any, order: Any, best_bid: Any, best_ask: Any
    ) -> OrderBook | None:
        if order.order_type != OrderType.MARKET:
            return None
        tick = float(instrument.price_increment)
        mid = (float(best_bid) + float(best_ask)) / 2.0
        ticks = round(self.slippage_bps(float(order.quantity)) / BPS * mid / tick)
        book = OrderBook(instrument_id=instrument.id, book_type=BookType.L2_MBP)
        prec = instrument.price_precision
        size = Quantity(UNLIMITED_SIZE, instrument.size_precision)
        book.add(BookOrder(OrderSide.BUY, Price(float(best_bid) - ticks * tick, prec), size, 1), 0, 0)
        book.add(BookOrder(OrderSide.SELL, Price(float(best_ask) + ticks * tick, prec), size, 2), 0, 0)
        return book


class _AdapterConfig(StrategyConfig, frozen=True):
    pass


class _Context:
    """:class:`StrategyContext` over a running Nautilus strategy."""

    def __init__(self, owner: _HandlerStrategy, spec: InstrumentSpec) -> None:
        self._owner = owner
        self._spec = spec
        self._cancel_requested: set[str] = set()

    @property
    def instrument(self) -> InstrumentSpec:
        return self._spec

    @property
    def now_ns(self) -> int:
        return int(self._owner.clock.timestamp_ns())

    @property
    def position(self) -> float:
        return float(self._owner.portfolio.net_position(self._owner.instrument_id))

    def _open_orders(self) -> list[Any]:
        orders = self._owner.cache.orders_open(instrument_id=self._owner.instrument_id)
        return [o for o in orders if o.client_order_id.value not in self._cancel_requested]

    @property
    def working_qty(self) -> float:
        return float(
            sum((1 if o.side == OrderSide.BUY else -1) * float(o.leaves_qty) for o in self._open_orders())
        )

    def open_order_count(self) -> int:
        return len(self._open_orders())

    def _side(self, side: Side) -> OrderSide:
        return OrderSide.BUY if side is Side.BUY else OrderSide.SELL

    def submit_market(self, side: Side, qty: float) -> str:
        order = self._owner.order_factory.market(
            self._owner.instrument_id,
            self._side(side),
            Quantity(self._spec.round_qty(qty), self._spec.size_precision),
        )
        self._owner.submit_order(order)
        return str(order.client_order_id.value)

    def submit_limit(self, side: Side, qty: float, price: float) -> str:
        order = self._owner.order_factory.limit(
            self._owner.instrument_id,
            self._side(side),
            Quantity(self._spec.round_qty(qty), self._spec.size_precision),
            Price(self._spec.round_price(price), self._spec.price_precision),
            time_in_force=TimeInForce.GTC,
        )
        self._owner.submit_order(order)
        return str(order.client_order_id.value)

    def cancel_all(self) -> None:
        for order in self._open_orders():
            self._cancel_requested.add(order.client_order_id.value)
        self._owner.cancel_all_orders(self._owner.instrument_id)


class _HandlerStrategy(NautilusStrategy):  # type: ignore[misc]
    """Nautilus strategy that forwards bars and fills to an engine-agnostic :class:`EventHandler`."""

    def __init__(
        self,
        handler: EventHandler,
        spec: InstrumentSpec,
        instrument_id: InstrumentId,
        bar_type: BarType,
        bar_interval_ns: int,
        first_bar_ts: int,
    ) -> None:
        super().__init__(_AdapterConfig())
        self.handler = handler
        self.instrument_id = instrument_id
        self.bar_type = bar_type
        self.ctx = _Context(self, spec)
        self.fills: list[dict[str, Any]] = []
        self._interval = bar_interval_ns
        self._first = first_bar_ts

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        ts_close = int(bar.ts_event)
        ts_open = ts_close - self._interval
        view = BarView(
            index=(ts_open - self._first) // self._interval,
            ts=ts_open,
            ts_close=ts_close,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        )
        self.handler.on_bar(self.ctx, view)

    def on_order_filled(self, event: Any) -> None:
        side = Side.BUY if event.order_side == OrderSide.BUY else Side.SELL
        liquidity = Liquidity.MAKER if event.liquidity_side == LiquiditySide.MAKER else Liquidity.TAKER
        fill = FillView(
            ts=int(event.ts_event),
            side=side,
            qty=float(event.last_qty),
            price=float(event.last_px),
            liquidity=liquidity,
            order_ref=str(event.client_order_id.value),
        )
        self.fills.append(
            {
                "ts": fill.ts,
                "side": int(side),
                "qty": fill.qty,
                "price": fill.price,
                "fee": float(event.commission),
                "liquidity": liquidity.value,
                "order_ref": fill.order_ref,
            }
        )
        self.handler.on_fill(self.ctx, fill)


class NautilusEngine:
    name = "nautilus"
    fidelity = Fidelity.EVENT

    def __init__(
        self,
        cfg: NautilusSettings,
        leverage: float,
        seconds_per_day: float,
    ) -> None:
        self._cfg = cfg
        self._leverage = leverage
        self._seconds_per_day = seconds_per_day

    def _fill_model(self, job: EngineJob) -> tuple[FillModel, dict[str, Any]]:
        if job.fill_models is None:
            return FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=job.seed), {
                "fill_model": "touch_fills"
            }
        model = CalibratedFillModel(
            job.fill_models,
            vol_bps_sqrt_s=bar_volatility_bps_sqrt_s(job.bars, job.bar_interval_ns),
            adv=average_daily_volume(job.bars, job.bar_interval_ns, self._seconds_per_day),
            random_seed=job.seed,
        )
        return model, {
            "fill_model": f"calibrated-v{job.fill_models.version}",
            "prob_fill_at_touch": model.prob_fill_at_touch,
        }

    def _bar_type(self, instrument_id: InstrumentId, interval_ns: int, internal: bool) -> BarType:
        spec = BarSpecification(*_bar_step(interval_ns), PriceType.LAST)
        source = AggregationSource.INTERNAL if internal else AggregationSource.EXTERNAL
        return BarType(instrument_id, spec, source)

    def _bars(self, job: EngineJob, bar_type: BarType) -> list[Bar]:
        spec = job.instrument
        out = []
        for row in job.bars.itertuples(index=False):
            ts_close = int(row.ts) + job.bar_interval_ns
            out.append(
                Bar(
                    bar_type,
                    Price(spec.round_price(row.open), spec.price_precision),
                    Price(spec.round_price(row.high), spec.price_precision),
                    Price(spec.round_price(row.low), spec.price_precision),
                    Price(spec.round_price(row.close), spec.price_precision),
                    Quantity(max(row.volume, 0.0), spec.size_precision),
                    ts_close,
                    ts_close,
                )
            )
        return out

    def _trades(self, job: EngineJob, instrument_id: InstrumentId) -> list[TradeTick]:
        if job.tape is None:
            raise ValueError("nautilus trades mode needs a tick tape")
        spec = job.instrument
        trades = job.tape.trades
        ts = event_time(trades)
        start = int(job.bars["ts"].iloc[0])
        end = int(job.bars["ts"].iloc[-1]) + job.bar_interval_ns
        keep = (ts >= start) & (ts < end)
        ticks = []
        for t, row in zip(ts[keep], trades.loc[keep].itertuples(index=False), strict=True):
            ticks.append(
                TradeTick(
                    instrument_id,
                    Price(spec.round_price(row.price), spec.price_precision),
                    Quantity(spec.round_qty(row.size), spec.size_precision),
                    AggressorSide.BUYER if row.side == "buy" else AggressorSide.SELLER,
                    TradeId(str(row.trade_id)),
                    int(t),
                    int(t),
                )
            )
        return ticks

    def run(self, job: EngineJob) -> EngineResult:
        started = time.perf_counter()
        spec = job.instrument
        instrument = build_instrument(spec, job.costs.maker_fee, job.costs.taker_fee)
        venue = instrument.id.venue
        quote = instrument.quote_currency
        trades_mode = self._cfg.data_mode == "trades"
        bar_type = self._bar_type(instrument.id, job.bar_interval_ns, internal=trades_mode)
        fill_model, fill_meta = self._fill_model(job)

        engine = BacktestEngine(
            BacktestEngineConfig(
                trader_id=TraderId(self._cfg.trader_id),
                logging=LoggingConfig(log_level=self._cfg.log_level, bypass_logging=True),
            )
        )
        try:
            engine.add_venue(
                venue,
                OmsType.NETTING,
                AccountType.MARGIN,
                [Money(job.initial_cash, quote)],
                base_currency=quote,
                default_leverage=Decimal(str(self._leverage)),
                fill_model=fill_model,
                fee_model=MakerTakerFeeModel(),
            )
            engine.add_instrument(instrument)
            engine.add_data(self._trades(job, instrument.id) if trades_mode else self._bars(job, bar_type))
            handler = job.strategy.event_handler(job.bars, job.params, spec)
            strategy = _HandlerStrategy(
                handler,
                spec,
                instrument.id,
                bar_type,
                job.bar_interval_ns,
                int(job.bars["ts"].iloc[0]),
            )
            engine.add_strategy(strategy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engine.run()
            account = engine.trader.generate_account_report(venue)
            balance = float(account["total"].iloc[-1]) if not account.empty else job.initial_cash
            unrealized = engine.portfolio.unrealized_pnl(instrument.id)
            native_pnl = balance - job.initial_cash + (float(unrealized) if unrealized else 0.0)
            fills = make_fills(strategy.fills)
        finally:
            engine.dispose()

        ledger = mark_to_market(fills, job.bars, job.bar_interval_ns, job.initial_cash)
        return EngineResult(
            engine=self.name,
            fidelity=self.fidelity,
            symbol=spec.symbol,
            fills=fills,
            ledger=ledger,
            native={
                "pnl": native_pnl,
                "fees": float(fills["fee"].sum()),
                "n_fills": float(len(fills)),
            },
            wall_time_s=time.perf_counter() - started,
            initial_cash=job.initial_cash,
            meta={"data_mode": self._cfg.data_mode} | fill_meta,
        )

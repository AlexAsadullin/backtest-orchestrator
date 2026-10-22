"""The orchestrator: load point-in-time data, run an engine per symbol, log every run to S6.

Every public entry point (single runs, the fidelity ladder, parity, sweeps) goes through
:meth:`Orchestrator.run_level`, so no run escapes trial logging.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from bt_orchestrator.analytics.metrics import compute_metrics
from bt_orchestrator.calibration.fit import calibrate, load_calibration_input
from bt_orchestrator.calibration.models import FillModelBundle
from bt_orchestrator.calibration.store import FillModelStore
from bt_orchestrator.clients.ledger import LedgerClient
from bt_orchestrator.contracts.ledger import Actor, TrialRecord
from bt_orchestrator.data.market import MarketData
from bt_orchestrator.data.source import LeakageError
from bt_orchestrator.data.tape import bars_from_trades
from bt_orchestrator.domain import CostSpec, DataBundle, Fidelity, InstrumentSpec
from bt_orchestrator.engines.base import EngineJob, EngineResult
from bt_orchestrator.engines.factory import EngineFactory
from bt_orchestrator.engines.nautilus_engine import average_daily_volume
from bt_orchestrator.settings import Settings
from bt_orchestrator.strategies.base import Strategy, StrategyParams
from bt_orchestrator.strategies.registry import StrategyRegistry
from bt_orchestrator.timeutil import ns_to_iso, to_ns, utc_now
from bt_orchestrator.trials import TrialBuilder, TrialInputs

log = structlog.get_logger(__name__)
TICK_LEVELS = frozenset({Fidelity.TICK})


def instrument_spec(settings: Settings, symbol: str) -> InstrumentSpec:
    i = settings.instrument(symbol)
    return InstrumentSpec(
        symbol=symbol,
        venue=settings.market.venue,
        base=i.base,
        quote=i.quote,
        tick_size=i.tick_size,
        lot_size=i.lot_size,
        price_precision=i.price_precision,
        size_precision=i.size_precision,
    )


class LookAheadError(ValueError):
    """A vectorised strategy's targets change when future bars are removed."""


@dataclass(frozen=True)
class RunSpec:
    """What to run: the backtest request in typed form."""

    strategy: str
    params: dict[str, Any]
    universe: list[str]
    start: str
    end: str
    known_at: str
    seed: int
    actor: Actor | None = None
    tags: tuple[str, ...] = ()


@dataclass
class LevelRun:
    """One fidelity level over the whole universe."""

    fidelity: Fidelity
    engine: str
    results: list[EngineResult]
    returns: pd.Series
    metrics: dict[str, float | int | None]
    trial: TrialRecord
    fingerprint: str | None
    fill_models: dict[str, int | None] = field(default_factory=dict)

    @property
    def wall_time_s(self) -> float:
        return float(sum(r.wall_time_s for r in self.results))


def check_causality(
    strategy: Strategy, bars: pd.DataFrame, params: StrategyParams, inst: InstrumentSpec, cuts: int
) -> None:
    """Prefix invariance: targets computed on ``bars[:k]`` must equal the full-sample targets up to k."""
    if not strategy.supports_vectorised or cuts <= 0:
        return
    full = strategy.vectorised(bars, params, inst).target
    for k in np.linspace(len(bars) // 4, len(bars), cuts + 1, dtype=int)[:-1]:
        if k < 2:
            continue
        part = strategy.vectorised(bars.iloc[:k].reset_index(drop=True), params, inst).target
        if not np.array_equal(part, full[:k], equal_nan=True):
            raise LookAheadError(
                f"strategy {strategy.name!r} uses future bars: targets differ when cut at bar {k}"
            )


def combine_results(results: list[EngineResult], initial_cash: float) -> pd.DataFrame:
    """Equal-weight portfolio ledger across symbols, aligned on bar close times."""
    if len(results) == 1:
        return results[0].ledger
    rets = pd.concat([r.returns.rename(r.symbol) for r in results], axis=1, join="inner")
    ret = rets.mean(axis=1)
    pos = pd.concat(
        [r.ledger.set_index("ts_close")["position"].abs().rename(r.symbol) for r in results],
        axis=1,
        join="inner",
    ).sum(axis=1)
    equity = initial_cash * np.cumprod(1.0 + ret.to_numpy())
    return pd.DataFrame(
        {
            "ts_close": ret.index.to_numpy(dtype="int64"),
            "ret": ret.to_numpy(),
            "equity": equity,
            "position": pos.to_numpy(),
        }
    )


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        market: MarketData,
        ledger: LedgerClient,
        engines: EngineFactory,
        models: FillModelStore,
        registry: StrategyRegistry,
    ) -> None:
        self.settings = settings
        self.market = market
        self.ledger = ledger
        self.engines = engines
        self.models = models
        self.registry = registry
        self._trials = TrialBuilder(settings.ledger, settings.trials_dir, settings.code_version)

    # -- building blocks -------------------------------------------------------------------------

    def instrument(self, symbol: str) -> InstrumentSpec:
        return instrument_spec(self.settings, symbol)

    def costs(self) -> CostSpec:
        return CostSpec(self.settings.costs.maker_fee, self.settings.costs.taker_fee)

    def params(self, strategy: Strategy, raw: dict[str, Any]) -> StrategyParams:
        defaults = self.settings.strategy_settings(strategy.name).defaults
        return strategy.parse_params(raw, defaults)

    def load(self, spec: RunSpec, symbol: str, fidelity: Fidelity) -> DataBundle:
        try:
            bundle = self.market.load(
                symbol,
                spec.start,
                spec.end,
                spec.known_at,
                with_tape=fidelity in TICK_LEVELS,
                strategy_id=self._trials.strategy_id(spec.strategy, spec.universe),
            )
        except LeakageError as exc:
            if not exc.reported:
                self.ledger.report_leakage(exc.event)
            raise
        if bundle.tape is not None:
            # The tick engine sees bars built from the very tape it replays.
            bundle.bars = bars_from_trades(
                bundle.tape.trades,
                to_ns(spec.start),
                to_ns(spec.end),
                self.settings.bar_interval_ns,
            )
        return bundle

    def fill_models_for(self, symbol: str, fidelity: Fidelity) -> FillModelBundle | None:
        if fidelity is Fidelity.VECTORISED:
            return None
        return self.models.load(self.settings.market.venue, symbol)

    # -- single level ----------------------------------------------------------------------------

    def run_level(
        self,
        spec: RunSpec,
        fidelity: Fidelity,
        *,
        calibrated: bool = True,
        log_trial: bool = True,
        bundles: dict[str, DataBundle] | None = None,
    ) -> LevelRun:
        """Run ``spec`` at one fidelity on every symbol and log one trial for the combined result."""
        s = self.settings
        strategy = self.registry.get(spec.strategy)
        params = self.params(strategy, spec.params)
        if fidelity is Fidelity.VECTORISED and not strategy.supports_vectorised:
            raise ValueError(f"strategy {strategy.name!r} has no vectorised form")
        engine = self.engines.build(fidelity)
        started = utc_now()
        results: list[EngineResult] = []
        primary: str | None = None
        fingerprint_tags: list[str] = []
        flag_tags: list[str] = []
        versions: dict[str, int | None] = {}
        for symbol in spec.universe:
            bundle = (bundles or {}).get(symbol) or self.load(spec, symbol, fidelity)
            inst = self.instrument(symbol)
            if fidelity is Fidelity.VECTORISED:
                check_causality(strategy, bundle.bars, params, inst, s.ladder.causality_cuts)
            models = self.fill_models_for(symbol, fidelity) if calibrated else None
            versions[symbol] = models.version if models else None
            job = EngineJob(
                strategy=strategy,
                params=params,
                instrument=inst,
                bars=bundle.bars,
                bar_interval_ns=s.bar_interval_ns,
                costs=self.costs(),
                initial_cash=s.account.initial_cash,
                seed=spec.seed,
                tape=bundle.tape,
                fill_models=models,
            )
            result = engine.run(job)
            results.append(result)
            primary = primary or bundle.primary_fingerprint
            fingerprint_tags += [f"fingerprint:{symbol}:{k}={v}" for k, v in bundle.fingerprints.items()]
            flag_tags += [f"leakage-flag:{symbol}:{f}" for f in bundle.leakage_flags]
            log.info(
                "engine_run",
                engine=result.engine,
                fidelity=fidelity.value,
                symbol=symbol,
                fills=len(result.fills),
                pnl=round(result.final_pnl, 2),
                wall_s=round(result.wall_time_s, 3),
            )
        ledger = combine_results(results, s.account.initial_cash)
        fills = pd.concat([r.fills for r in results], ignore_index=True)
        metrics = compute_metrics(ledger, fills, s.periods_per_year, s.account.initial_cash)
        returns = pd.Series(ledger["ret"].to_numpy(), index=ledger["ts_close"].to_numpy())
        # The trial keeps I2's own fingerprint of the primary dataset (never a derived hash), so
        # GET /v1/fingerprints/{fp} on I2 reproduces it; every dataset's fingerprint is also tagged.
        fingerprint = primary
        trial = self._trials.build(
            TrialInputs(
                strategy=spec.strategy,
                universe=spec.universe,
                params=params.model_dump(),
                returns=returns,
                metrics=metrics,
                dataset_fingerprint=fingerprint,
                seed=spec.seed,
                started_at=started,
                finished_at=utc_now(),
                periods_per_year=s.periods_per_year,
                tags=[
                    f"fidelity:{fidelity.value}",
                    f"engine:{engine.name}",
                    "calibrated" if any(versions.values()) else "uncalibrated",
                    *spec.tags,
                    *fingerprint_tags,
                    *flag_tags,
                ],
                actor=spec.actor,
            )
        )
        if log_trial:
            self.ledger.log_trials([trial])
        return LevelRun(
            fidelity=fidelity,
            engine=engine.name,
            results=results,
            returns=returns,
            metrics=metrics,
            trial=trial,
            fingerprint=fingerprint,
            fill_models=versions,
        )

    # -- calibration -----------------------------------------------------------------------------

    def calibrate(
        self,
        symbol: str,
        *,
        known_at: str | None = None,
        fills_root: Path | None = None,
        orders_root: Path | None = None,
        with_orders: bool = True,
    ) -> FillModelBundle:
        """Fit and store a new fill-model version from the I6 fills export for ``symbol``.

        ADV comes from the bars over the fills' sample window, read through the point-in-time source.
        A new version is written only when the input data changed since the latest one.
        """
        s = self.settings
        venue = s.market.venue
        orders = (orders_root or s.orders_root) if with_orders else None
        data = load_calibration_input(fills_root or s.fills_root, orders, venue, symbol)
        latest = self.models.load(venue, symbol)
        if latest is not None and latest.data_fingerprint == data.fingerprint:
            log.info("calibration_unchanged", version=latest.version, symbol=symbol)
            return latest
        start_ns = int(data.fills["ts_submit"].min())
        end_ns = int(data.fills["ts_submit"].max()) + s.bar_interval_ns
        start = ns_to_iso(start_ns // s.bar_interval_ns * s.bar_interval_ns)
        end = ns_to_iso(end_ns)
        bars = self.market.load(
            symbol,
            start,
            end,
            known_at or end,
            with_tape=False,
            strategy_id=self._trials.strategy_id("calibration", [symbol]),
        ).bars
        adv = average_daily_volume(bars, s.bar_interval_ns, s.market.seconds_per_day)
        bundle = calibrate(
            data,
            venue=venue,
            symbol=symbol,
            adv=adv,
            cfg=s.calibration,
            version=self.models.next_version(venue, symbol),
            fitted_at=utc_now(),
        )
        path = self.models.save(bundle)
        log.info("calibrated", symbol=symbol, version=bundle.version, path=str(path), **bundle.evaluation)
        return bundle


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]

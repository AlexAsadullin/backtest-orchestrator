"""Builds an :class:`Orchestrator` and its clients from settings (the composition root)."""

from __future__ import annotations

import logging

import structlog

from bt_orchestrator.calibration.store import FillModelStore
from bt_orchestrator.clients.ledger import LedgerClient
from bt_orchestrator.clients.sim import FakeSimService, SimClient, SimHttpClient
from bt_orchestrator.data.market import MarketData
from bt_orchestrator.data.source import DataSource, LocalParquetSource, PitHttpSource
from bt_orchestrator.data.synthetic import MarketSpec
from bt_orchestrator.engines.factory import EngineFactory
from bt_orchestrator.orchestrator import Orchestrator, instrument_spec
from bt_orchestrator.settings import Settings
from bt_orchestrator.strategies.registry import default_registry


def configure_logging(settings: Settings) -> None:
    level = logging.getLevelName(settings.logging.level.upper())
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.logging.json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=False,
    )


def build_source(settings: Settings) -> DataSource:
    if settings.pit.mode == "http":
        return PitHttpSource(
            settings.pit.base_url,
            settings.pit.timeout_s,
            fmt=settings.pit.format,
            timestamp_columns=tuple(settings.pit.timestamp_columns),
        )
    return LocalParquetSource(settings.project_dir, settings.local_source)


def build_sim(settings: Settings) -> SimClient:
    if settings.sim.mode == "http":
        return SimHttpClient(settings.sim.base_url, settings.sim.timeout_s)
    base = MarketSpec.from_settings(
        settings.synthetic,
        settings.market.default_universe[0],
        settings.market.venue,
        settings.bar_interval_ns,
    )
    instruments = {sym: instrument_spec(settings, sym) for sym in settings.market.instruments}
    return FakeSimService(settings.sim_fake_dir, base, instruments, settings.sim.fake_scenarios)


def build_orchestrator(settings: Settings) -> Orchestrator:
    ledger = LedgerClient(
        mode=settings.ledger.mode,
        base_url=settings.ledger.base_url,
        timeout_s=settings.ledger.timeout_s,
        inbox_dir=settings.ledger_inbox_dir,
        source=settings.ledger.source,
    )
    return Orchestrator(
        settings=settings,
        market=MarketData(
            build_source(settings),
            settings.pit,
            venue=settings.market.venue,
            bar_interval_ns=settings.bar_interval_ns,
        ),
        ledger=ledger,
        engines=EngineFactory(settings, build_sim(settings)),
        models=FillModelStore(settings.models_dir),
        registry=default_registry(),
    )

"""Builds the engine that serves each fidelity level from settings."""

from __future__ import annotations

from bt_orchestrator.clients.sim import SimClient
from bt_orchestrator.domain import Fidelity
from bt_orchestrator.engines.base import Engine
from bt_orchestrator.engines.hftbacktest_engine import HftBacktestEngine
from bt_orchestrator.engines.nautilus_engine import NautilusEngine
from bt_orchestrator.engines.reactive_sim import ReactiveSimEngine
from bt_orchestrator.engines.vectorbt_engine import VectorbtEngine
from bt_orchestrator.settings import Settings


class EngineFactory:
    def __init__(self, settings: Settings, sim_client: SimClient) -> None:
        self._s = settings
        self._sim = sim_client

    def tick(self) -> HftBacktestEngine:
        s = self._s
        return HftBacktestEngine(
            s.engines.hftbacktest, s.market.seconds_per_day, s.calibration.latency_quantile
        )

    def build(self, fidelity: Fidelity) -> Engine:
        s = self._s
        if fidelity is Fidelity.VECTORISED:
            return VectorbtEngine(s.engines.vectorbt)
        if fidelity is Fidelity.EVENT:
            return NautilusEngine(s.engines.nautilus, s.account.leverage, s.market.seconds_per_day)
        if fidelity is Fidelity.TICK:
            return self.tick()
        return ReactiveSimEngine(self._sim, self.tick(), s.sim.scenario, s.sim.max_duration_s)

"""Reactive-simulator rung: ask S3 for a scenario tape (contract 6.5), then replay it on hftbacktest."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from bt_orchestrator.clients.sim import SimClient
from bt_orchestrator.contracts.sim import SimulationRequest
from bt_orchestrator.data.tape import bars_from_trades, event_time, read_i1_tape
from bt_orchestrator.domain import Fidelity
from bt_orchestrator.engines.base import EngineJob, EngineResult
from bt_orchestrator.engines.hftbacktest_engine import HftBacktestEngine
from bt_orchestrator.timeutil import NANOS_PER_SECOND


class ReactiveSimEngine:
    name = "s3-sim+hftbacktest"
    fidelity = Fidelity.REACTIVE_SIM

    def __init__(
        self, client: SimClient, tick_engine: HftBacktestEngine, scenario: str, max_duration_s: int
    ) -> None:
        self._client = client
        self._tick = tick_engine
        self._scenario = scenario
        self._max_duration_s = max_duration_s

    def run(self, job: EngineJob) -> EngineResult:
        duration_s = min(len(job.bars) * job.bar_interval_ns // NANOS_PER_SECOND, self._max_duration_s)
        request = SimulationRequest(
            scenario=self._scenario,
            seed=job.seed,
            duration_s=int(duration_s),
            venue=job.instrument.venue,
            symbol=job.instrument.symbol,
        )
        response = self._client.simulate(request)
        tape = read_i1_tape(Path(response.tape_path), job.instrument.symbol)
        ts = event_time(tape.trades)
        start = int(ts.min()) // job.bar_interval_ns * job.bar_interval_ns
        end = int(ts.max()) // job.bar_interval_ns * job.bar_interval_ns + job.bar_interval_ns
        bars = bars_from_trades(tape.trades, start, end, job.bar_interval_ns)
        result = self._tick.run(replace(job, bars=bars, tape=tape))
        result.engine = self.name
        result.fidelity = self.fidelity
        result.meta |= {
            "simulation_id": response.simulation_id,
            "scenario": self._scenario,
            "tape_path": response.tape_path,
        }
        return result

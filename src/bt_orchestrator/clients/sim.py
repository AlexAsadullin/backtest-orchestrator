"""Contract 6.5 simulation service (S3): HTTP client plus an in-process stand-in for offline runs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

import httpx

from bt_orchestrator.contracts.sim import Scenario, SimulationRequest, SimulationResponse
from bt_orchestrator.data.synthetic import MarketSpec, generate_tape
from bt_orchestrator.data.tape import write_i1_tape
from bt_orchestrator.domain import InstrumentSpec
from bt_orchestrator.settings import ScenarioSettings


class SimClient(Protocol):
    def scenarios(self) -> list[str]: ...

    def simulate(self, request: SimulationRequest) -> SimulationResponse: ...


class SimHttpClient:
    def __init__(self, base_url: str, timeout_s: float, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def scenarios(self) -> list[str]:
        response = self._client.get("/v1/scenarios")
        response.raise_for_status()
        body = response.json()
        items = body.get("scenarios", body) if isinstance(body, dict) else body
        return [item if isinstance(item, str) else Scenario.model_validate(item).name for item in items]

    def simulate(self, request: SimulationRequest) -> SimulationResponse:
        response = self._client.post("/v1/simulations", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return SimulationResponse.model_validate(response.json())


class FakeSimService:
    """Writes a scenario-stressed synthetic tape in the I1 layout, as S3 would.

    It is not reactive: the tape does not respond to the strategy's orders. It exists so the reactive
    rung of the ladder runs end to end offline and in tests.
    """

    def __init__(
        self,
        out_dir: Path,
        base_spec: MarketSpec,
        instruments: dict[str, InstrumentSpec],
        scenarios: dict[str, ScenarioSettings],
    ) -> None:
        self._out = out_dir
        self._base = base_spec
        self._instruments = instruments
        self._scenarios = scenarios

    def scenarios(self) -> list[str]:
        return sorted(self._scenarios)

    def simulate(self, request: SimulationRequest) -> SimulationResponse:
        try:
            scenario = self._scenarios[request.scenario]
            inst = self._instruments[request.symbol]
        except KeyError as exc:
            raise ValueError(f"unknown scenario or symbol: {exc}") from exc
        n_bars = max(1, request.duration_s * 1_000_000_000 // self._base.bar_interval_ns)
        spec = self._base.with_scenario(scenario, request.seed, int(n_bars))
        spec = MarketSpec(**{**spec.__dict__, "symbol": request.symbol, "venue": request.venue})
        tape = generate_tape(spec, inst)
        simulation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, request.model_dump_json()))
        root = self._out / simulation_id
        if not root.exists():
            write_i1_tape(tape, root, request.venue)
        return SimulationResponse(simulation_id=simulation_id, tape_path=str(root))

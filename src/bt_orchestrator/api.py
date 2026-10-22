"""FastAPI service: contract 6.5 ``POST /v1/backtests`` plus strategy discovery."""

from __future__ import annotations

import math
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from bt_orchestrator.contracts.backtest import BacktestRequest, BacktestResponse
from bt_orchestrator.data.source import LeakageError
from bt_orchestrator.ladder import run_ladder
from bt_orchestrator.orchestrator import LookAheadError, Orchestrator, RunSpec, new_run_id
from bt_orchestrator.trials import finite_metrics

HTTP_CONFLICT = 409
HTTP_UNPROCESSABLE = 422


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats by ``None`` (strict JSON for clients such as I9)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def spec_from_request(req: BacktestRequest, default_seed: int) -> RunSpec:
    return RunSpec(
        strategy=req.strategy,
        params=req.params,
        universe=req.universe,
        start=req.start,
        end=req.end,
        known_at=req.known_at,
        seed=req.seed if req.seed is not None else default_seed,
        actor=req.actor,
        tags=("api",),
    )


def execute(orch: Orchestrator, req: BacktestRequest) -> BacktestResponse:
    """Run a request at one fidelity, or climb the ladder to it when ``ladder`` is set."""
    spec = spec_from_request(req, orch.settings.metrics.seed)
    if req.ladder:
        report, _ = run_ladder(orch, spec, top=req.fidelity)
        top = report.top_run()
        if top is None or top.trial_id is None:
            raise ValueError("no ladder level could run for this strategy")
        return BacktestResponse(
            run_id=report.run_id,
            trial_id=top.trial_id,
            metrics=finite_metrics(top.metrics),
            fidelity_report=json_safe(report.model_dump(mode="json")),
        )
    run = orch.run_level(spec, req.fidelity)
    return BacktestResponse(run_id=new_run_id(), trial_id=run.trial.trial_id, metrics=run.trial.metrics)


def create_app(orch: Orchestrator) -> FastAPI:
    app = FastAPI(title="Tradeq backtest orchestrator (I5)", version="0.1.0")
    # The engines keep process-global state (Nautilus logging, numba caches), so runs are serialised.
    lock = threading.Lock()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/strategies")
    def strategies() -> dict[str, Any]:
        return {
            "schema_version": "1",
            "strategies": [orch.registry.get(n).describe() for n in orch.registry.names()],
        }

    @app.post("/v1/backtests", response_model=BacktestResponse)
    def backtests(req: BacktestRequest) -> BacktestResponse | JSONResponse:
        try:
            with lock:
                return execute(orch, req)
        except LeakageError as exc:
            return JSONResponse(status_code=HTTP_CONFLICT, content=exc.event.model_dump(mode="json"))
        except (LookAheadError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=HTTP_UNPROCESSABLE, detail=str(exc)) from exc

    return app

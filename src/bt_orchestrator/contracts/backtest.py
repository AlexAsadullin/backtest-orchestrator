"""Contract 6.5 (Backtest service, I5): ``POST /v1/backtests``.

Optional extensions (documented in the README): ``ladder`` climbs every rung up to ``fidelity`` with
promotion gates and returns ``fidelity_report``; ``actor`` attributes the trial; ``seed`` fixes
stochastic fill models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bt_orchestrator.contracts.ledger import Actor
from bt_orchestrator.domain import Fidelity


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    universe: list[str] = Field(min_length=1)
    start: str
    end: str
    fidelity: Fidelity
    known_at: str
    ladder: bool = False
    actor: Actor | None = None
    seed: int | None = None


class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    run_id: str
    trial_id: str
    metrics: dict[str, float | int | None]
    fidelity_report: dict[str, Any] | None = None

"""Contract 6.1 (Research Ledger, S6): trial records and leakage events."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["1"] = "1"
TrialSource = Literal["i5", "s1", "s3", "s4", "s5", "s7", "s8", "i9", "notebook"]


class Actor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["human", "agent", "robot"]
    id: str


class ReturnSeries(BaseModel):
    """Per-period net returns with ISO-8601 timestamps."""

    model_config = ConfigDict(extra="forbid")

    ts: list[str]
    values: list[float]

    @model_validator(mode="after")
    def _same_length(self) -> ReturnSeries:
        if len(self.ts) != len(self.values):
            raise ValueError("returns.ts and returns.values must have the same length")
        return self


class TrialRecord(BaseModel):
    """One backtest trial as S6 counts it (deflated Sharpe needs every trial, not only winners)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    trial_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    family: str
    strategy_id: str
    source: TrialSource
    actor: Actor
    params: dict[str, Any]
    dataset_fingerprint: str | None
    code_version: str | None
    seed: int | None
    started_at: datetime
    finished_at: datetime
    periods_per_year: float
    returns: ReturnSeries | None = None
    returns_path: str | None = None
    metrics: dict[str, float | int | None]
    tags: list[str]

    @model_validator(mode="after")
    def _returns_present(self) -> TrialRecord:
        if (self.returns is None) == (self.returns_path is None):
            raise ValueError("exactly one of returns / returns_path must be set")
        return self


class TrialAck(BaseModel):
    model_config = ConfigDict(extra="allow")

    trial_id: str


class LeakageEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1"] = SCHEMA_VERSION
    source: Literal["i2", "s7"]
    strategy_id: str | None
    trial_id: str | None
    severity: Literal["info", "warn", "block"]
    kind: str
    detail: dict[str, Any]
    observed_at: datetime

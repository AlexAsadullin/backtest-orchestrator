"""Contract 6.5 (Simulation service, S3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    scenario: str
    seed: int
    duration_s: int
    venue: str
    symbol: str


class SimulationResponse(BaseModel):
    """``tape_path`` points at a directory written in the I1 tape layout (contract 6.2)."""

    model_config = ConfigDict(extra="allow")

    simulation_id: str
    tape_path: str


class Scenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str

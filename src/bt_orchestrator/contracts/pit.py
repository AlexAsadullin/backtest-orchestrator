"""Contract 6.3 (Point-in-time Store, I2): the "as known at T" query.

The shapes follow the shipped I2 service (``point-in-time-store/src/pit_store/models.py``): filters are a
list of typed predicates ``{column, op, value}``, and ``format``, ``strategy_id`` and ``trial_id`` are
I2's optional request fields. Responses carry ``X-Dataset-Fingerprint`` and, when I2's ``flag`` policy
clamped a query, ``X-Leakage-Flag`` (comma-separated leakage kinds).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

FINGERPRINT_HEADER = "X-Dataset-Fingerprint"
LEAKAGE_FLAG_HEADER = "X-Leakage-Flag"
ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"

FilterOp = Literal["==", "!=", "<", "<=", ">", ">=", "in"]
Scalar = str | int | float | bool


class PitFilter(BaseModel):
    """A typed predicate on one column; ``in`` takes a list, every other operator a scalar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    op: FilterOp
    value: Scalar | list[Scalar]

    @model_validator(mode="after")
    def _check(self) -> PitFilter:
        if (self.op == "in") != isinstance(self.value, list):
            raise ValueError("operator 'in' takes a list value; every other operator takes a scalar")
        return self


class PitQuery(BaseModel):
    """Body of ``POST /v1/query``. Times are ISO-8601 UTC strings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    dataset: str
    start: str
    end: str
    known_at: str
    columns: list[str] | None = None
    filters: list[PitFilter] | None = None
    mode: Literal["research", "live"] = "research"
    format: Literal["arrow", "json"] | None = None
    strategy_id: str | None = None
    trial_id: str | None = None

    def canonical(self) -> dict[str, Any]:
        """A deterministic dict used to fingerprint the query offline (output format excluded)."""
        return self.model_dump(mode="json", exclude_none=True, exclude={"format", "strategy_id", "trial_id"})

"""Contract 6.3 (Point-in-time Store, I2): the "as known at T" query."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FINGERPRINT_HEADER = "X-Dataset-Fingerprint"
ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"


class PitQuery(BaseModel):
    """Body of ``POST /v1/query``. Times are ISO-8601 UTC strings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    dataset: str
    start: str
    end: str
    known_at: str
    columns: list[str] | None = None
    filters: dict[str, Any] | None = None
    mode: Literal["research", "live"] = "research"

    def canonical(self) -> dict[str, Any]:
        """A deterministic dict used to fingerprint the query offline."""
        return self.model_dump(mode="json", exclude_none=True)

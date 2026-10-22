"""Point-in-time data sources (contract 6.3): the I2 HTTP client and an offline Parquet stand-in.

Both apply the same rules: rows are those with ``knowledge_time <= known_at``, the latest revision per
key wins, and a research-mode query that reaches past ``known_at`` (or touches a dataset without
knowledge times) is refused with a :class:`LeakageError`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import pandas as pd
import pyarrow as pa
import structlog

from bt_orchestrator.contracts.ledger import LeakageEvent
from bt_orchestrator.contracts.pit import FINGERPRINT_HEADER, PitQuery
from bt_orchestrator.settings import LocalSourceSettings
from bt_orchestrator.timeutil import to_ns, utc_now

log = structlog.get_logger(__name__)
HTTP_CONFLICT = 409


@dataclass(frozen=True)
class QueryResult:
    frame: pd.DataFrame
    fingerprint: str


class LeakageError(RuntimeError):
    """A query would read data that was not knowable at ``known_at``."""

    def __init__(self, event: LeakageEvent) -> None:
        super().__init__(f"leakage guard: {event.kind}: {event.detail}")
        self.event = event


class DataSource(Protocol):
    def query(self, q: PitQuery) -> QueryResult: ...


def _leakage(q: PitQuery, kind: str, **detail: Any) -> LeakageError:
    return LeakageError(
        LeakageEvent(
            source="i2",
            strategy_id=None,
            trial_id=None,
            severity="block",
            kind=kind,
            detail={"dataset": q.dataset, "start": q.start, "end": q.end, "known_at": q.known_at} | detail,
            observed_at=utc_now(),
        )
    )


def _file_digest(path: Path, cache: dict[tuple[str, int, int], str]) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in cache:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
    return cache[key]


class LocalParquetSource:
    """Serves configured Parquet datasets with I2 semantics, for synthetic data and offline tests."""

    def __init__(self, root: Path, cfg: LocalSourceSettings) -> None:
        self._root = root
        self._cfg = cfg
        self._digests: dict[tuple[str, int, int], str] = {}

    def path_of(self, dataset: str) -> Path:
        try:
            return self._root / self._cfg.datasets[dataset].path
        except KeyError as exc:
            raise KeyError(f"dataset {dataset!r} is not configured in local_source") from exc

    def query(self, q: PitQuery) -> QueryResult:
        path = self.path_of(q.dataset)
        if not path.exists():
            raise FileNotFoundError(f"dataset {q.dataset!r} not found at {path}")
        start, end, known_at = to_ns(q.start), to_ns(q.end), to_ns(q.known_at)
        if q.mode == "research" and end > known_at:
            raise _leakage(q, "end_after_known_at")
        frame = pd.read_parquet(path)
        vt, kt = self._cfg.valid_time_column, self._cfg.knowledge_time_column
        if kt not in frame.columns:
            if q.mode == "research":
                raise _leakage(q, "missing_knowledge_time")
            frame[kt] = frame[vt]
        mask = (frame[vt] >= start) & (frame[vt] < end) & (frame[kt] <= known_at)
        for column, value in (q.filters or {}).items():
            values = value if isinstance(value, list) else [value]
            mask &= frame[column].isin(values)
        frame = frame.loc[mask]
        keys = self._cfg.datasets[q.dataset].keys
        frame = frame.sort_values(kt, kind="stable").drop_duplicates(subset=keys, keep="last")
        frame = frame.sort_values(vt, kind="stable").reset_index(drop=True)
        if q.columns:
            frame = frame.loc[:, list(q.columns)]
        payload = json.dumps(q.canonical(), sort_keys=True) + _file_digest(path, self._digests)
        fingerprint = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
        log.debug("local_query", dataset=q.dataset, rows=len(frame), fingerprint=fingerprint)
        return QueryResult(frame=frame, fingerprint=fingerprint)


class PitHttpSource:
    """Client for I2 ``POST /v1/query`` returning an Arrow IPC stream and a fingerprint header."""

    def __init__(self, base_url: str, timeout_s: float, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def query(self, q: PitQuery) -> QueryResult:
        response = self._client.post("/v1/query", json=q.model_dump(mode="json", exclude_none=True))
        if response.status_code == HTTP_CONFLICT:
            raise LeakageError(LeakageEvent.model_validate(response.json()))
        response.raise_for_status()
        fingerprint = response.headers.get(FINGERPRINT_HEADER)
        if not fingerprint:
            raise ValueError(f"I2 response for {q.dataset!r} lacks {FINGERPRINT_HEADER}")
        reader = pa.ipc.open_stream(pa.BufferReader(response.content))
        return QueryResult(frame=reader.read_all().to_pandas(), fingerprint=fingerprint)

    def close(self) -> None:
        self._client.close()

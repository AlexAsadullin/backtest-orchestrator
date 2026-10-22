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
from typing import Any, Literal, Protocol

import httpx
import pandas as pd
import pyarrow as pa
import structlog

from bt_orchestrator.contracts.ledger import LeakageEvent
from bt_orchestrator.contracts.pit import FINGERPRINT_HEADER, LEAKAGE_FLAG_HEADER, PitFilter, PitQuery
from bt_orchestrator.settings import LocalSourceSettings
from bt_orchestrator.timeutil import to_ns, utc_now

log = structlog.get_logger(__name__)
HTTP_CONFLICT = 409


@dataclass(frozen=True)
class QueryResult:
    """Rows plus I2's fingerprint; ``leakage_flags`` lists kinds from I2's ``X-Leakage-Flag`` header."""

    frame: pd.DataFrame
    fingerprint: str
    leakage_flags: tuple[str, ...] = ()


class LeakageError(RuntimeError):
    """A query would read data that was not knowable at ``known_at``."""

    def __init__(self, event: LeakageEvent, *, reported: bool = False) -> None:
        super().__init__(f"leakage guard: {event.kind}: {event.detail}")
        self.event = event
        # True when the refusing service (the real I2) already sent the event to S6 itself.
        self.reported = reported


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


def filter_mask(frame: pd.DataFrame, filters: list[PitFilter]) -> pd.Series:
    """Boolean mask of the rows matching every filter (I2 operator semantics)."""
    mask = pd.Series(True, index=frame.index)
    for f in filters:
        col = frame[f.column]
        match f.op:
            case "in":
                mask &= col.isin(f.value if isinstance(f.value, list) else [f.value])
            case "==":
                mask &= col == f.value
            case "!=":
                mask &= col != f.value
            case "<":
                mask &= col < f.value
            case "<=":
                mask &= col <= f.value
            case ">":
                mask &= col > f.value
            case ">=":
                mask &= col >= f.value
    return mask


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
        mask &= filter_mask(frame, q.filters or [])
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
    """Client for I2 ``POST /v1/query``.

    ``fmt="arrow"`` reads the Arrow IPC stream (times stay int64 ns). ``fmt="json"`` reads I2's JSON body
    (``{fingerprint, columns, rows, flags}``), in which timestamp columns are ISO-8601 strings; the columns
    named in ``timestamp_columns`` are converted back to int64 ns, so both formats give the same frame.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        *,
        fmt: Literal["arrow", "json"] = "arrow",
        timestamp_columns: tuple[str, ...] = (),
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._fmt = fmt
        self._ts_columns = timestamp_columns

    def _json_frame(self, body: dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame(body.get("rows", []), columns=body.get("columns"))
        for column in self._ts_columns:
            if column in frame.columns:
                stamps = pd.to_datetime(frame[column], utc=True, format="ISO8601")
                frame[column] = stamps.astype("int64").where(stamps.notna())
        return frame

    def query(self, q: PitQuery) -> QueryResult:
        payload = q.model_copy(update={"format": self._fmt}).model_dump(mode="json", exclude_none=True)
        response = self._client.post("/v1/query", json=payload)
        if response.status_code == HTTP_CONFLICT:
            raise LeakageError(LeakageEvent.model_validate(response.json()), reported=True)
        response.raise_for_status()
        fingerprint = response.headers.get(FINGERPRINT_HEADER)
        if not fingerprint:
            raise ValueError(f"I2 response for {q.dataset!r} lacks {FINGERPRINT_HEADER}")
        raw_flags = response.headers.get(LEAKAGE_FLAG_HEADER, "")
        flags = tuple(f.strip() for f in raw_flags.split(",") if f.strip())
        if flags:
            log.warning("i2_leakage_flag", dataset=q.dataset, flags=flags)
        if self._fmt == "json":
            frame = self._json_frame(response.json())
        else:
            frame = pa.ipc.open_stream(pa.BufferReader(response.content)).read_all().to_pandas()
        return QueryResult(frame=frame, fingerprint=fingerprint, leakage_flags=flags)

    def close(self) -> None:
        self._client.close()

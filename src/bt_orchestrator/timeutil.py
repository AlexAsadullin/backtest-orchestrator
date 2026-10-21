"""UTC time helpers. Wire format is ISO-8601; storage is int64 nanoseconds since the epoch."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

NANOS_PER_SECOND = 1_000_000_000


def parse_iso(value: str | datetime) -> datetime:
    """Parse an ISO-8601 string (``Z`` allowed) into an aware UTC datetime; naive input is rejected."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {value!r} must be timezone-aware")
    return dt.astimezone(UTC)


def to_ns(value: str | datetime) -> int:
    """Convert an ISO string or aware datetime into int64 nanoseconds since the epoch (UTC)."""
    return int(pd.Timestamp(parse_iso(value)).value)


def ns_to_datetime(ns: int) -> datetime:
    """Convert nanoseconds since the epoch into an aware UTC datetime (microsecond precision)."""
    return pd.Timestamp(ns, unit="ns", tz="UTC").to_pydatetime()


def ns_to_iso(ns: int) -> str:
    """Render nanoseconds since the epoch as an ISO-8601 UTC string with nanosecond precision."""
    return pd.Timestamp(ns, unit="ns", tz="UTC").isoformat().replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(UTC)

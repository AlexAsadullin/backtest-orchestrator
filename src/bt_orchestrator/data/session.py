"""Exchange trading sessions: which wall-clock times are tradable, and trading time to wall time.

A :class:`SessionCalendar` is one daily window ``[open, close)`` in UTC on the given weekdays (MOEX main
session 10:00 to 18:40 Moscow time is 07:00 to 15:40 UTC; Moscow has no daylight saving time). The
synthetic market generates a continuous path in trading time and maps it onto the calendar, so nights
and weekends carry no quotes, trades or bars. Exchange holidays are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bt_orchestrator.timeutil import NANOS_PER_SECOND

NANOS_PER_DAY = 86_400 * NANOS_PER_SECOND
_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def parse_hhmm(value: str) -> int:
    """``"07:00"`` -> nanoseconds since midnight."""
    hours, minutes = value.split(":")
    return (int(hours) * 3600 + int(minutes) * 60) * NANOS_PER_SECOND


@dataclass(frozen=True)
class SessionCalendar:
    open_ns: int  # since UTC midnight
    close_ns: int
    weekdays: tuple[int, ...]  # 0 = Monday

    def __post_init__(self) -> None:
        if not 0 <= self.open_ns < self.close_ns <= NANOS_PER_DAY:
            raise ValueError("session must open before it closes, within one UTC day")
        if not self.weekdays or any(d not in range(7) for d in self.weekdays):
            raise ValueError("weekdays must be a non-empty subset of 0..6")

    @classmethod
    def from_strings(cls, open_utc: str, close_utc: str, weekdays: list[int]) -> SessionCalendar:
        return cls(parse_hhmm(open_utc), parse_hhmm(close_utc), tuple(weekdays))

    @property
    def length_ns(self) -> int:
        return self.close_ns - self.open_ns

    @property
    def seconds_per_day(self) -> float:
        return self.length_ns / NANOS_PER_SECOND

    @property
    def _weekmask(self) -> str:
        return " ".join(_WEEKDAY_NAMES[d] for d in sorted(self.weekdays))

    def to_wall(self, start_ns: int, offsets: np.ndarray, *, closing: bool = False) -> np.ndarray:
        """Wall times (int64 ns) of trading-time ``offsets`` after ``start_ns``, an in-session time.

        With ``closing``, an offset that lands exactly on a session boundary maps to the close of the
        earlier session instead of the next open (use it for interval ends).
        """
        if not self.in_session(np.array([start_ns], dtype="int64"))[0]:
            raise ValueError("the start of a session-mapped path must be inside a session")
        day0 = start_ns // NANOS_PER_DAY
        total = (start_ns - day0 * NANOS_PER_DAY - self.open_ns) + np.asarray(offsets, dtype="int64")
        k, within = np.divmod(total, self.length_ns)
        if closing:
            edge = (within == 0) & (k > 0)
            k = np.where(edge, k - 1, k)
            within = np.where(edge, self.length_ns, within)
        days = np.busday_offset(
            np.datetime64(int(day0), "D"), k, roll="forward", weekmask=self._weekmask
        ).astype("int64")
        return (days * NANOS_PER_DAY + self.open_ns + within).astype("int64")

    def in_session(self, ts: np.ndarray) -> np.ndarray:
        """Boolean mask: ``ts`` (int64 ns) falls inside a session window."""
        ts = np.asarray(ts, dtype="int64")
        tod = ts % NANOS_PER_DAY
        weekday = pd.to_datetime(ts, unit="ns", utc=True).weekday.to_numpy()
        return np.isin(weekday, self.weekdays) & (tod >= self.open_ns) & (tod < self.close_ns)

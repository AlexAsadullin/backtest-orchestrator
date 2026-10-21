"""Stationary bootstrap confidence intervals (Politis and Romano, 1994, JASA 89(428)).

Blocks start at uniform positions and have geometric lengths with mean ``block_len``, wrapping around the
series, so resamples keep the short-range dependence of bar returns. Intervals are percentile intervals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Statistic = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "level": self.level}


def stationary_indices(
    n: int, block_len: float, n_boot: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """``(n_boot, n)`` resampling indices of the stationary bootstrap."""
    if n <= 0:
        raise ValueError("series must not be empty")
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    new_block = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _interval(estimate: float, samples: NDArray[np.float64], level: float) -> Interval:
    alpha = (1.0 - level) / 2.0
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return Interval(estimate, float("nan"), float("nan"), level)
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return Interval(estimate, float(low), float(high), level)


def bootstrap_ci(
    x: NDArray[np.float64],
    stat: Statistic,
    *,
    n_boot: int,
    block_len: float,
    level: float,
    seed: int,
) -> Interval:
    rng = np.random.default_rng(seed)
    idx = stationary_indices(len(x), block_len, n_boot, rng)
    samples = np.array([stat(x[row]) for row in idx])
    return _interval(stat(x), samples, level)


def paired_diff_ci(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    stat: Statistic,
    *,
    n_boot: int,
    block_len: float,
    level: float,
    seed: int,
) -> Interval:
    """CI of ``stat(b) - stat(a)`` resampling the same time blocks for both aligned series."""
    if len(a) != len(b):
        raise ValueError("paired series must have equal length")
    rng = np.random.default_rng(seed)
    idx = stationary_indices(len(a), block_len, n_boot, rng)
    samples = np.array([stat(b[row]) - stat(a[row]) for row in idx])
    return _interval(stat(b) - stat(a), samples, level)


def independent_diff_ci(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    stat: Statistic,
    *,
    n_boot: int,
    block_len: float,
    level: float,
    seed: int,
) -> Interval:
    """CI of ``stat(b) - stat(a)`` for series on different paths (resampled independently)."""
    rng = np.random.default_rng(seed)
    ia = stationary_indices(len(a), block_len, n_boot, rng)
    ib = stationary_indices(len(b), block_len, n_boot, rng)
    samples = np.array([stat(b[rb]) - stat(a[ra]) for ra, rb in zip(ia, ib, strict=True)])
    return _interval(stat(b) - stat(a), samples, level)

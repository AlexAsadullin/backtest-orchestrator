from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bt_orchestrator.analytics.bootstrap import (
    bootstrap_ci,
    independent_diff_ci,
    paired_diff_ci,
    stationary_indices,
)
from bt_orchestrator.analytics.metrics import (
    compute_metrics,
    max_drawdown,
    sharpe,
    sharpe_se_lo,
    sortino,
    total_return,
)
from bt_orchestrator.domain import empty_fills

KW = {"n_boot": 300, "block_len": 5.0, "level": 0.95, "seed": 1}


def test_stationary_indices_shape_and_range() -> None:
    idx = stationary_indices(50, 5.0, 20, np.random.default_rng(0))
    assert idx.shape == (20, 50)
    assert idx.min() >= 0
    assert idx.max() < 50
    with pytest.raises(ValueError, match="empty"):
        stationary_indices(0, 5.0, 1, np.random.default_rng(0))


def test_bootstrap_ci_covers_the_mean() -> None:
    x = np.random.default_rng(3).normal(1.0, 1.0, 400)
    ci = bootstrap_ci(x, lambda r: float(np.mean(r)), **KW)
    assert ci.low < 1.0 < ci.high
    assert ci.as_dict()["level"] == 0.95


def test_bootstrap_ci_all_nan_samples() -> None:
    ci = bootstrap_ci(np.ones(10), lambda r: float("nan"), **KW)
    assert math.isnan(ci.low)
    assert math.isnan(ci.high)


def test_paired_and_independent_differences() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 1.0, 300)
    b = a + 0.5
    paired = paired_diff_ci(a, b, lambda r: float(np.mean(r)), **KW)
    assert paired.estimate == pytest.approx(0.5)
    assert paired.low == pytest.approx(0.5)
    indep = independent_diff_ci(a, b, lambda r: float(np.mean(r)), **KW)
    assert indep.low < 0.5 < indep.high
    assert indep.high - indep.low > paired.high - paired.low
    with pytest.raises(ValueError, match="equal length"):
        paired_diff_ci(a, b[:-1], np.mean, **KW)  # type: ignore[arg-type]


def test_sharpe_and_friends() -> None:
    r = np.array([0.01, -0.005, 0.02, 0.0])
    assert sharpe(r, 1.0) == pytest.approx(np.mean(r) / np.std(r, ddof=1))
    assert sharpe(np.array([0.01]), 1.0) == 0.0
    assert sharpe(np.zeros(5), 1.0) == 0.0
    assert math.isnan(sharpe_se_lo(np.array([0.1]), 1.0))
    assert sharpe_se_lo(r, 4.0) == pytest.approx(2 * sharpe_se_lo(r, 1.0))
    assert sortino(np.array([0.01, 0.02]), 1.0) == 0.0
    assert sortino(r, 1.0) > 0
    assert total_return(np.array([0.1, -0.1])) == pytest.approx(-0.01)
    assert max_drawdown(np.array([1.0, 2.0, 1.0, 3.0])) == pytest.approx(0.5)
    assert max_drawdown(np.array([])) == 0.0


def test_compute_metrics_without_fills() -> None:
    ledger = pd.DataFrame({"ret": [0.0, 0.01, -0.01], "equity": [100.0, 101.0, 99.99], "position": [0, 1, 0]})
    m = compute_metrics(ledger, empty_fills(), 252.0, 100.0)
    assert m["n_fills"] == 0
    assert m["fees"] == 0.0
    assert m["turnover"] == 0.0
    assert m["hit_rate"] == pytest.approx(0.5)
    assert m["n_periods"] == 3
    assert m["pnl"] == pytest.approx(-0.01)

"""Standard performance metrics on per-bar net returns.

Sharpe ratio: ``mean(r) / std(r, ddof=1) * sqrt(periods_per_year)``. Its standard error uses Lo (2002),
"The Statistics of Sharpe Ratios", eq. (9) for i.i.d. returns: ``SE(SR_period) = sqrt((1 + SR^2 / 2) / n)``,
annualised by ``sqrt(periods_per_year)``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def sharpe(returns: NDArray[np.float64], periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return float(np.mean(returns)) / std * math.sqrt(periods_per_year)


def sharpe_se_lo(returns: NDArray[np.float64], periods_per_year: float) -> float:
    """Lo (2002) i.i.d. standard error of the annualised Sharpe ratio."""
    n = len(returns)
    if n < 2:
        return float("nan")
    sr_period = sharpe(returns, 1.0)
    return math.sqrt((1.0 + 0.5 * sr_period**2) / n) * math.sqrt(periods_per_year)


def sortino(returns: NDArray[np.float64], periods_per_year: float) -> float:
    downside = np.minimum(returns, 0.0)
    dd = float(np.sqrt(np.mean(downside**2))) if len(returns) else 0.0
    if dd == 0.0:
        return 0.0
    return float(np.mean(returns)) / dd * math.sqrt(periods_per_year)


def max_drawdown(equity: NDArray[np.float64]) -> float:
    """Largest peak-to-trough fall as a positive fraction of the peak."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def total_return(returns: NDArray[np.float64]) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def compute_metrics(
    ledger: pd.DataFrame, fills: pd.DataFrame, periods_per_year: float, initial_cash: float
) -> dict[str, float | int | None]:
    """Metrics of one run from its per-bar ledger and fills."""
    r = ledger["ret"].to_numpy(dtype="float64")
    equity = np.r_[initial_cash, ledger["equity"].to_numpy(dtype="float64")]
    notional = float((fills["qty"] * fills["price"]).sum()) if not fills.empty else 0.0
    position = ledger["position"].to_numpy()
    active = r[r != 0.0]
    return {
        "sharpe": sharpe(r, periods_per_year),
        "sharpe_se": sharpe_se_lo(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "total_return": total_return(r),
        "ann_return": float(np.mean(r) * periods_per_year) if len(r) else 0.0,
        "ann_vol": float(np.std(r, ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 else 0.0,
        "max_drawdown": max_drawdown(equity),
        "pnl": float(equity[-1] - initial_cash),
        "n_trades": len(fills),
        "fees": float(fills["fee"].sum()) if not fills.empty else 0.0,
        "turnover": notional / initial_cash,
        "exposure": float(np.mean(np.abs(position) > 0)) if len(position) else 0.0,
        "hit_rate": float(np.mean(active > 0)) if len(active) else None,
        "n_periods": len(r),
    }

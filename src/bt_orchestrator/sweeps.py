"""Parameter sweeps (grid and random) and walk-forward evaluation. Every trial is logged to S6.

Walk-forward: the bar sample is cut into windows of ``train_bars`` followed by ``test_bars``, moving by
``step_bars``. In each window the sweep picks the parameters with the best in-sample objective, then
runs them out of sample; the out-of-sample returns are stitched in time order.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from bt_orchestrator.analytics.bootstrap import Interval, bootstrap_ci
from bt_orchestrator.analytics.metrics import sharpe
from bt_orchestrator.domain import DataBundle, Fidelity
from bt_orchestrator.orchestrator import Orchestrator, RunSpec
from bt_orchestrator.strategies.base import StrategyParams
from bt_orchestrator.timeutil import ns_to_iso


@dataclass(frozen=True)
class SweepTrial:
    params: dict[str, Any]
    metrics: dict[str, float | int | None]
    trial_id: str


@dataclass(frozen=True)
class SweepResult:
    strategy: str
    fidelity: Fidelity
    objective: str
    trials: list[SweepTrial]
    n_invalid: int

    def best(self) -> SweepTrial:
        if not self.trials:
            raise ValueError("sweep produced no valid trials")
        return max(self.trials, key=lambda t: float(t.metrics[self.objective] or -np.inf))


def grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))]


def random_points(
    ranges: dict[str, list[float]], model: type[StrategyParams], n: int, seed: int
) -> list[dict[str, Any]]:
    """Uniform draws inside ``[low, high]`` per parameter; integer fields are rounded."""
    rng = np.random.default_rng(seed)
    fields = model.model_fields
    points = []
    for _ in range(n):
        point: dict[str, Any] = {}
        for name, (low, high) in sorted(ranges.items()):
            value = rng.uniform(low, high)
            point[name] = round(value) if fields[name].annotation is int else float(value)
        points.append(point)
    return points


def run_sweep(
    orch: Orchestrator,
    spec: RunSpec,
    points: list[dict[str, Any]],
    fidelity: Fidelity,
    *,
    bundles: dict[str, DataBundle] | None = None,
    tags: tuple[str, ...] = ("sweep",),
) -> SweepResult:
    """Run every valid parameter point and log all trials in one batch."""
    strategy = orch.registry.get(spec.strategy)
    data = bundles or {sym: orch.load(spec, sym, fidelity) for sym in spec.universe}
    trials, records, invalid = [], [], 0
    for point in points:
        try:
            orch.params(strategy, point)
        except ValidationError:
            invalid += 1
            continue
        run = orch.run_level(
            replace(spec, params=point, tags=(*spec.tags, *tags)),
            fidelity,
            log_trial=False,
            bundles=data,
        )
        records.append(run.trial)
        trials.append(SweepTrial(params=run.trial.params, metrics=run.metrics, trial_id=run.trial.trial_id))
    orch.ledger.log_trials(records)
    return SweepResult(spec.strategy, fidelity, orch.settings.sweep.objective, trials, invalid)


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict[str, Any]
    in_sample: float
    out_of_sample: float
    test_trial_id: str


@dataclass(frozen=True)
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    oos_returns: pd.Series
    oos_sharpe: Interval
    n_trials: int


def _slice(bundle: DataBundle, lo: int, hi: int) -> DataBundle:
    return replace(bundle, bars=bundle.bars.iloc[lo:hi].reset_index(drop=True))


def walk_forward(orch: Orchestrator, spec: RunSpec, points: list[dict[str, Any]]) -> WalkForwardResult:
    s = orch.settings
    wf = s.walk_forward
    if wf.fidelity not in (Fidelity.VECTORISED, Fidelity.EVENT):
        raise ValueError("walk-forward runs on bar-level fidelities")
    full = {sym: orch.load(spec, sym, wf.fidelity) for sym in spec.universe}
    n = min(len(b.bars) for b in full.values())
    first = next(iter(full.values())).bars["ts"].to_numpy()
    windows: list[WalkForwardWindow] = []
    oos: list[pd.Series] = []
    n_trials = 0
    lo = 0
    while lo + wf.train_bars + wf.test_bars <= n:
        mid, hi = lo + wf.train_bars, lo + wf.train_bars + wf.test_bars
        iso = [ns_to_iso(int(first[i])) for i in (lo, mid - 1, mid, hi - 1)]
        train = {k: _slice(v, lo, mid) for k, v in full.items()}
        test = {k: _slice(v, mid, hi) for k, v in full.items()}
        sweep = run_sweep(orch, spec, points, wf.fidelity, bundles=train, tags=("walk-forward", "train"))
        best = sweep.best()
        run = orch.run_level(
            replace(spec, params=best.params, tags=(*spec.tags, "walk-forward", "test")),
            wf.fidelity,
            bundles=test,
        )
        n_trials += len(sweep.trials) + 1
        oos.append(run.returns)
        windows.append(
            WalkForwardWindow(
                train_start=iso[0],
                train_end=iso[1],
                test_start=iso[2],
                test_end=iso[3],
                best_params=best.params,
                in_sample=float(best.metrics[sweep.objective] or 0.0),
                out_of_sample=float(run.metrics[sweep.objective] or 0.0),
                test_trial_id=run.trial.trial_id,
            )
        )
        lo += wf.step_bars
    if not windows:
        raise ValueError(f"need at least {wf.train_bars + wf.test_bars} bars for one window, got {n}")
    stitched = pd.concat(oos)
    stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()
    m = s.metrics
    ci = bootstrap_ci(
        stitched.to_numpy(),
        lambda r: sharpe(r, s.periods_per_year),
        n_boot=m.n_boot,
        block_len=m.block_len,
        level=m.ci_level,
        seed=m.seed,
    )
    return WalkForwardResult(windows=windows, oos_returns=stitched, oos_sharpe=ci, n_trials=n_trials)

"""Fidelity ladder: vectorised -> event -> tick/queue -> reactive sim, with promotion gates.

A strategy moves up a level only if its run passes that level's promotion criteria (from settings).
Each level reports its metrics with stationary-bootstrap CIs and the Sharpe/total-return drop versus the
previous level: paired (same time blocks) when both levels ran on the same bar grid, independent when
the level ran on a different path (the simulator tape).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from bt_orchestrator.analytics.bootstrap import (
    Interval,
    bootstrap_ci,
    independent_diff_ci,
    paired_diff_ci,
)
from bt_orchestrator.analytics.metrics import sharpe, total_return
from bt_orchestrator.domain import DataBundle, Fidelity
from bt_orchestrator.orchestrator import LevelRun, Orchestrator, RunSpec, new_run_id
from bt_orchestrator.settings import MetricsSettings, PromotionCriteria

Status = Literal["passed", "failed", "skipped", "not_run"]


class LevelReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fidelity: Fidelity
    engine: str | None
    status: Status
    reasons: list[str]
    gate_blocked_before: bool
    metrics: dict[str, float | int | None]
    sharpe_ci: dict[str, float] | None
    total_return_ci: dict[str, float] | None
    sharpe_drop_ci: dict[str, float] | None
    total_return_drop_ci: dict[str, float] | None
    drop_method: Literal["paired", "independent"] | None
    trial_id: str | None
    wall_time_s: float | None
    fill_models: dict[str, int | None]
    meta: dict[str, Any]


class LadderReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    run_id: str
    strategy: str
    params: dict[str, Any]
    universe: list[str]
    start: str
    end: str
    known_at: str
    promoted_to: Fidelity | None
    levels: list[LevelReport]

    def top_run(self) -> LevelReport | None:
        ran = [lv for lv in self.levels if lv.trial_id is not None]
        return ran[-1] if ran else None


def evaluate_gate(
    metrics: dict[str, float | int | None],
    criteria: PromotionCriteria,
    previous_sharpe: float | None,
) -> list[str]:
    """Reasons the run fails its promotion criteria (empty: it passes)."""
    reasons: list[str] = []
    sr = float(metrics["sharpe"] or 0.0)
    trades = int(metrics["n_fills"] or 0)
    mdd = float(metrics["max_drawdown"] or 0.0)
    if sr < criteria.min_sharpe:
        reasons.append(f"sharpe {sr:.2f} < {criteria.min_sharpe}")
    if trades < criteria.min_fills:
        reasons.append(f"fills {trades} < {criteria.min_fills}")
    if mdd > criteria.max_drawdown:
        reasons.append(f"max drawdown {mdd:.3f} > {criteria.max_drawdown}")
    if criteria.max_sharpe_drop is not None and previous_sharpe is not None:
        drop = previous_sharpe - sr
        if drop > criteria.max_sharpe_drop:
            reasons.append(f"sharpe drop {drop:.2f} > {criteria.max_sharpe_drop}")
    return reasons


class _Stats:
    def __init__(self, cfg: MetricsSettings, periods_per_year: float) -> None:
        self._cfg = cfg
        self._ppy = periods_per_year

    def sharpe(self, r: np.ndarray) -> float:
        return sharpe(r, self._ppy)

    def _kw(self) -> dict[str, Any]:
        c = self._cfg
        return {"n_boot": c.n_boot, "block_len": c.block_len, "level": c.ci_level, "seed": c.seed}

    def level(self, r: np.ndarray) -> tuple[Interval, Interval]:
        return (
            bootstrap_ci(r, self.sharpe, **self._kw()),
            bootstrap_ci(r, total_return, **self._kw()),
        )

    def drop(self, prev: pd.Series, cur: pd.Series) -> tuple[Interval, Interval, Literal["paired", "independent"]]:
        """CI of ``prev - cur`` (positive = performance lost moving up)."""
        common = prev.index.intersection(cur.index)
        paired = len(common) >= 0.9 * max(len(prev), len(cur))

        if paired:
            a, b = prev.loc[common].to_numpy(), cur.loc[common].to_numpy()
            sr = paired_diff_ci(b, a, self.sharpe, **self._kw())
            tr = paired_diff_ci(b, a, total_return, **self._kw())
            return sr, tr, "paired"
        a, b = prev.to_numpy(), cur.to_numpy()
        sr = independent_diff_ci(b, a, self.sharpe, **self._kw())
        tr = independent_diff_ci(b, a, total_return, **self._kw())
        return sr, tr, "independent"


def run_ladder(
    orch: Orchestrator,
    spec: RunSpec,
    top: Fidelity | None = None,
    *,
    run_all_levels: bool | None = None,
) -> tuple[LadderReport, list[LevelRun]]:
    """Climb the configured levels up to ``top`` (default: all)."""
    s = orch.settings
    strategy = orch.registry.get(spec.strategy)
    levels = list(s.ladder.levels)
    if top is not None:
        levels = levels[: levels.index(top) + 1]
    run_all = s.ladder.run_all_levels if run_all_levels is None else run_all_levels
    stats = _Stats(s.metrics, s.periods_per_year)
    reports: list[LevelReport] = []
    runs: list[LevelRun] = []
    blocked = False
    previous: LevelRun | None = None
    promoted: Fidelity | None = None
    bar_bundles: dict[str, DataBundle] = {}

    for fidelity in levels:
        if fidelity is Fidelity.VECTORISED and not strategy.supports_vectorised:
            reports.append(_empty(fidelity, "skipped", ["strategy has no vectorised form"], blocked))
            continue
        if blocked and not run_all:
            reports.append(_empty(fidelity, "not_run", ["blocked by a failed gate below"], blocked))
            continue
        share = fidelity in (Fidelity.VECTORISED, Fidelity.EVENT)
        if share and not bar_bundles:
            bar_bundles = {sym: orch.load(spec, sym, fidelity) for sym in spec.universe}
        run = orch.run_level(
            spec,
            fidelity,
            bundles=bar_bundles if share else None,
        )
        runs.append(run)
        sr_ci, tr_ci = stats.level(run.returns.to_numpy())
        drop_sr = drop_tr = None
        method: Literal["paired", "independent"] | None = None
        if previous is not None:
            d_sr, d_tr, method = stats.drop(previous.returns, run.returns)
            drop_sr, drop_tr = d_sr.as_dict(), d_tr.as_dict()
        prev_sr = float(previous.metrics["sharpe"] or 0.0) if previous is not None else None
        reasons = evaluate_gate(run.metrics, s.ladder.promotion[fidelity], prev_sr)
        status: Status = "failed" if reasons else "passed"
        if not reasons and not blocked:
            promoted = fidelity
        meta: dict[str, Any] = {}
        for r in run.results:
            meta[r.symbol] = {k: v for k, v in r.meta.items() if isinstance(v, (str, int, float))}
        reports.append(
            LevelReport(
                fidelity=fidelity,
                engine=run.engine,
                status=status,
                reasons=reasons,
                gate_blocked_before=blocked,
                metrics=run.metrics,
                sharpe_ci=sr_ci.as_dict(),
                total_return_ci=tr_ci.as_dict(),
                sharpe_drop_ci=drop_sr,
                total_return_drop_ci=drop_tr,
                drop_method=method,
                trial_id=run.trial.trial_id,
                wall_time_s=run.wall_time_s,
                fill_models=run.fill_models,
                meta=meta,
            )
        )
        blocked = blocked or bool(reasons)
        previous = run

    report = LadderReport(
        run_id=new_run_id(),
        strategy=spec.strategy,
        params=orch.params(strategy, spec.params).model_dump(),
        universe=spec.universe,
        start=spec.start,
        end=spec.end,
        known_at=spec.known_at,
        promoted_to=promoted,
        levels=reports,
    )
    return report, runs


def _empty(fidelity: Fidelity, status: Status, reasons: list[str], blocked: bool) -> LevelReport:
    return LevelReport(
        fidelity=fidelity,
        engine=None,
        status=status,
        reasons=reasons,
        gate_blocked_before=blocked,
        metrics={},
        sharpe_ci=None,
        total_return_ci=None,
        sharpe_drop_ci=None,
        total_return_drop_ci=None,
        drop_method=None,
        trial_id=None,
        wall_time_s=None,
        fill_models={},
        meta={},
    )

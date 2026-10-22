"""Experiments behind docs/report.md (``make report``).

Every experiment runs on synthetic markets with fixed seeds (``experiments.seeds``), one private data
subtree per seed, and a private ledger inbox, because the runs test the method and are not research
ideas. Results go to ``$TRADEQ_DATA_DIR/backtest-orchestrator/reports/experiments.json`` and
``tables.md``.

E1 Fill models against the generating process
    The synthetic I6 export has a known fill hazard and a known slippage law. On the out-of-time test
    split we compare the calibrated logistic fill probability, the touch-fills assumption of bar engines
    (p = 1), a constant base rate, and a fills-only model (no orders extension), each against the true
    probability; and the calibrated slippage regression against zero and half-spread slippage, each
    against the true expected slippage. The regression coefficients are compared with the true ones.
E2 Fidelity ladder
    Both strategies with default parameters climb every rung (gates evaluated, climbing not stopped).
    Per rung: Sharpe, fills, engine wall time; per step: the paired-bootstrap Sharpe drop.
E3 Candidate screening
    Every grid point of both strategies runs on the vectorised, event and tick rungs. Three promotion
    policies are compared offline on the same runs: the gated ladder, vectorbt alone, hftbacktest alone.
    The reference verdict is the tick rung's gate (the highest-fidelity replay of the same tape).
E4 Calibrated against uncalibrated fills inside the engines
    The event and tick rungs run with and without the calibrated models; the distance to the tick rung
    with calibrated fills measures how much each event-rung variant misstates the strategy.
E5 Parity and injected faults
    Real parity: vectorbt against NautilusTrader with matched costs and touch fills. Fault injection:
    known bugs are planted in a copy of the vectorbt run, and three detectors (the execution-level
    classifier, a PnL-only tolerance, a fill-count check) are scored on whether they flag each bug.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from bt_orchestrator.calibration.fit import (
    build_samples,
    fit_fill_prob,
    fit_slippage,
    load_calibration_input,
    predict_fill_prob,
    predict_slippage,
    split_out_of_time,
)
from bt_orchestrator.contracts.ledger import Actor
from bt_orchestrator.data.synthetic_store import SyntheticPaths, write_synthetic
from bt_orchestrator.domain import Fidelity, make_fills
from bt_orchestrator.engines.base import EngineResult, mark_to_market
from bt_orchestrator.engines.nautilus_engine import average_daily_volume
from bt_orchestrator.ladder import evaluate_gate, run_ladder
from bt_orchestrator.orchestrator import Orchestrator, RunSpec
from bt_orchestrator.parity import compare_runs, run_parity
from bt_orchestrator.settings import ExperimentSettings, Settings
from bt_orchestrator.sweeps import grid_points
from bt_orchestrator.wiring import build_orchestrator

BPS = 1e4
SCREEN_LEVELS = (Fidelity.VECTORISED, Fidelity.EVENT, Fidelity.TICK)
FAULTS = ("fee_rate", "one_bar_delay", "fill_price", "missing_fills")
DETECTORS = ("classifier", "pnl_only", "fill_count")


# -- statistics ------------------------------------------------------------------------------------


def t_interval(values: Sequence[float], level: float) -> dict[str, float]:
    """Mean with a Student t interval across independent seeds (n - 1 degrees of freedom)."""
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype="float64")
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    mean = float(x.mean())
    if n < 2:
        return {"mean": mean, "low": float("nan"), "high": float("nan"), "n": n}
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": n}


def wilson_interval(k: int, n: int, level: float) -> dict[str, float]:
    """Wilson score interval for a proportion k / n."""
    if n == 0:
        return {"rate": float("nan"), "low": float("nan"), "high": float("nan"), "k": k, "n": n}
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": p, "low": max(0.0, centre - half), "high": min(1.0, centre + half), "k": k, "n": n}


def paired_t_test(a: Sequence[float], b: Sequence[float], level: float) -> dict[str, float]:
    """Student's t-test for paired samples on ``a - b`` (two-sided)."""
    d = np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")
    n = len(d)
    if n < 2 or float(d.std(ddof=1)) == 0.0:
        return {"t": float("nan"), "df": float(n - 1), "p": float("nan"), "critical": float("nan")}
    t = float(d.mean() / (d.std(ddof=1) / math.sqrt(n)))
    p = float(2 * stats.t.sf(abs(t), n - 1))
    critical = float(stats.t.ppf(0.5 + level / 2, n - 1))
    return {"t": t, "df": float(n - 1), "p": p, "critical": critical}


# -- per-seed setup --------------------------------------------------------------------------------


def seed_settings(base: Settings, seed: int) -> Settings:
    """Settings of one experiment seed: own data subtree, own market seed, private ledger inbox."""
    e = base.experiments
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(update={"subtree": f"{e.subtree}/seed-{seed}"}),
            "synthetic": base.synthetic.model_copy(update={"seed": seed, "n_bars": e.n_bars}),
            "ledger": base.ledger.model_copy(update={"mode": "inbox", "inbox_subdir": e.ledger_inbox_subdir}),
            "sim": base.sim.model_copy(update={"mode": "fake"}),
            "pit": base.pit.model_copy(update={"mode": "local"}),
        }
    )


@dataclass
class SeedContext:
    seed: int
    settings: Settings
    orch: Orchestrator
    paths: SyntheticPaths
    symbol: str

    def spec(self, strategy: str, params: dict[str, Any] | None = None, tag: str = "experiment") -> RunSpec:
        return RunSpec(
            strategy=strategy,
            params=params or {},
            universe=[self.symbol],
            start=self.paths.start,
            end=self.paths.end,
            known_at=self.paths.end,
            seed=self.seed,
            actor=Actor(kind="robot", id="bt-experiments"),
            tags=("experiment", tag),
        )


def prepare_seed(base: Settings, seed: int) -> SeedContext:
    s = seed_settings(base, seed)
    if s.project_dir.exists():
        shutil.rmtree(s.project_dir)
    symbol = s.market.default_universe[0]
    paths = write_synthetic(s, [symbol])
    orch = build_orchestrator(s)
    orch.calibrate(symbol, fills_root=paths.fills_root, orders_root=paths.orders_root)
    return SeedContext(seed=seed, settings=s, orch=orch, paths=paths, symbol=symbol)


# -- E1 fill models against the generating process ------------------------------------------------


def true_fill_prob(s: Settings, frame: pd.DataFrame) -> np.ndarray:
    """P(fill within the horizon) under the synthetic hazard, at the estimated volatility."""
    f = s.synthetic.fills
    hazard = f.base_hazard_per_s * np.exp(
        -f.distance_coef * frame["distance_to_touch_bps"].to_numpy()
        - f.queue_coef * np.log1p(frame["queue_ahead"].to_numpy())
        + f.vol_coef * (frame["vol_bps_sqrt_s"].to_numpy() - s.synthetic.vol_bps_per_sqrt_s)
    )
    horizon_s = s.calibration.horizon_ns / 1e9
    result: np.ndarray = 1.0 - np.exp(-hazard * horizon_s)
    return result


def true_slippage(s: Settings, frame: pd.DataFrame) -> np.ndarray:
    """Expected taker slippage (bps) under the synthetic law: half spread plus square-root impact."""
    f = s.synthetic.fills
    result: np.ndarray = (
        frame["half_spread_bps"].to_numpy() + f.impact_coef_bps * frame["sqrt_participation"].to_numpy()
    )
    return result


def experiment_fill_models(ctx: SeedContext) -> dict[str, float]:
    s = ctx.settings
    venue = s.market.venue
    bars = ctx.orch.market.load(
        ctx.symbol, ctx.paths.start, ctx.paths.end, ctx.paths.end, with_tape=False
    ).bars
    adv = average_daily_volume(bars, s.bar_interval_ns, s.market.seconds_per_day)
    full = build_samples(
        load_calibration_input(ctx.paths.fills_root, ctx.paths.orders_root, venue, ctx.symbol),
        adv,
        s.calibration,
    )
    fills_only = build_samples(
        load_calibration_input(ctx.paths.fills_root, None, venue, ctx.symbol), adv, s.calibration
    )
    lim_train, lim_test = split_out_of_time(full.limit, s.calibration.test_fraction)
    tak_train, tak_test = split_out_of_time(full.taker, s.calibration.test_fraction)
    fo_train, _ = split_out_of_time(fills_only.limit, s.calibration.test_fraction)
    prob = fit_fill_prob(lim_train, s.calibration)
    slip = fit_slippage(tak_train)

    y = lim_test["label"].to_numpy(dtype="float64")
    p_true = true_fill_prob(s, lim_test)
    p_cal = predict_fill_prob(prob, lim_test)
    base_rate = float(lim_train["label"].mean())
    out: dict[str, float] = {
        "n_limit_test": float(len(y)),
        "n_taker_test": float(len(tak_test)),
        "fill_rate_test": float(y.mean()),
        "prob_mae_calibrated": float(np.mean(np.abs(p_cal - p_true))),
        "prob_mae_touch_fills": float(np.mean(np.abs(1.0 - p_true))),
        "prob_mae_base_rate": float(np.mean(np.abs(base_rate - p_true))),
        "brier_calibrated": float(np.mean((p_cal - y) ** 2)),
        "brier_touch_fills": float(np.mean((1.0 - y) ** 2)),
        "brier_base_rate": float(np.mean((base_rate - y) ** 2)),
        "brier_true_prob": float(np.mean((p_true - y) ** 2)),
    }
    # A fills-only export has no unfilled orders, so a logistic fit is impossible when every order filled
    # within the horizon; its implied probability is then the share of fills inside the horizon.
    try:
        fo_prob = fit_fill_prob(fo_train, s.calibration)
        p_fo = predict_fill_prob(fo_prob, lim_test)
    except ValueError:
        p_fo = np.full(len(lim_test), float(fo_train["label"].mean()))
    out["prob_mae_fills_only"] = float(np.mean(np.abs(p_fo - p_true)))
    out["brier_fills_only"] = float(np.mean((p_fo - y) ** 2))
    out["fill_rate_fills_only_train"] = float(fo_train["label"].mean())

    expected = true_slippage(s, tak_test)
    pred = predict_slippage(slip, tak_test)
    half = tak_test["half_spread_bps"].to_numpy()
    out |= {
        "slip_mae_calibrated": float(np.mean(np.abs(pred - expected))),
        "slip_mae_zero": float(np.mean(np.abs(expected))),
        "slip_mae_half_spread": float(np.mean(np.abs(half - expected))),
        "slip_coef_intercept": slip.intercept,
        "slip_coef_half_spread": slip.coef[0],
        "slip_coef_sqrt_participation": slip.coef[1],
        "slip_coef_vol": slip.coef[2],
        "slip_resid_std": slip.resid_std_bps,
    }
    return out


# -- E2 fidelity ladder ----------------------------------------------------------------------------


def experiment_ladder(ctx: SeedContext, strategies: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        report, _ = run_ladder(ctx.orch, ctx.spec(strategy, tag="ladder"), run_all_levels=True)
        for lv in report.levels:
            rows.append(
                {
                    "seed": ctx.seed,
                    "strategy": strategy,
                    "fidelity": lv.fidelity.value,
                    "status": lv.status,
                    "sharpe": lv.metrics.get("sharpe"),
                    "n_fills": lv.metrics.get("n_fills"),
                    "total_return": lv.metrics.get("total_return"),
                    "fees": lv.metrics.get("fees"),
                    "wall_time_s": lv.wall_time_s,
                    "sharpe_drop": lv.sharpe_drop_ci["estimate"] if lv.sharpe_drop_ci else None,
                    "sharpe_drop_low": lv.sharpe_drop_ci["low"] if lv.sharpe_drop_ci else None,
                    "sharpe_drop_high": lv.sharpe_drop_ci["high"] if lv.sharpe_drop_ci else None,
                    "drop_method": lv.drop_method,
                    "promoted_to": report.promoted_to.value if report.promoted_to else None,
                }
            )
    return rows


# -- E3 candidate screening ------------------------------------------------------------------------


def experiment_screening(ctx: SeedContext, strategies: Sequence[str]) -> list[dict[str, Any]]:
    """Run every valid grid point on every screening rung; the policies are scored afterwards."""
    rows: list[dict[str, Any]] = []
    orch = ctx.orch
    for strategy in strategies:
        st = ctx.settings.strategy_settings(strategy)
        model = orch.registry.get(strategy)
        bar_bundles = {ctx.symbol: orch.load(ctx.spec(strategy), ctx.symbol, Fidelity.EVENT)}
        tick_bundles = {ctx.symbol: orch.load(ctx.spec(strategy), ctx.symbol, Fidelity.TICK)}
        for k, point in enumerate(grid_points(st.grid)):
            try:
                orch.params(model, point)
            except ValueError:
                continue
            spec = ctx.spec(strategy, point, tag="screening")
            row: dict[str, Any] = {"seed": ctx.seed, "strategy": strategy, "candidate": k, "params": point}
            for fidelity in SCREEN_LEVELS:
                bundles = tick_bundles if fidelity is Fidelity.TICK else bar_bundles
                run = orch.run_level(spec, fidelity, bundles=bundles)
                row[f"{fidelity.value}_sharpe"] = run.metrics["sharpe"]
                row[f"{fidelity.value}_n_fills"] = run.metrics["n_fills"]
                row[f"{fidelity.value}_max_drawdown"] = run.metrics["max_drawdown"]
                row[f"{fidelity.value}_wall_s"] = run.wall_time_s
            rows.append(row)
    return rows


def _metrics(row: dict[str, Any], level: str) -> dict[str, float | int | None]:
    return {
        "sharpe": row[f"{level}_sharpe"],
        "n_fills": row[f"{level}_n_fills"],
        "max_drawdown": row[f"{level}_max_drawdown"],
    }


def score_policies(rows: list[dict[str, Any]], s: Settings) -> dict[str, Any]:
    """Promotion verdicts and compute cost of the three policies against the tick-rung verdict."""
    crit = s.ladder.promotion
    level = s.metrics.ci_level
    per_policy: dict[str, dict[str, list[float]]] = {
        p: {"promoted": [], "false": [], "missed": [], "cost": []}
        for p in ("ladder", "vectorbt_only", "tick_only")
    }
    n_reference = 0
    for row in rows:
        vec = not evaluate_gate(_metrics(row, "vectorised"), crit[Fidelity.VECTORISED], None)
        vec_sr = float(row["vectorised_sharpe"] or 0.0)
        ev = not evaluate_gate(_metrics(row, "event"), crit[Fidelity.EVENT], vec_sr)
        ev_sr = float(row["event_sharpe"] or 0.0)
        tick_ladder = not evaluate_gate(_metrics(row, "tick"), crit[Fidelity.TICK], ev_sr)
        reference = not evaluate_gate(_metrics(row, "tick"), crit[Fidelity.TICK], None)
        n_reference += int(reference)
        verdicts = {
            "ladder": vec and ev and tick_ladder,
            "vectorbt_only": vec,
            "tick_only": reference,
        }
        costs = {
            "ladder": row["vectorised_wall_s"]
            + (row["event_wall_s"] if vec else 0.0)
            + (row["tick_wall_s"] if vec and ev else 0.0),
            "vectorbt_only": row["vectorised_wall_s"],
            "tick_only": row["tick_wall_s"],
        }
        for p, promoted in verdicts.items():
            per_policy[p]["promoted"].append(float(promoted))
            per_policy[p]["false"].append(float(promoted and not reference))
            per_policy[p]["missed"].append(float(reference and not promoted))
            per_policy[p]["cost"].append(costs[p])
    n = len(rows)
    out: dict[str, Any] = {"n_candidates": n, "n_reference_promoted": n_reference}
    for p, d in per_policy.items():
        out[p] = {
            "promoted": wilson_interval(int(sum(d["promoted"])), n, level),
            "false_promotions": wilson_interval(int(sum(d["false"])), n, level),
            "missed": wilson_interval(int(sum(d["missed"])), n, level),
            "total_cost_s": float(sum(d["cost"])),
        }
    abs_err = [abs(float(r["vectorised_sharpe"] or 0) - float(r["tick_sharpe"] or 0)) for r in rows]
    ev_err = [abs(float(r["event_sharpe"] or 0) - float(r["tick_sharpe"] or 0)) for r in rows]
    out["sharpe_abs_error_vectorbt_vs_tick"] = t_interval(abs_err, level)
    out["sharpe_abs_error_event_vs_tick"] = t_interval(ev_err, level)
    for lv in ("vectorised", "event", "tick"):
        bars = s.experiments.n_bars
        out[f"wall_ms_per_1000_bars_{lv}"] = t_interval(
            [1000.0 * float(r[f"{lv}_wall_s"]) * 1000.0 / bars for r in rows], level
        )
    return out


# -- E4 calibrated against uncalibrated fills --------------------------------------------------------


def experiment_calibrated(ctx: SeedContext, strategies: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    orch = ctx.orch
    for strategy in strategies:
        spec = ctx.spec(strategy, tag="calibration-effect")
        bar_bundles = {ctx.symbol: orch.load(spec, ctx.symbol, Fidelity.EVENT)}
        tick_bundles = {ctx.symbol: orch.load(spec, ctx.symbol, Fidelity.TICK)}
        row: dict[str, Any] = {"seed": ctx.seed, "strategy": strategy}
        for fidelity, bundles in ((Fidelity.EVENT, bar_bundles), (Fidelity.TICK, tick_bundles)):
            for calibrated in (True, False):
                run = orch.run_level(spec, fidelity, calibrated=calibrated, bundles=bundles)
                key = f"{fidelity.value}_{'cal' if calibrated else 'uncal'}"
                row[f"{key}_sharpe"] = run.metrics["sharpe"]
                row[f"{key}_n_fills"] = run.metrics["n_fills"]
                row[f"{key}_fees"] = run.metrics["fees"]
                row[f"{key}_total_return"] = run.metrics["total_return"]
                if fidelity is Fidelity.EVENT and calibrated:
                    row["prob_fill_at_touch"] = run.results[0].meta.get("prob_fill_at_touch")
        rows.append(row)
    return rows


# -- E5 parity and injected faults -----------------------------------------------------------------


def _with_fills(result: EngineResult, fills: pd.DataFrame, bar_interval_ns: int) -> EngineResult:
    bars = result.ledger.loc[:, ["ts", "close"]]
    ledger = mark_to_market(fills, bars, bar_interval_ns, result.initial_cash)
    native = dict(result.native)
    native["pnl"] = float(ledger["equity"].iloc[-1] - result.initial_cash)
    return replace(result, engine=f"{result.engine}+fault", fills=fills, ledger=ledger, native=native)


def inject_fault(
    result: EngineResult,
    fault: str,
    *,
    bar_interval_ns: int,
    tick_size: float,
    cfg: ExperimentSettings,
    rng: np.random.Generator,
) -> EngineResult:
    """A copy of ``result`` with one planted bug (see the module docstring)."""
    fills = result.fills.copy()
    n = len(fills)
    pick = rng.random(n) < cfg.fault_share
    if n and not pick.any():
        pick[rng.integers(0, n)] = True
    if fault == "fee_rate":
        fills["fee"] = fills["fee"] + fills["qty"] * fills["price"] * cfg.fault_fee_bps / BPS
    elif fault == "one_bar_delay":
        fills["ts"] = fills["ts"] + bar_interval_ns
    elif fault == "fill_price":
        adverse = fills["side"].to_numpy() * cfg.fault_price_ticks * tick_size
        fills.loc[pick, "price"] = fills.loc[pick, "price"] + adverse[pick]
    elif fault == "missing_fills":
        fills = fills.loc[~pick]
    else:
        raise ValueError(f"unknown fault {fault!r}")
    return _with_fills(result, make_fills(fills.to_dict("records")), bar_interval_ns)


def experiment_parity(ctx: SeedContext, strategies: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    s = ctx.settings
    real: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    inst = ctx.orch.instrument(ctx.symbol)
    for j, strategy in enumerate(strategies):
        spec = ctx.spec(strategy, tag="parity")
        report = run_parity(ctx.orch, spec)[0]
        real.append(
            {
                "seed": ctx.seed,
                "strategy": strategy,
                "n_exec_a": report.n_exec_a,
                "n_exec_b": report.n_exec_b,
                "n_identical": report.n_identical,
                "counts": report.counts,
                "material_rate": report.material_disagreement_rate,
                "pnl_diff_rel": report.pnl_diff_rel,
                "position_mismatch_bars": report.position_mismatch_bars,
                "passed": report.passed,
                "pnl_only_passed": report.pnl_diff_rel <= s.parity.pnl_rel_tol,
            }
        )
        bundles = {ctx.symbol: ctx.orch.load(spec, ctx.symbol, Fidelity.VECTORISED)}
        base = ctx.orch.run_level(
            spec, Fidelity.VECTORISED, calibrated=False, log_trial=False, bundles=bundles
        )
        a = base.results[0]
        for i, fault in enumerate(FAULTS):
            rng = np.random.default_rng([ctx.seed, j, i])
            b = inject_fault(
                a,
                fault,
                bar_interval_ns=s.bar_interval_ns,
                tick_size=inst.tick_size,
                cfg=s.experiments,
                rng=rng,
            )
            rep = compare_runs(a, b, inst, s.parity, strategy)
            faults.append(
                {
                    "seed": ctx.seed,
                    "strategy": strategy,
                    "fault": fault,
                    "kinds": rep.counts,
                    "classifier": not rep.passed,
                    "pnl_only": rep.pnl_diff_rel > s.parity.pnl_rel_tol,
                    "fill_count": len(a.fills) != len(b.fills),
                    "pnl_diff_rel": rep.pnl_diff_rel,
                    "material_rate": rep.material_disagreement_rate,
                }
            )
    return {"real": real, "faults": faults}


# -- aggregation and output ------------------------------------------------------------------------


def _agg(rows: list[dict[str, Any]], keys: Sequence[str], value: str, level: float) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for r in rows:
        v = r.get(value)
        if v is None:
            continue
        groups.setdefault(tuple(r[k] for k in keys), []).append(float(v))
    return {"|".join(map(str, k)): t_interval(v, level) for k, v in groups.items()}


def summarise(raw: dict[str, Any], s: Settings) -> dict[str, Any]:
    level = s.metrics.ci_level
    e1 = raw["fill_models"]
    e1_summary = {k: t_interval([r[k] for r in e1], level) for k in e1[0] if k != "seed"}
    e1_tests = {
        "prob_mae_calibrated_vs_touch": paired_t_test(
            [r["prob_mae_calibrated"] for r in e1], [r["prob_mae_touch_fills"] for r in e1], level
        ),
        "prob_mae_calibrated_vs_fills_only": paired_t_test(
            [r["prob_mae_calibrated"] for r in e1], [r["prob_mae_fills_only"] for r in e1], level
        ),
        "slip_mae_calibrated_vs_half_spread": paired_t_test(
            [r["slip_mae_calibrated"] for r in e1], [r["slip_mae_half_spread"] for r in e1], level
        ),
    }

    ladder = raw["ladder"]
    e2 = {
        "sharpe": _agg(ladder, ("strategy", "fidelity"), "sharpe", level),
        "n_fills": _agg(ladder, ("strategy", "fidelity"), "n_fills", level),
        "sharpe_drop": _agg(ladder, ("strategy", "fidelity"), "sharpe_drop", level),
        "wall_time_s": _agg(ladder, ("strategy", "fidelity"), "wall_time_s", level),
        "drop_ci_excludes_zero": {},
        "status": {},
    }
    for r in ladder:
        key = f"{r['strategy']}|{r['fidelity']}"
        if r["sharpe_drop_low"] is not None:
            hit = r["sharpe_drop_low"] > 0 or r["sharpe_drop_high"] < 0
            e2["drop_ci_excludes_zero"].setdefault(key, [0, 0])
            e2["drop_ci_excludes_zero"][key][0] += int(hit)
            e2["drop_ci_excludes_zero"][key][1] += 1
        e2["status"].setdefault(key, {}).setdefault(r["status"], 0)
        e2["status"][key][r["status"]] += 1

    e3 = {"all": score_policies(raw["screening"], s)}
    for strategy in s.experiments.strategies:
        e3[strategy] = score_policies([r for r in raw["screening"] if r["strategy"] == strategy], s)

    cal = raw["calibrated"]
    e4: dict[str, Any] = {}
    for strategy in s.experiments.strategies:
        rows = [r for r in cal if r["strategy"] == strategy]
        ref = [float(r["tick_cal_sharpe"] or 0) for r in rows]
        err_cal = [abs(float(r["event_cal_sharpe"] or 0) - t) for r, t in zip(rows, ref, strict=True)]
        err_uncal = [abs(float(r["event_uncal_sharpe"] or 0) - t) for r, t in zip(rows, ref, strict=True)]
        e4[strategy] = {
            key: t_interval([r[key] for r in rows], level)
            for key in rows[0]
            if key not in ("seed", "strategy") and rows[0][key] is not None
        }
        e4[strategy]["abs_sharpe_error_event_cal_vs_tick_cal"] = t_interval(err_cal, level)
        e4[strategy]["abs_sharpe_error_event_uncal_vs_tick_cal"] = t_interval(err_uncal, level)
        e4[strategy]["paired_t_error_cal_vs_uncal"] = paired_t_test(err_cal, err_uncal, level)
        e4[strategy]["paired_t_event_sharpe_cal_vs_uncal"] = paired_t_test(
            [r["event_cal_sharpe"] for r in rows], [r["event_uncal_sharpe"] for r in rows], level
        )

    par = raw["parity"]
    e5: dict[str, Any] = {"real": {}, "faults": {}}
    for strategy in s.experiments.strategies:
        rows = [r for r in par["real"] if r["strategy"] == strategy]
        kinds: dict[str, int] = {}
        for r in rows:
            for k, v in r["counts"].items():
                kinds[k] = kinds.get(k, 0) + v
        e5["real"][strategy] = {
            "material_rate": t_interval([r["material_rate"] for r in rows], level),
            "pnl_diff_rel": t_interval([r["pnl_diff_rel"] for r in rows], level),
            "n_exec_a": int(sum(r["n_exec_a"] for r in rows)),
            "n_exec_b": int(sum(r["n_exec_b"] for r in rows)),
            "n_identical": int(sum(r["n_identical"] for r in rows)),
            "kinds": kinds,
            "classifier_passed": wilson_interval(sum(int(r["passed"]) for r in rows), len(rows), level),
            "pnl_only_passed": wilson_interval(
                sum(int(r["pnl_only_passed"]) for r in rows), len(rows), level
            ),
        }
    for fault in FAULTS:
        rows = [r for r in par["faults"] if r["fault"] == fault]
        kinds = {}
        for r in rows:
            for k, v in r["kinds"].items():
                kinds[k] = kinds.get(k, 0) + v
        e5["faults"][fault] = {
            d: wilson_interval(sum(int(r[d]) for r in rows), len(rows), level) for d in DETECTORS
        } | {"kinds": kinds, "pnl_diff_rel": t_interval([r["pnl_diff_rel"] for r in rows], level)}
    return {"E1": {"mean": e1_summary, "tests": e1_tests}, "E2": e2, "E3": e3, "E4": e4, "E5": e5}


def _ci(d: dict[str, float], digits: int = 3) -> str:
    if not d or not math.isfinite(d.get("mean", float("nan"))):
        return "n/a"
    return f"{d['mean']:.{digits}f} ({d['low']:.{digits}f} to {d['high']:.{digits}f})"


def _rate(d: dict[str, float]) -> str:
    return f"{d['k']}/{d['n']} = {d['rate']:.2f} ({d['low']:.2f} to {d['high']:.2f})"


def _test(t: dict[str, float]) -> str:
    return f"t = {t['t']:.2f}, df = {t['df']:.0f}, p = {t['p']:.4f}, critical = {t['critical']:.2f}"


def tables_markdown(summary: dict[str, Any], s: Settings, elapsed_s: float) -> str:
    e = s.experiments
    n = len(e.seeds)
    lines = [
        "# Report tables (generated by make report)",
        "",
        f"Seeds: {', '.join(map(str, e.seeds))} ({n} synthetic markets of {e.n_bars} one-minute bars each). "
        f"Intervals: Student t across seeds, or Wilson score for proportions, at {s.metrics.ci_level:.0%}. "
        f"Run time {elapsed_s:.0f} s.",
        "",
        "## E1 fill models against the generating process (mean over seeds, 95% CI)",
        "",
        "| Quantity | Value |",
        "|---|---|",
    ]
    m = summary["E1"]["mean"]
    for key in m:
        lines.append(f"| {key} | {_ci(m[key], 4)} |")
    lines += ["", "Paired tests across seeds:", ""]
    for key, t in summary["E1"]["tests"].items():
        lines.append(f"{key}: {_test(t)}.")
    lines += [
        "",
        "## E2 fidelity ladder, default parameters (mean over seeds, 95% CI)",
        "",
        "| Strategy and rung | Sharpe | Fills | Sharpe drop vs previous rung | Drop CI excludes 0 "
        "| Wall s | Gate status |",
        "|---|---|---|---|---|---|---|",
    ]
    e2 = summary["E2"]
    for key in e2["sharpe"]:
        drop = e2["sharpe_drop"].get(key, {})
        ex = e2["drop_ci_excludes_zero"].get(key)
        lines.append(
            f"| {key} | {_ci(e2['sharpe'][key], 2)} | {_ci(e2['n_fills'][key], 1)} | {_ci(drop, 2)} "
            f"| {f'{ex[0]}/{ex[1]}' if ex else 'n/a'} | {_ci(e2['wall_time_s'][key], 3)} "
            f"| {', '.join(f'{k}: {v}' for k, v in sorted(e2['status'][key].items()))} |"
        )
    lines += [
        "",
        "## E3 candidate screening: promotion policies against the tick-rung verdict",
        "",
        "| Scope | Policy | Promoted | False promotions | Missed | Engine time s |",
        "|---|---|---|---|---|---|",
    ]
    for scope, res in summary["E3"].items():
        for p in ("ladder", "vectorbt_only", "tick_only"):
            r = res[p]
            lines.append(
                f"| {scope} ({res['n_candidates']} candidates, "
                f"{res['n_reference_promoted']} pass tick) | {p} "
                f"| {_rate(r['promoted'])} | {_rate(r['false_promotions'])} | {_rate(r['missed'])} "
                f"| {r['total_cost_s']:.1f} |"
            )
    a = summary["E3"]["all"]
    lines += [
        "",
        "Absolute Sharpe error against the tick rung: "
        f"vectorbt {_ci(a['sharpe_abs_error_vectorbt_vs_tick'], 2)}, "
        f"Nautilus {_ci(a['sharpe_abs_error_event_vs_tick'], 2)}.",
        f"Engine milliseconds per 1000 bars: vectorbt {_ci(a['wall_ms_per_1000_bars_vectorised'], 2)}, "
        f"Nautilus {_ci(a['wall_ms_per_1000_bars_event'], 2)}, "
        f"hftbacktest {_ci(a['wall_ms_per_1000_bars_tick'], 2)}.",
        "",
        "## E4 calibrated against uncalibrated fills (mean over seeds, 95% CI)",
        "",
    ]
    for strategy, res in summary["E4"].items():
        lines += [f"### {strategy}", "", "| Quantity | Value |", "|---|---|"]
        for key, val in res.items():
            if key.startswith("paired_t"):
                lines.append(f"| {key} | {_test(val)} |")
            else:
                lines.append(f"| {key} | {_ci(val, 3)} |")
        lines.append("")
    lines += [
        "## E5 parity, vectorbt against NautilusTrader (matched costs, touch fills)",
        "",
        "| Strategy | Executions A/B | Identical | Kinds (summed over seeds) | Material rate "
        "| PnL diff / cash "
        "| Classifier passed | PnL-only passed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for strategy, r in summary["E5"]["real"].items():
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(r["kinds"].items())) or "none"
        lines.append(
            f"| {strategy} | {r['n_exec_a']}/{r['n_exec_b']} | {r['n_identical']} | {kinds} "
            f"| {_ci(r['material_rate'], 3)} | {_ci(r['pnl_diff_rel'], 6)} | {_rate(r['classifier_passed'])} "
            f"| {_rate(r['pnl_only_passed'])} |"
        )
    lines += [
        "",
        f"## E5 injected faults: detection rate ({n} seeds x {len(e.strategies)} strategies)",
        "",
        f"Faults: fee rate +{e.fault_fee_bps} bps on every fill; every fill one bar late; "
        f"{e.fault_price_ticks} adverse ticks on {e.fault_share:.0%} of fills; "
        f"{e.fault_share:.0%} of fills dropped.",
        "",
        "| Fault | Classifier | PnL-only | Fill count | Kinds found | PnL diff / cash |",
        "|---|---|---|---|---|---|",
    ]
    for fault, r in summary["E5"]["faults"].items():
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(r["kinds"].items())) or "none"
        lines.append(
            f"| {fault} | {_rate(r['classifier'])} | {_rate(r['pnl_only'])} | {_rate(r['fill_count'])} "
            f"| {kinds} | {_ci(r['pnl_diff_rel'], 6)} |"
        )
    return "\n".join(lines) + "\n"


def warm_up(ctx: SeedContext) -> None:
    """Compile the numba kernels of hftbacktest once, so engine timings exclude compilation."""
    spec = ctx.spec(ctx.settings.experiments.strategies[0], tag="warm-up")
    ctx.orch.run_level(spec, Fidelity.TICK, log_trial=False)


def run_experiments(settings: Settings, progress: Callable[[str], None] | None = None) -> str:
    """Run E1 to E5 for every seed, write the JSON and Markdown outputs, return the tables."""
    started = time.perf_counter()
    e = settings.experiments
    raw: dict[str, Any] = {
        "fill_models": [],
        "ladder": [],
        "screening": [],
        "calibrated": [],
        "parity": {"real": [], "faults": []},
    }
    for i, seed in enumerate(e.seeds):
        ctx = prepare_seed(settings, seed)
        if i == 0:
            warm_up(ctx)
        raw["fill_models"].append({"seed": seed} | experiment_fill_models(ctx))
        raw["ladder"] += experiment_ladder(ctx, e.strategies)
        raw["screening"] += experiment_screening(ctx, e.strategies)
        raw["calibrated"] += experiment_calibrated(ctx, e.strategies)
        par = experiment_parity(ctx, e.strategies)
        raw["parity"]["real"] += par["real"]
        raw["parity"]["faults"] += par["faults"]
        if progress is not None:
            progress(f"seed {seed} done after {time.perf_counter() - started:.0f} s")
    summary = summarise(raw, settings)
    elapsed = time.perf_counter() - started
    out = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "settings": {
            "experiments": e.model_dump(),
            "promotion": {k.value: v.model_dump() for k, v in settings.ladder.promotion.items()},
            "parity": settings.parity.model_dump(),
            "synthetic": settings.synthetic.model_dump(),
            "calibration": settings.calibration.model_dump(),
        },
        "summary": summary,
        "raw": raw,
        "elapsed_s": elapsed,
    }
    (out / "experiments.json").write_text(json.dumps(payload, indent=2, default=str))
    md = tables_markdown(summary, settings, elapsed)
    (out / "tables.md").write_text(md)
    return md + f"\nOutputs in {out}\n"

"""End-to-end runs of the three engines, the ladder, parity, sweeps and walk-forward on a small market."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bt_orchestrator.data.source import LeakageError
from bt_orchestrator.data.synthetic_store import SyntheticPaths
from bt_orchestrator.domain import Fidelity
from bt_orchestrator.engines.base import EngineJob
from bt_orchestrator.engines.nautilus_engine import NautilusEngine
from bt_orchestrator.ladder import run_ladder
from bt_orchestrator.orchestrator import Orchestrator
from bt_orchestrator.parity import run_parity
from bt_orchestrator.reporting import calibration_markdown, ladder_markdown, parity_markdown, write_report
from bt_orchestrator.sweeps import grid_points, random_points, run_sweep, walk_forward
from conftest import run_spec

FAST = {"fast": 10, "slow": 40}


@pytest.mark.parametrize("fidelity", list(Fidelity))
def test_every_rung_runs_sma_cross(orch: Orchestrator, synthetic: SyntheticPaths, fidelity: Fidelity) -> None:
    run = orch.run_level(run_spec(synthetic, "sma_cross", **FAST), fidelity)
    assert run.metrics["n_fills"] > 0
    assert len(run.returns) > 0
    assert run.trial.dataset_fingerprint is not None
    expect_cal = fidelity is not Fidelity.VECTORISED
    assert ("calibrated" in run.trial.tags) is expect_cal


@pytest.mark.parametrize("fidelity", [Fidelity.EVENT, Fidelity.TICK])
def test_limit_orders_on_event_engines(
    orch: Orchestrator, synthetic: SyntheticPaths, fidelity: Fidelity
) -> None:
    spec = run_spec(synthetic, "limit_mean_reversion", entry_z=1.0)
    cal = orch.run_level(spec, fidelity)
    uncal = orch.run_level(spec, fidelity, calibrated=False, log_trial=False)
    assert cal.metrics["n_fills"] > 0
    assert uncal.metrics["n_fills"] > 0
    meta = cal.results[0].meta
    assert meta["fill_model"].startswith("calibrated")


def test_nautilus_trades_mode(orch: Orchestrator, synthetic: SyntheticPaths) -> None:
    s = orch.settings
    spec = run_spec(synthetic, "sma_cross", **FAST)
    bundle = orch.load(spec, "BTC-USDT-PERP", Fidelity.TICK)
    strategy = orch.registry.get("sma_cross")
    engine = NautilusEngine(s.engines.nautilus.model_copy(update={"data_mode": "trades"}), 10.0, 86400.0)
    job = EngineJob(
        strategy=strategy,
        params=orch.params(strategy, FAST),
        instrument=orch.instrument("BTC-USDT-PERP"),
        bars=bundle.bars,
        bar_interval_ns=s.bar_interval_ns,
        costs=orch.costs(),
        initial_cash=s.account.initial_cash,
        seed=1,
        tape=bundle.tape,
    )
    result = engine.run(job)
    assert result.meta["data_mode"] == "trades"
    with pytest.raises(ValueError, match="tick tape"):
        engine.run(replace(job, tape=None))
    with pytest.raises(ValueError, match="tick tape"):
        orch.engines.tick().run(replace(job, tape=None))


def test_vectorised_rung_refuses_event_only_and_leakage(
    orch: Orchestrator, synthetic: SyntheticPaths
) -> None:
    spec = run_spec(synthetic, "sma_cross", **FAST)
    with pytest.raises(LeakageError):
        orch.run_level(replace(spec, known_at=synthetic.start), Fidelity.VECTORISED)


def test_ladder_reports_every_rung(orch: Orchestrator, synthetic: SyntheticPaths, tmp_path: object) -> None:
    report, runs = run_ladder(orch, run_spec(synthetic, "sma_cross", **FAST), run_all_levels=True)
    assert [lv.fidelity for lv in report.levels] == list(Fidelity)
    assert len(runs) == 4
    assert report.levels[1].drop_method == "paired"
    assert report.levels[3].drop_method in ("paired", "independent")
    assert report.top_run() is not None
    md = ladder_markdown(report)
    assert "| tick |" in md
    blocked, _ = run_ladder(
        orch, run_spec(synthetic, "limit_mean_reversion"), top=Fidelity.EVENT, run_all_levels=False
    )
    assert len(blocked.levels) == 2
    if blocked.levels[0].status == "failed":
        assert blocked.levels[1].status == "not_run"
        assert "Gate notes" in ladder_markdown(blocked)


def test_parity_vectorbt_against_nautilus(orch: Orchestrator, synthetic: SyntheticPaths) -> None:
    reports = run_parity(orch, run_spec(synthetic, "sma_cross", **FAST))
    assert len(reports) == 1
    r = reports[0]
    assert r.engine_a == "vectorbt"
    assert r.engine_b == "nautilus"
    assert r.material_disagreement_rate == 0.0
    assert r.passed
    assert "vectorbt vs nautilus" in parity_markdown(reports)
    with pytest.raises(ValueError, match="bar-level"):
        run_parity(orch, run_spec(synthetic, "sma_cross"), Fidelity.TICK)


def test_sweeps_and_walk_forward(orch: Orchestrator, synthetic: SyntheticPaths) -> None:
    s = orch.settings
    spec = run_spec(synthetic, "sma_cross")
    points = grid_points({"fast": [5, 10, 50], "slow": [20, 40], "qty": [1.0]})
    sweep = run_sweep(orch, spec, points, Fidelity.VECTORISED)
    assert sweep.n_invalid == 2
    assert len(sweep.trials) == 4
    assert sweep.best().params["fast"] in (5, 10)
    rnd = random_points(
        s.strategy_settings("sma_cross").ranges, orch.registry.get("sma_cross").params_model, 2, 1
    )
    assert len(run_sweep(orch, spec, rnd, Fidelity.VECTORISED).trials) == 2
    wf = walk_forward(orch, spec, points)
    assert len(wf.windows) == (360 - 180) // 60 + 1
    assert wf.n_trials == len(wf.windows) * 5
    assert wf.oos_sharpe.low <= wf.oos_sharpe.high


def test_report_writers(orch: Orchestrator, tmp_path: object) -> None:
    from pathlib import Path

    bundle = orch.models.load(orch.settings.market.venue, "BTC-USDT-PERP")
    assert bundle is not None
    md = calibration_markdown(bundle)
    assert "Brier" in md
    out = write_report(Path(str(tmp_path)), "x", {"a": 1}, md)
    assert out.read_text() == md

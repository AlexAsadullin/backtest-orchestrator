"""``make reproduce``: synthetic data -> calibration -> ladders -> parity -> sweep -> report."""

from __future__ import annotations

import json

from bt_orchestrator.contracts.ledger import Actor
from bt_orchestrator.data.synthetic_store import write_synthetic
from bt_orchestrator.ladder import run_ladder
from bt_orchestrator.orchestrator import RunSpec
from bt_orchestrator.parity import run_parity
from bt_orchestrator.reporting import (
    calibration_markdown,
    ladder_markdown,
    parity_markdown,
    write_report,
)
from bt_orchestrator.settings import Settings
from bt_orchestrator.sweeps import grid_points, run_sweep, walk_forward
from bt_orchestrator.wiring import build_orchestrator


def reproduce(settings: Settings) -> str:
    """Run the offline demo; returns the summary (also written to ``reports/reproduce/summary.md``)."""
    s = settings
    universe = list(s.market.default_universe)
    paths = write_synthetic(s, universe)
    orch = build_orchestrator(s)
    sections: list[str] = ["# bt-orchestrator reproduce summary", ""]

    sections += ["## Calibrated fill models", ""]
    for symbol in universe:
        bundle = orch.calibrate(symbol, fills_root=paths.fills_root, orders_root=paths.orders_root)
        sections.append(calibration_markdown(bundle))

    def spec(strategy: str, tag: str) -> RunSpec:
        return RunSpec(
            strategy=strategy,
            params={},
            universe=universe,
            start=paths.start,
            end=paths.end,
            known_at=paths.end,
            seed=s.metrics.seed,
            actor=Actor(kind="robot", id="bt-reproduce"),
            tags=("reproduce", tag),
        )

    sections += ["## Fidelity ladders", ""]
    for strategy in s.reproduce.strategies:
        report, _ = run_ladder(orch, spec(strategy, "ladder"), run_all_levels=s.reproduce.run_all_levels)
        md = ladder_markdown(report)
        write_report(s.reports_dir / "reproduce", f"ladder-{strategy}", report.model_dump(mode="json"), md)
        sections.append(md)

    sections += [
        "## Cross-engine parity (vectorbt vs NautilusTrader, matched costs, touch fills)",
        "",
    ]
    parity_reports = []
    for strategy in s.reproduce.parity_strategies:
        parity_reports += run_parity(orch, spec(strategy, "parity"))
    md = parity_markdown(parity_reports)
    write_report(
        s.reports_dir / "reproduce",
        "parity",
        [r.model_dump(mode="json") for r in parity_reports],
        md,
    )
    sections.append(md)

    sweep_strategy = s.reproduce.sweep_strategy
    points = grid_points(s.strategy_settings(sweep_strategy).grid)
    sweep = run_sweep(orch, spec(sweep_strategy, "sweep"), points, s.walk_forward.fidelity)
    best = sweep.best()
    wf = walk_forward(orch, spec(sweep_strategy, "walk-forward"), points)
    ci = wf.oos_sharpe
    sections += [
        "## Sweep and walk-forward",
        "",
        f"Grid sweep of {sweep_strategy}: {len(sweep.trials)} trials ({sweep.n_invalid} invalid points "
        f"skipped); best in-sample Sharpe {float(best.metrics['sharpe'] or 0):.2f} with {best.params}.",
        f"Walk-forward: {len(wf.windows)} windows, {wf.n_trials} trials; stitched out-of-sample Sharpe "
        f"{ci.estimate:.2f} (95% CI {ci.low:.2f} to {ci.high:.2f}).",
        "",
        f"Trials were logged through the ledger client (last sink: {orch.ledger.last_sink}).",
        "",
    ]
    summary = "\n".join(sections)
    out = s.reports_dir / "reproduce"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary)
    (out / "walk_forward.json").write_text(
        json.dumps({"windows": [w.__dict__ for w in wf.windows], "oos_sharpe": ci.as_dict()}, indent=2)
    )
    return summary + f"\nReports written to {out}\n"

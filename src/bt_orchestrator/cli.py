"""Command line: ``bt run | ladder | parity | calibrate | sweep | walk-forward | synth | serve | reproduce | report``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
import yaml

from bt_orchestrator.api import create_app
from bt_orchestrator.contracts.ledger import Actor
from bt_orchestrator.data.synthetic import MarketSpec
from bt_orchestrator.data.synthetic_store import synthetic_fill_roots, write_synthetic
from bt_orchestrator.domain import Fidelity
from bt_orchestrator.ladder import run_ladder
from bt_orchestrator.orchestrator import Orchestrator, RunSpec
from bt_orchestrator.parity import run_parity
from bt_orchestrator.reporting import (
    calibration_markdown,
    ladder_markdown,
    parity_markdown,
    write_report,
)
from bt_orchestrator.settings import Settings, load_settings
from bt_orchestrator.sweeps import grid_points, random_points, run_sweep, walk_forward
from bt_orchestrator.timeutil import ns_to_iso
from bt_orchestrator.wiring import build_orchestrator, configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

Strategy = Annotated[str, typer.Option(help="Registered strategy name.")]
Params = Annotated[list[str] | None, typer.Option("--param", "-p", help="key=value (YAML value).")]
Universe = Annotated[list[str] | None, typer.Option("--symbol", "-s", help="Symbol; repeatable.")]
Start = Annotated[str | None, typer.Option(help="ISO-8601 start (default: synthetic window).")]
End = Annotated[str | None, typer.Option(help="ISO-8601 end, exclusive.")]
KnownAt = Annotated[str | None, typer.Option(help="Knowledge time; default = end.")]
Seed = Annotated[int | None, typer.Option(help="Seed for stochastic fill models.")]


def _context() -> tuple[Settings, Orchestrator]:
    settings = load_settings()
    configure_logging(settings)
    return settings, build_orchestrator(settings)


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"expected key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def _spec(
    settings: Settings,
    strategy: str,
    params: list[str] | None,
    universe: list[str] | None,
    start: str | None,
    end: str | None,
    known_at: str | None,
    seed: int | None,
) -> RunSpec:
    spec0 = MarketSpec.from_settings(
        settings.synthetic, settings.market.default_universe[0], settings.market.venue,
        settings.bar_interval_ns,
    )
    start = start or ns_to_iso(spec0.start_ns)
    end = end or ns_to_iso(spec0.end_ns)
    return RunSpec(
        strategy=strategy,
        params=_parse_params(params),
        universe=universe or list(settings.market.default_universe),
        start=start,
        end=end,
        known_at=known_at or end,
        seed=seed if seed is not None else settings.metrics.seed,
        actor=Actor(kind=settings.ledger.actor.kind, id=settings.ledger.actor.id),
        tags=("cli",),
    )


def _echo(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def run(
    strategy: Strategy,
    fidelity: Annotated[Fidelity, typer.Option(help="Fidelity level.")] = Fidelity.VECTORISED,
    param: Params = None,
    symbol: Universe = None,
    start: Start = None,
    end: End = None,
    known_at: KnownAt = None,
    seed: Seed = None,
) -> None:
    """Run one strategy at one fidelity and log the trial."""
    settings, orch = _context()
    result = orch.run_level(_spec(settings, strategy, param, symbol, start, end, known_at, seed), fidelity)
    _echo({"trial_id": result.trial.trial_id, "engine": result.engine, "metrics": result.metrics})


@app.command()
def ladder(
    strategy: Strategy,
    top: Annotated[Fidelity | None, typer.Option(help="Highest level to climb to.")] = None,
    all_levels: Annotated[bool, typer.Option(help="Keep climbing after a failed gate.")] = False,
    param: Params = None,
    symbol: Universe = None,
    start: Start = None,
    end: End = None,
    known_at: KnownAt = None,
    seed: Seed = None,
) -> None:
    """Climb the fidelity ladder with promotion gates and write the report."""
    settings, orch = _context()
    spec = _spec(settings, strategy, param, symbol, start, end, known_at, seed)
    report, _ = run_ladder(orch, spec, top, run_all_levels=all_levels or None)
    path = write_report(
        settings.reports_dir / f"ladder-{report.run_id}",
        "ladder",
        report.model_dump(mode="json"),
        ladder_markdown(report),
    )
    typer.echo(ladder_markdown(report))
    typer.echo(f"report: {path}")


@app.command()
def parity(
    strategy: Strategy,
    param: Params = None,
    symbol: Universe = None,
    start: Start = None,
    end: End = None,
    known_at: KnownAt = None,
    seed: Seed = None,
) -> None:
    """Compare vectorbt and NautilusTrader on the same strategy with matched settings."""
    settings, orch = _context()
    reports = run_parity(orch, _spec(settings, strategy, param, symbol, start, end, known_at, seed))
    md = parity_markdown(reports)
    path = write_report(
        settings.reports_dir / f"parity-{strategy}",
        "parity",
        [r.model_dump(mode="json") for r in reports],
        md,
    )
    typer.echo(md)
    typer.echo(f"report: {path}")


@app.command()
def calibrate(
    symbol: Annotated[str | None, typer.Option(help="Symbol (default: first in universe).")] = None,
    fills_root: Annotated[Path | None, typer.Option(help="Fills export root (default: I6).")] = None,
    orders_root: Annotated[Path | None, typer.Option(help="Orders extension root.")] = None,
    synthetic: Annotated[bool, typer.Option(help="Use the synthetic fills written by `bt synth`.")] = False,
    orders: Annotated[bool, typer.Option(help="Use the orders extension if present.")] = True,
) -> None:
    """Fit and version fill-probability, slippage and latency models from fills."""
    settings, orch = _context()
    if synthetic:
        fills_root, orders_root = synthetic_fill_roots(settings)
    sym = symbol or settings.market.default_universe[0]
    bundle = orch.calibrate(sym, fills_root=fills_root, orders_root=orders_root, with_orders=orders)
    typer.echo(calibration_markdown(bundle))


@app.command()
def sweep(
    strategy: Strategy,
    mode: Annotated[str, typer.Option(help="grid | random")] = "grid",
    fidelity: Annotated[Fidelity, typer.Option()] = Fidelity.VECTORISED,
    symbol: Universe = None,
    start: Start = None,
    end: End = None,
    known_at: KnownAt = None,
) -> None:
    """Sweep the configured grid or random ranges; every point is a logged trial."""
    settings, orch = _context()
    st = settings.strategy_settings(strategy)
    model = orch.registry.get(strategy).params_model
    points = (
        grid_points(st.grid)
        if mode == "grid"
        else random_points(st.ranges, model, settings.sweep.random_samples, settings.sweep.seed)
    )
    spec = _spec(settings, strategy, None, symbol, start, end, known_at, None)
    result = run_sweep(orch, spec, points, fidelity)
    best = result.best()
    _echo({"n_trials": len(result.trials), "n_invalid": result.n_invalid, "best": best.params,
           "best_metrics": best.metrics})


@app.command("walk-forward")
def walk_forward_cmd(
    strategy: Strategy,
    symbol: Universe = None,
    start: Start = None,
    end: End = None,
    known_at: KnownAt = None,
) -> None:
    """Rolling in-sample grid search with out-of-sample evaluation."""
    settings, orch = _context()
    points = grid_points(settings.strategy_settings(strategy).grid)
    result = walk_forward(orch, _spec(settings, strategy, None, symbol, start, end, known_at, None), points)
    _echo({"windows": [w.__dict__ for w in result.windows], "oos_sharpe": result.oos_sharpe.as_dict(),
           "n_trials": result.n_trials})


@app.command()
def synth(
    n_bars: Annotated[int | None, typer.Option(help="Bars per symbol (default from settings).")] = None,
) -> None:
    """Write synthetic bars, tapes and fills under the data subtree."""
    settings = load_settings()
    configure_logging(settings)
    paths = write_synthetic(settings, list(settings.market.default_universe), n_bars)
    _echo({"datasets": paths.datasets, "fills": paths.fills_root, "start": paths.start, "end": paths.end})


@app.command()
def serve() -> None:
    """Serve ``POST /v1/backtests`` on the configured host and port."""
    settings, orch = _context()
    uvicorn.run(create_app(orch), host=settings.service.host, port=settings.service.port)


@app.command()
def reproduce() -> None:
    """Offline end-to-end demo on synthetic data (see README)."""
    from bt_orchestrator.reproduce import reproduce as run_reproduce

    settings, _ = _context()
    typer.echo(run_reproduce(settings))


@app.command()
def report() -> None:
    """Run the experiments behind docs/report.md and write their tables."""
    from bt_orchestrator.experiments import run_experiments

    settings, _ = _context()
    typer.echo(run_experiments(settings))

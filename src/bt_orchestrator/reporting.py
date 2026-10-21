"""Markdown and JSON reports written under ``$TRADEQ_DATA_DIR/backtest-orchestrator/reports``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bt_orchestrator.calibration.models import FillModelBundle
from bt_orchestrator.ladder import LadderReport
from bt_orchestrator.parity import ParityReport


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _ci(ci: dict[str, float] | None, digits: int = 2) -> str:
    if not ci:
        return "-"
    return f"{ci['estimate']:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"


def ladder_markdown(report: LadderReport) -> str:
    lines = [
        f"### Fidelity ladder: {report.strategy} on {', '.join(report.universe)}",
        "",
        f"Run `{report.run_id}`, {report.start} to {report.end}, known at {report.known_at}. "
        f"Promoted to: **{report.promoted_to.value if report.promoted_to else 'none'}**.",
        "",
        "| Level | Engine | Status | Sharpe [95% CI] | Sharpe drop vs previous [CI] | Total return [CI] "
        "| Max DD | Fills | Fees | Wall s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lv in report.levels:
        m = lv.metrics
        lines.append(
            f"| {lv.fidelity.value} | {lv.engine or '-'} | {lv.status} | {_ci(lv.sharpe_ci)} "
            f"| {_ci(lv.sharpe_drop_ci)} | {_ci(lv.total_return_ci, 4)} | {_fmt(m.get('max_drawdown'))} "
            f"| {_fmt(m.get('n_fills'))} | {_fmt(m.get('fees'), 1)} | {_fmt(lv.wall_time_s, 2)} |"
        )
    failed = [lv for lv in report.levels if lv.reasons]
    if failed:
        lines += ["", "Gate notes:"]
        lines += [f"- {lv.fidelity.value}: {'; '.join(lv.reasons)}" for lv in failed]
    return "\n".join(lines) + "\n"


def parity_markdown(reports: list[ParityReport]) -> str:
    lines = [
        "| Strategy | Symbol | Engines | Executions A/B | Identical | Disagreements by kind "
        "| Material rate | PnL A | PnL B | PnL diff / cash | Passed |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(r.counts.items())) or "none"
        lines.append(
            f"| {r.strategy} | {r.symbol} | {r.engine_a} vs {r.engine_b} | {r.n_exec_a}/{r.n_exec_b} "
            f"| {r.n_identical} | {kinds} | {r.material_disagreement_rate:.3f} | {r.pnl_a:.2f} "
            f"| {r.pnl_b:.2f} | {r.pnl_diff_rel:.2e} | {r.passed} |"
        )
    return "\n".join(lines) + "\n"


def calibration_markdown(bundle: FillModelBundle) -> str:
    e = bundle.evaluation
    return "\n".join(
        [
            f"Fill model v{bundle.version} for {bundle.venue}/{bundle.symbol} "
            f"({bundle.label_basis}, {bundle.n_limit_orders} limit orders, "
            f"{bundle.n_taker_fills} taker fills).",
            "",
            "| Held-out metric | Calibrated | Naive |",
            "|---|---|---|",
            f"| Fill probability Brier score | {e['brier']:.4f} | touch fills {e['brier_touch_fills']:.4f}, "
            f"base rate {e['brier_base_rate']:.4f} |",
            f"| Fill probability AUC | {e.get('auc', float('nan')):.3f} | 0.5 |",
            f"| Slippage MAE (bps) | {e['slippage_mae_bps']:.3f} | zero {e['slippage_mae_zero_bps']:.3f}, "
            f"half spread {e['slippage_mae_half_spread_bps']:.3f} |",
            "",
        ]
    )


def write_report(out_dir: Path, name: str, payload: Any, markdown: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    md = out_dir / f"{name}.md"
    md.write_text(markdown)
    return md

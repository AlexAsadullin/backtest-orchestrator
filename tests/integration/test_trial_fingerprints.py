"""Trials keep the point-in-time source's own fingerprint, also when bars are built from trades."""

from __future__ import annotations

from pathlib import Path

from bt_orchestrator.data.synthetic_store import write_synthetic
from bt_orchestrator.domain import Fidelity
from bt_orchestrator.wiring import build_orchestrator
from conftest import make_settings, run_spec


def test_trial_fingerprint_is_the_primary_dataset_fingerprint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data", pit={"bars_source": "trades"}, synthetic={"n_bars": 120})
    paths = write_synthetic(settings, ["BTC-USDT-PERP"])
    orch = build_orchestrator(settings)
    spec = run_spec(paths, "sma_cross", fast=5, slow=20)

    bundle = orch.load(spec, "BTC-USDT-PERP", Fidelity.VECTORISED)
    assert list(bundle.fingerprints) == ["trades"]
    assert len(bundle.bars) == 120

    run = orch.run_level(spec, Fidelity.VECTORISED, log_trial=False)
    assert run.trial.dataset_fingerprint == bundle.fingerprints["trades"]
    assert f"fingerprint:BTC-USDT-PERP:trades={bundle.fingerprints['trades']}" in run.trial.tags

    tick = orch.run_level(spec, Fidelity.TICK, log_trial=False)
    assert tick.trial.dataset_fingerprint == bundle.fingerprints["trades"]
    assert sum(t.startswith("fingerprint:BTC-USDT-PERP:") for t in tick.trial.tags) == 2

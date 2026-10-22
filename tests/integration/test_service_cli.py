"""The HTTP service, the CLI, make reproduce and make report on shrunken settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from bt_orchestrator import cli
from bt_orchestrator.api import create_app
from bt_orchestrator.data.synthetic_store import SyntheticPaths
from bt_orchestrator.experiments import run_experiments
from bt_orchestrator.orchestrator import Orchestrator
from bt_orchestrator.reproduce import reproduce
from bt_orchestrator.settings import Settings
from conftest import make_settings


def body(paths: SyntheticPaths, **kw: object) -> dict[str, object]:
    return {
        "strategy": "sma_cross",
        "params": {"fast": 10, "slow": 40},
        "universe": ["BTC-USDT-PERP"],
        "start": paths.start,
        "end": paths.end,
        "known_at": paths.end,
        "fidelity": "vectorised",
    } | kw


def test_api(orch: Orchestrator, synthetic: SyntheticPaths) -> None:
    client = TestClient(create_app(orch))
    assert client.get("/healthz").json() == {"status": "ok"}
    names = [s["name"] for s in client.get("/v1/strategies").json()["strategies"]]
    assert names == ["limit_mean_reversion", "sma_cross"]
    r = client.post("/v1/backtests", json=body(synthetic))
    assert r.status_code == 200
    assert r.json()["fidelity_report"] is None
    r = client.post("/v1/backtests", json=body(synthetic, fidelity="event", ladder=True, seed=3))
    assert r.status_code == 200
    assert len(r.json()["fidelity_report"]["levels"]) == 2
    r = client.post("/v1/backtests", json=body(synthetic, known_at=synthetic.start))
    assert r.status_code == 409
    assert r.json()["kind"] == "end_after_known_at"
    r = client.post("/v1/backtests", json=body(synthetic, strategy="nope"))
    assert r.status_code == 422
    r = client.post("/v1/backtests", json=body(synthetic, params={"fast": 50, "slow": 10}))
    assert r.status_code == 422


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    env = {
        "TRADEQ_DATA_DIR": str(data),
        "BT_LEDGER__MODE": "inbox",
        "BT_SYNTHETIC__N_BARS": "240",
        "BT_SYNTHETIC__FILLS__N_ORDERS": "1200",
        "BT_METRICS__N_BOOT": "100",
        "BT_LOGGING__LEVEL": "WARNING",
        "BT_CODE_VERSION": "test",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return make_settings(data, synthetic={"n_bars": 240, "fills": {"n_orders": 1200}})


def test_cli_commands(cli_env: Settings) -> None:
    runner = CliRunner()

    def ok(*args: str) -> str:
        result = runner.invoke(cli.app, list(args))
        assert result.exit_code == 0, result.output
        return result.output

    synth = json.loads(ok("synth"))
    assert synth["start"] == "2026-01-05T00:00:00Z"
    assert "Fill model v1" in ok("calibrate", "--synthetic")
    run = json.loads(ok("run", "--strategy", "sma_cross", "-p", "fast=10", "-p", "slow=40"))
    assert run["engine"] == "vectorbt"
    assert "Fidelity ladder" in ok(
        "ladder", "--strategy", "sma_cross", "--top", "event", "-p", "fast=10", "-p", "slow=40"
    )
    assert "vectorbt vs nautilus" in ok("parity", "--strategy", "sma_cross", "-p", "fast=10", "-p", "slow=40")
    sweep = json.loads(ok("sweep", "--strategy", "sma_cross", "--mode", "random"))
    assert sweep["n_trials"] + sweep["n_invalid"] == cli_env.sweep.random_samples
    bad = runner.invoke(cli.app, ["run", "--strategy", "sma_cross", "-p", "fast"])
    assert bad.exit_code != 0


def test_reproduce_end_to_end(cli_env: Settings) -> None:
    summary = reproduce(
        cli_env.model_copy(
            update={
                "walk_forward": cli_env.walk_forward.model_copy(
                    update={"train_bars": 120, "test_bars": 60, "step_bars": 60}
                )
            }
        )
    )
    assert "Fidelity ladder: sma_cross" in summary
    assert "Walk-forward" in summary
    assert (cli_env.reports_dir / "reproduce" / "summary.md").exists()


def test_report_experiments_shrunken(cli_env: Settings) -> None:
    s = cli_env.model_copy(
        update={
            "experiments": cli_env.experiments.model_copy(update={"seeds": [1, 2], "n_bars": 240}),
            "strategies": {
                "sma_cross": cli_env.strategies["sma_cross"].model_copy(
                    update={"grid": {"fast": [10], "slow": [40], "qty": [1.0]}}
                ),
                "limit_mean_reversion": cli_env.strategies["limit_mean_reversion"].model_copy(
                    update={
                        "grid": {
                            "lookback": [20],
                            "entry_z": [1.5, 2.5],
                            "exit_z": [0.25],
                            "offset_bps": [1.0],
                            "qty": [1.0],
                            "order_ttl_bars": [1],
                        }
                    }
                ),
            },
        }
    )
    messages: list[str] = []
    md = run_experiments(s, progress=messages.append)
    assert len(messages) == 2
    assert "## E5 injected faults" in md
    payload = json.loads((s.reports_dir / "experiments.json").read_text())
    faults = payload["summary"]["E5"]["faults"]
    assert faults["fee_rate"]["classifier"]["rate"] == 1.0
    assert faults["one_bar_delay"]["classifier"]["rate"] == 1.0
    assert payload["summary"]["E3"]["all"]["n_candidates"] == 2 * 3

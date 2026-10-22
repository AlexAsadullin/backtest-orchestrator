"""Opt-in integration test against the shipped I2 (point-in-time-store) and S6 (research-ledger).

Run with ``make integration`` (``pytest -m integration``). The test starts both services from their own
virtual environments on free loopback ports, with a scratch ``TRADEQ_DATA_DIR`` under this repository's
gitignored ``tmp/`` folder (never the shared data root). It writes a synthetic tape in the I1 layout,
ingests it into I2 with ``pit ingest``, climbs the first two fidelity levels with bars that I2 serves from
``tape.trades``, and then checks in S6's DuckDB file that both trials arrived with I2's fingerprint.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import pytest

from bt_orchestrator.contracts.ledger import Actor
from bt_orchestrator.data.source import LeakageError
from bt_orchestrator.data.synthetic import MarketSpec, generate_tape
from bt_orchestrator.data.tape import write_i1_tape
from bt_orchestrator.domain import Fidelity
from bt_orchestrator.ladder import run_ladder
from bt_orchestrator.orchestrator import RunSpec, instrument_spec
from bt_orchestrator.settings import REPO_ROOT
from bt_orchestrator.timeutil import ns_to_iso
from bt_orchestrator.wiring import build_orchestrator
from conftest import make_settings

pytestmark = pytest.mark.integration

WORKSPACE = REPO_ROOT.parent
PIT_REPO = WORKSPACE / "point-in-time-store"
LEDGER_REPO = WORKSPACE / "research-ledger"
PIT_BIN = PIT_REPO / ".venv" / "bin" / "pit"
LEDGER_BIN = LEDGER_REPO / ".venv" / "bin" / "ledger"
LEDGER_PYTHON = LEDGER_REPO / ".venv" / "bin" / "python"
SYMBOL = "BTC-USDT-PERP"
N_BARS = 240
STARTUP_TIMEOUT_S = 60.0
LOOPBACK = "127.0.0.1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return int(sock.getsockname()[1])


def _wait_healthy(url: str, proc: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"service exited early; log:\n{log.read_text()}")
        try:
            if httpx.get(f"{url}/healthz", timeout=1.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"{url} did not become healthy; log:\n{log.read_text()}")


@dataclass
class Services:
    data_dir: Path
    pit_url: str
    ledger_url: str
    env: dict[str, str]
    procs: dict[str, subprocess.Popen[bytes]]

    def stop(self, name: str) -> None:
        proc = self.procs.pop(name)
        proc.terminate()
        proc.wait(timeout=30)


@pytest.fixture
def services() -> Iterator[Services]:
    for binary in (PIT_BIN, LEDGER_BIN, LEDGER_PYTHON):
        if not binary.exists():
            pytest.skip(f"sibling service not installed: {binary}")
    data_dir = REPO_ROOT / "tmp" / f"integration-{uuid.uuid4().hex[:8]}"
    data_dir.mkdir(parents=True)
    pit_port, ledger_port = _free_port(), _free_port()
    ledger_url = f"http://{LOOPBACK}:{ledger_port}"
    env = {
        **os.environ,
        "TRADEQ_DATA_DIR": str(data_dir),
        "LEDGER_SERVICE__HOST": LOOPBACK,
        "LEDGER_SERVICE__PORT": str(ledger_port),
        "PIT_SERVICE__HOST": LOOPBACK,
        "PIT_SERVICE__PORT": str(pit_port),
        "PIT_LEDGER__BASE_URL": ledger_url,
    }
    procs: dict[str, subprocess.Popen[bytes]] = {}
    svc = Services(data_dir, f"http://{LOOPBACK}:{pit_port}", ledger_url, env, procs)
    try:
        for name, cmd, url in (
            ("ledger", [str(LEDGER_BIN), "serve"], svc.ledger_url),
            ("pit", [str(PIT_BIN), "serve"], svc.pit_url),
        ):
            log = data_dir / f"{name}.log"
            with log.open("wb") as fh:
                procs[name] = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            _wait_healthy(url, procs[name], log)
        yield svc
    finally:
        for name in list(procs):
            svc.stop(name)
    shutil.rmtree(data_dir, ignore_errors=True)


def _seed_i2(svc: Services) -> MarketSpec:
    """Write a synthetic I1 tape under the scratch root and ingest it with I2's own CLI."""
    settings = make_settings(svc.data_dir)
    spec = MarketSpec.from_settings(
        settings.synthetic, SYMBOL, settings.market.venue, settings.bar_interval_ns
    )
    spec = replace(spec, n_bars=N_BARS)
    tape = generate_tape(spec, instrument_spec(settings, SYMBOL))
    write_i1_tape(tape, svc.data_dir / "market-tape-recorder" / "tape", settings.market.venue)
    for dataset in ("tape.trades", "tape.book_delta"):
        out = subprocess.run(
            [str(PIT_BIN), "ingest", dataset, "--venue", settings.market.venue],
            env=svc.env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.returncode == 0, out.stderr
    return spec


def _ledger_rows(svc: Services) -> list[tuple[str, str | None, str]]:
    """Read S6's trials table with S6's own interpreter (DuckDB is its dependency, not ours)."""
    db = svc.data_dir / "research-ledger" / "ledger.duckdb"
    script = (
        "import duckdb, json, sys\n"
        "con = duckdb.connect(sys.argv[1], read_only=True)\n"
        "rows = con.execute('SELECT trial_id, dataset_fingerprint, source FROM trials').fetchall()\n"
        "print(json.dumps(rows))\n"
    )
    out = subprocess.run(
        [str(LEDGER_PYTHON), "-c", script, str(db)], capture_output=True, text=True, check=True
    )
    return [tuple(r) for r in json.loads(out.stdout)]


def test_ladder_against_real_i2_and_s6(services: Services) -> None:
    spec = _seed_i2(services)
    settings = make_settings(
        services.data_dir,
        pit={"mode": "http", "base_url": services.pit_url, "bars_source": "trades"},
        ledger={"mode": "http", "base_url": services.ledger_url},
        synthetic={"n_bars": N_BARS},
    )
    orch = build_orchestrator(settings)
    start, end = ns_to_iso(spec.start_ns), ns_to_iso(spec.end_ns)
    run = RunSpec(
        strategy="sma_cross",
        params={"fast": 5, "slow": 20},
        universe=[SYMBOL],
        start=start,
        end=end,
        known_at=end,
        seed=7,
        actor=Actor(kind="robot", id="pytest-integration"),
        tags=("integration",),
    )

    bundle = orch.load(run, SYMBOL, Fidelity.EVENT)
    fingerprint = bundle.fingerprints["trades"]
    assert fingerprint.startswith("sha256:")
    assert len(bundle.bars) > N_BARS // 2

    report, runs = run_ladder(orch, run, top=Fidelity.EVENT, run_all_levels=True)
    assert [lv.fidelity for lv in report.levels] == [Fidelity.VECTORISED, Fidelity.EVENT]
    assert orch.ledger.last_sink == "http"
    trial_ids = {r.trial.trial_id for r in runs}
    assert all(r.trial.dataset_fingerprint == fingerprint for r in runs)

    # I2 can resolve the fingerprint the trials carry, so the backtest is reproducible.
    resolved = httpx.get(f"{services.pit_url}/v1/fingerprints/{fingerprint}", timeout=10.0)
    assert resolved.status_code == 200

    # A research query that reaches past known_at is refused by I2 with a LeakageEvent body.
    with pytest.raises(LeakageError) as leak:
        orch.load(replace(run, known_at=start), SYMBOL, Fidelity.VECTORISED)
    assert leak.value.reported
    assert leak.value.event.source == "i2"

    services.stop("pit")
    services.stop("ledger")
    rows = {trial_id: (fp, source) for trial_id, fp, source in _ledger_rows(services)}
    for trial_id in trial_ids:
        assert rows[trial_id] == (fingerprint, "i5")

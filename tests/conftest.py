"""Shared fixtures: settings on a temporary data root, a small synthetic market and an orchestrator.

Every test runs offline. The ledger writes to a private inbox, the point-in-time source reads local
Parquet files and the S3 simulator is the in-process fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bt_orchestrator.contracts.ledger import Actor
from bt_orchestrator.data.synthetic_store import SyntheticPaths, write_synthetic
from bt_orchestrator.orchestrator import Orchestrator, RunSpec
from bt_orchestrator.settings import Settings, load_settings
from bt_orchestrator.wiring import build_orchestrator

SMALL_BARS = 360


def make_settings(data_dir: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "TRADEQ_DATA_DIR": str(data_dir),
        "code_version": "test",
        "ledger": {"mode": "inbox"},
        "pit": {"mode": "local"},
        "sim": {"mode": "fake"},
        "synthetic": {"n_bars": SMALL_BARS, "fills": {"n_orders": 1500}},
        "metrics": {"n_boot": 200},
        "walk_forward": {"train_bars": 120, "test_bars": 60, "step_bars": 60},
        "logging": {"level": "WARNING"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return load_settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path / "data")


@pytest.fixture(scope="session")
def market_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("market")


@pytest.fixture(scope="session")
def shared_settings(market_dir: Path) -> Settings:
    return make_settings(market_dir / "data")


@pytest.fixture(scope="session")
def synthetic(shared_settings: Settings) -> SyntheticPaths:
    return write_synthetic(shared_settings, list(shared_settings.market.default_universe))


@pytest.fixture(scope="session")
def orch(shared_settings: Settings, synthetic: SyntheticPaths) -> Orchestrator:
    o = build_orchestrator(shared_settings)
    o.calibrate(
        shared_settings.market.default_universe[0],
        fills_root=synthetic.fills_root,
        orders_root=synthetic.orders_root,
    )
    return o


def run_spec(paths: SyntheticPaths, strategy: str, **params: Any) -> RunSpec:
    return RunSpec(
        strategy=strategy,
        params=params,
        universe=["BTC-USDT-PERP"],
        start=paths.start,
        end=paths.end,
        known_at=paths.end,
        seed=7,
        actor=Actor(kind="robot", id="pytest"),
        tags=("test",),
    )

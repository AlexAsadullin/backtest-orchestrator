from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bt_orchestrator.domain import Fidelity
from bt_orchestrator.experiments import paired_t_test, seed_settings, t_interval, wilson_interval
from bt_orchestrator.ladder import evaluate_gate
from bt_orchestrator.orchestrator import combine_results
from bt_orchestrator.settings import PromotionCriteria, Settings, default_config_file
from bt_orchestrator.strategies.sma_cross import SmaCrossParams
from bt_orchestrator.sweeps import grid_points, random_points
from conftest import make_settings


def test_settings_paths_and_lookups(settings: Settings) -> None:
    assert settings.project_dir.name == "backtest-orchestrator"
    assert settings.reports_dir.parent == settings.project_dir
    assert settings.models_dir.is_relative_to(settings.project_dir)
    assert settings.synthetic_dir.is_relative_to(settings.project_dir)
    assert settings.trials_dir.is_relative_to(settings.project_dir)
    assert settings.sim_fake_dir.is_relative_to(settings.project_dir)
    assert settings.ledger_inbox_dir.is_relative_to(settings.data_dir)
    assert settings.fills_root.is_relative_to(settings.data_dir)
    assert settings.orders_root.is_relative_to(settings.data_dir)
    assert settings.bar_interval_ns == 60_000_000_000
    assert settings.periods_per_year == pytest.approx(525600.0)
    with pytest.raises(KeyError, match="no settings"):
        settings.strategy_settings("nope")
    with pytest.raises(KeyError, match="not configured"):
        settings.instrument("DOGE")


def test_config_file_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BT_CONFIG_FILE", str(tmp_path / "x.yaml"))
    assert default_config_file() == tmp_path / "x.yaml"
    monkeypatch.delenv("BT_CONFIG_FILE")
    assert default_config_file().name == "settings.yaml"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BT_SERVICE__PORT", "9999")
    assert make_settings(tmp_path).service.port == 9999


def test_seed_settings_isolate_each_seed(settings: Settings) -> None:
    s = seed_settings(settings, 7)
    assert s.paths.subtree.endswith("seed-7")
    assert s.synthetic.seed == 7
    assert s.synthetic.n_bars == settings.experiments.n_bars
    assert s.ledger.mode == "inbox"
    assert s.ledger.inbox_subdir == settings.experiments.ledger_inbox_subdir


def test_statistics_helpers() -> None:
    ci = t_interval([1.0, 2.0, 3.0], 0.95)
    assert ci["mean"] == pytest.approx(2.0)
    assert ci["low"] < 2.0 < ci["high"]
    assert ci["n"] == 3
    assert math.isnan(t_interval([], 0.95)["mean"])
    assert math.isnan(t_interval([1.0], 0.95)["low"])
    w = wilson_interval(0, 10, 0.95)
    assert w["low"] == 0.0
    assert 0.2 < w["high"] < 0.35
    assert math.isnan(wilson_interval(0, 0, 0.95)["rate"])
    t = paired_t_test([1.0, 2.0, 3.0], [0.0, 0.5, 1.0], 0.95)
    assert t["t"] > 0
    assert t["df"] == 2.0
    assert t["critical"] == pytest.approx(4.303, abs=1e-3)
    assert math.isnan(paired_t_test([1.0, 1.0], [0.0, 0.0], 0.95)["t"])


def test_gate_reasons() -> None:
    crit = PromotionCriteria(min_sharpe=1.0, min_fills=5, max_drawdown=0.1, max_sharpe_drop=0.5)
    good = {"sharpe": 2.0, "n_fills": 10, "max_drawdown": 0.05}
    assert evaluate_gate(good, crit, 2.2) == []
    bad = {"sharpe": 0.5, "n_fills": 1, "max_drawdown": 0.2}
    reasons = evaluate_gate(bad, crit, 2.0)
    assert len(reasons) == 4
    assert evaluate_gate({"sharpe": None, "n_fills": None, "max_drawdown": None}, crit, None)


def test_grid_and_random_points() -> None:
    pts = grid_points({"b": [1, 2], "a": [3]})
    assert pts == [{"a": 3, "b": 1}, {"a": 3, "b": 2}]
    rnd = random_points({"fast": [5, 10], "slow": [20, 30], "qty": [1.0, 1.0]}, SmaCrossParams, 4, 1)
    assert len(rnd) == 4
    assert all(isinstance(p["fast"], int) and p["qty"] == 1.0 for p in rnd)
    assert rnd == random_points({"fast": [5, 10], "slow": [20, 30], "qty": [1.0, 1.0]}, SmaCrossParams, 4, 1)


def test_combine_results_equal_weights() -> None:
    from bt_orchestrator.domain import Fidelity as Fid
    from bt_orchestrator.domain import empty_fills
    from bt_orchestrator.engines.base import EngineResult

    def res(sym: str, ret: list[float]) -> EngineResult:
        ledger = pd.DataFrame(
            {"ts": [0, 1], "ts_close": [1, 2], "ret": ret, "equity": [1.0, 1.0], "position": [1.0, -1.0]}
        )
        return EngineResult("e", Fid.EVENT, sym, empty_fills(), ledger, {}, 0.0, 100.0)

    one = res("a", [0.1, 0.0])
    assert combine_results([one], 100.0) is one.ledger
    both = combine_results([one, res("b", [0.0, 0.2])], 100.0)
    assert both["ret"].tolist() == pytest.approx([0.05, 0.1])
    assert both["position"].tolist() == [2.0, 2.0]
    assert np.isclose(both["equity"].iloc[-1], 100.0 * 1.05 * 1.1)
    assert Fidelity("tick") is Fidelity.TICK

"""Calibration recovers the known synthetic fill process, and models are versioned, never overwritten."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bt_orchestrator.calibration.fit import (
    build_samples,
    calibrate,
    fit_fill_prob,
    fit_latency,
    load_calibration_input,
    realised_vol,
    split_out_of_time,
)
from bt_orchestrator.calibration.store import FillModelStore
from bt_orchestrator.data.synthetic import generate_fills, write_date_partitioned
from bt_orchestrator.settings import Settings

ADV = 5000.0
NOW = datetime(2026, 1, 5, tzinfo=UTC)


def write_fills(settings: Settings, root: Path, n: int = 3000, seed: int = 1) -> tuple[Path, Path]:
    s = settings
    cfg = s.synthetic.fills.model_copy(update={"n_orders": n})
    out = generate_fills(
        cfg,
        symbol="BTC-USDT-PERP",
        start_ns=0,
        initial_price=60000.0,
        vol_bps_per_sqrt_s=s.synthetic.vol_bps_per_sqrt_s,
        adv=ADV,
        maker_fee=s.costs.maker_fee,
        taker_fee=s.costs.taker_fee,
        seed=seed,
    )
    fills_root, orders_root = root / "fills", root / "orders"
    write_date_partitioned(out.fills, fills_root, "ts_fill")
    write_date_partitioned(out.orders, orders_root, "ts_submit")
    return fills_root, orders_root


def test_duplicate_fill_ids_count_once(settings: Settings, tmp_path: Path) -> None:
    fills_root, orders_root = write_fills(settings, tmp_path, n=500)
    once = load_calibration_input(fills_root, orders_root, "SYNTH", "BTC-USDT-PERP")
    part = next(fills_root.glob("date=*/part-0.parquet"))
    pd.read_parquet(part).to_parquet(part.with_name("part-1.parquet"), index=False)
    twice = load_calibration_input(fills_root, orders_root, "SYNTH", "BTC-USDT-PERP")
    assert len(twice.fills) == len(once.fills)
    assert twice.fills["fill_id"].is_unique


def test_generate_fills_shapes(settings: Settings) -> None:
    out = generate_fills(
        settings.synthetic.fills.model_copy(update={"n_orders": 400}),
        symbol="X",
        start_ns=0,
        initial_price=100.0,
        vol_bps_per_sqrt_s=1.0,
        adv=ADV,
        maker_fee=0.0,
        taker_fee=0.0,
        seed=3,
    )
    assert set(out.orders["order_type"]) == {"limit"}
    assert set(out.fills["liquidity"]) == {"maker", "taker"}
    assert (out.orders["status"] == "filled").sum() == (out.fills["order_type"] == "limit").sum()


def test_calibration_recovers_the_generating_process(settings: Settings, tmp_path: Path) -> None:
    fills_root, orders_root = write_fills(settings, tmp_path, n=4000)
    data = load_calibration_input(fills_root, orders_root, "SYNTH", "BTC-USDT-PERP")
    bundle = calibrate(
        data,
        venue="SYNTH",
        symbol="BTC-USDT-PERP",
        adv=ADV,
        cfg=settings.calibration,
        version=1,
        fitted_at=NOW,
    )
    assert bundle.label_basis == "orders+fills"
    # Fill probability falls with distance and queue and rises with volatility, as generated.
    coef = dict(zip(bundle.fill_prob.features, bundle.fill_prob.coef, strict=True))
    assert coef["distance_to_touch_bps"] < 0
    assert coef["log1p_queue"] < 0
    assert coef["vol_bps_sqrt_s"] > 0
    ev = bundle.evaluation
    assert ev["brier"] < ev["brier_base_rate"] < ev["brier_touch_fills"]
    assert ev["auc"] > 0.65
    # Slippage: the half spread passes through one to one and the square-root impact is positive.
    assert bundle.slippage.coef[0] == pytest.approx(1.0, abs=0.15)
    assert bundle.slippage.coef[1] > 0
    assert ev["slippage_mae_bps"] < ev["slippage_mae_half_spread_bps"] < ev["slippage_mae_zero_bps"]
    lat = bundle.latency.quantile(0.5)
    assert 2e6 < lat < 8e6
    assert bundle.slippage.predict_bps(1.0, 0.0, 0.0) >= bundle.slippage.impact_bps(0.0, 0.0)


def test_fills_only_basis_and_errors(settings: Settings, tmp_path: Path) -> None:
    fills_root, _ = write_fills(settings, tmp_path, n=600)
    data = load_calibration_input(fills_root, None, "SYNTH", "BTC-USDT-PERP")
    samples = build_samples(data, ADV, settings.calibration)
    assert samples.label_basis == "fills_only"
    with pytest.raises(ValueError, match="ADV"):
        build_samples(data, 0.0, settings.calibration)
    with pytest.raises(ValueError, match="at least"):
        calibrate(
            data,
            venue="SYNTH",
            symbol="BTC-USDT-PERP",
            adv=ADV,
            cfg=settings.calibration.model_copy(update={"min_samples": 10_000}),
            version=1,
            fitted_at=NOW,
        )
    one_class = pd.DataFrame(
        {c: [0.0, 1.0] for c in ("distance_to_touch_bps", "log1p_queue", "vol_bps_sqrt_s")}
    )
    with pytest.raises(ValueError, match="both filled"):
        fit_fill_prob(one_class.assign(label=[1, 1]), settings.calibration)
    with pytest.raises(ValueError, match="no taker"):
        fit_latency(np.array([]))


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_calibration_input(tmp_path / "none", None, "SYNTH", "X")
    bad = tmp_path / "bad" / "date=2026-01-05"
    bad.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(bad / "part-0.parquet")
    with pytest.raises(ValueError, match="contract columns"):
        load_calibration_input(tmp_path / "bad", None, "SYNTH", "X")


def test_realised_vol_and_split() -> None:
    ts = np.arange(100, dtype="int64") * 1_000_000_000
    mid = 100.0 * np.exp(np.cumsum(np.r_[0.0, np.full(99, 1e-4)]))
    vol = realised_vol(ts[::-1], mid[::-1], 10)
    assert np.isnan(vol[-1])
    assert vol[0] == pytest.approx(1.0, rel=1e-6)
    frame = pd.DataFrame({"ts": [3, 1, 2, 4]})
    train, test = split_out_of_time(frame, 0.25)
    assert train["ts"].tolist() == [1, 2, 3]
    assert test["ts"].tolist() == [4]


def test_store_versions(settings: Settings, tmp_path: Path) -> None:
    fills_root, orders_root = write_fills(settings, tmp_path, n=1500)
    data = load_calibration_input(fills_root, orders_root, "SYNTH", "BTC-USDT-PERP")
    store = FillModelStore(tmp_path / "models")
    assert store.load("SYNTH", "BTC-USDT-PERP") is None
    assert store.next_version("SYNTH", "BTC-USDT-PERP") == 1
    b1 = calibrate(
        data,
        venue="SYNTH",
        symbol="BTC-USDT-PERP",
        adv=ADV,
        cfg=settings.calibration,
        version=1,
        fitted_at=NOW,
    )
    store.save(b1)
    with pytest.raises(FileExistsError):
        store.save(b1)
    store.save(b1.model_copy(update={"version": 2}))
    assert store.versions("SYNTH", "BTC-USDT-PERP") == [1, 2]
    loaded = store.load("SYNTH", "BTC-USDT-PERP")
    assert loaded is not None
    assert loaded.version == 2
    assert store.load("SYNTH", "BTC-USDT-PERP", version=1) == b1
    assert store.load("SYNTH", "BTC-USDT-PERP", version=9) is None

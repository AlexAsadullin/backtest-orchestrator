"""Learn fill probability, slippage and latency from the I6 fills export (contract 6.4).

Labels and features
-------------------
* Fill probability: the population is every limit order. With the optional orders extension that
  includes never-filled orders (``label_basis = "orders+fills"``); without it only orders that filled
  at some point are visible (``"fills_only"``), which biases the probability upwards. The label is
  ``first ts_fill - ts_submit <= horizon_ns``.
* Slippage: taker fills, ``side * (price - arrival_mid) / arrival_mid`` in bps. For a marketable
  order ``distance_to_touch_bps`` is the mid-to-touch distance, i.e. the half spread it crosses.
* Volatility: realised volatility of the arrival-mid sequence, ``sqrt(sum(dlog^2) / sum(dt))`` over
  the last ``vol_window`` observations, in bps per square-root second, joined back as of each submit.

The train/test split is out-of-time: the last ``test_fraction`` of the sample (by submit time) is
held out, so evaluation measures what a model fitted on the past predicts about later fills.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bt_orchestrator.calibration.models import (
    FILL_PROB_FEATURES,
    SLIPPAGE_FEATURES,
    FillModelBundle,
    FillProbModel,
    LatencyModel,
    SlippageModel,
)
from bt_orchestrator.contracts.fills import FILL_EXPORT_COLUMNS
from bt_orchestrator.settings import CalibrationSettings
from bt_orchestrator.timeutil import NANOS_PER_SECOND, ns_to_datetime

BPS = 1e4
LATENCY_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class CalibrationInput:
    fills: pd.DataFrame
    orders: pd.DataFrame | None
    fingerprint: str


@dataclass(frozen=True)
class Samples:
    limit: pd.DataFrame  # FILL_PROB_FEATURES + label + ts
    taker: pd.DataFrame  # SLIPPAGE_FEATURES + slip_bps + ts
    latency_ns: np.ndarray
    label_basis: str


def _read_partitions(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    parts = sorted(root.glob("date=*/*.parquet"))
    if not parts:
        return pd.DataFrame(), []
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True), parts


def load_calibration_input(
    fills_root: Path, orders_root: Path | None, venue: str, symbol: str
) -> CalibrationInput:
    """Read the date-partitioned fills export (and orders extension, if present) for one symbol."""
    fills, fill_parts = _read_partitions(fills_root)
    if fills.empty:
        raise FileNotFoundError(f"no fills under {fills_root}")
    missing = [c for c in FILL_EXPORT_COLUMNS if c not in fills.columns]
    if missing:
        raise ValueError(f"fills export lacks contract columns {missing}")
    fills = fills[(fills["venue"] == venue) & (fills["symbol"] == symbol)].reset_index(drop=True)
    orders: pd.DataFrame | None = None
    order_parts: list[Path] = []
    if orders_root is not None and orders_root.exists():
        raw, order_parts = _read_partitions(orders_root)
        if not raw.empty:
            orders = raw[(raw["venue"] == venue) & (raw["symbol"] == symbol)].reset_index(drop=True)
    h = hashlib.sha256()
    for p in [*fill_parts, *order_parts]:
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    h.update(f"{venue}|{symbol}".encode())
    return CalibrationInput(fills=fills, orders=orders, fingerprint="sha256:" + h.hexdigest())


def realised_vol(ts_ns: np.ndarray, mid: np.ndarray, window: int) -> np.ndarray:
    """Rolling realised volatility in bps per sqrt-second (see module docstring); NaN until defined."""
    order = np.argsort(ts_ns, kind="stable")
    t = ts_ns[order].astype("float64") / NANOS_PER_SECOND
    dlog = np.diff(np.log(mid[order])) * BPS
    dt = np.diff(t)
    sq = pd.Series(np.r_[np.nan, dlog**2]).rolling(window, min_periods=2).sum()
    tt = pd.Series(np.r_[np.nan, dt]).rolling(window, min_periods=2).sum()
    vol = np.sqrt(sq / tt.where(tt > 0)).to_numpy()
    out = np.empty_like(vol)
    out[order] = vol
    return out


def _vol_lookup(frames: list[pd.DataFrame], window: int) -> pd.DataFrame:
    timeline = (
        pd.concat([f[["ts_submit", "arrival_mid"]] for f in frames], ignore_index=True)
        .drop_duplicates("ts_submit")
        .sort_values("ts_submit")
        .reset_index(drop=True)
    )
    vol = realised_vol(timeline["ts_submit"].to_numpy(), timeline["arrival_mid"].to_numpy(), window)
    timeline["vol_bps_sqrt_s"] = pd.Series(vol).bfill().to_numpy()
    return timeline[["ts_submit", "vol_bps_sqrt_s"]]


def _with_vol(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    left = frame.sort_values("ts_submit").reset_index(drop=True)
    return pd.merge_asof(left, lookup, on="ts_submit", direction="backward")


def build_samples(data: CalibrationInput, adv: float, cfg: CalibrationSettings) -> Samples:
    """Turn raw fills/orders into labelled feature tables (see module docstring)."""
    if adv <= 0:
        raise ValueError("ADV must be positive")
    fills = data.fills
    limit_fills = fills[fills["order_type"] == "limit"]
    first_fill = limit_fills.groupby("order_id")["ts_fill"].min()
    if data.orders is not None and not data.orders.empty:
        population = data.orders[data.orders["order_type"] == "limit"].copy()
        basis = "orders+fills"
    else:
        population = limit_fills.drop_duplicates("order_id").copy()
        basis = "fills_only"
    frames = [fills] + ([data.orders] if data.orders is not None else [])
    lookup = _vol_lookup(frames, cfg.vol_window)

    population = _with_vol(population, lookup)
    wait = population["order_id"].map(first_fill) - population["ts_submit"]
    limit = pd.DataFrame(
        {
            "ts": population["ts_submit"].to_numpy(),
            "distance_to_touch_bps": population["distance_to_touch_bps"].to_numpy(dtype="float64"),
            "log1p_queue": np.log1p(population["queue_ahead_at_entry"].clip(lower=0).to_numpy()),
            "queue_ahead": population["queue_ahead_at_entry"].to_numpy(dtype="float64"),
            "vol_bps_sqrt_s": population["vol_bps_sqrt_s"].to_numpy(),
            "label": (wait.notna() & (wait <= cfg.horizon_ns)).to_numpy().astype("int64"),
        }
    )

    taker_rows = _with_vol(fills[fills["liquidity"] == "taker"].copy(), lookup)
    sign = np.where(taker_rows["side"] == "buy", 1.0, -1.0)
    mid = taker_rows["arrival_mid"].to_numpy(dtype="float64")
    taker = pd.DataFrame(
        {
            "ts": taker_rows["ts_submit"].to_numpy(),
            "half_spread_bps": taker_rows["distance_to_touch_bps"].to_numpy(dtype="float64"),
            "sqrt_participation": np.sqrt(taker_rows["qty"].to_numpy(dtype="float64") / adv),
            "vol_bps_sqrt_s": taker_rows["vol_bps_sqrt_s"].to_numpy(),
            "slip_bps": sign * (taker_rows["price"].to_numpy(dtype="float64") - mid) / mid * BPS,
        }
    )
    return Samples(
        limit=limit.dropna().reset_index(drop=True),
        taker=taker.dropna().reset_index(drop=True),
        latency_ns=taker_rows["latency_ns"].to_numpy(dtype="float64"),
        label_basis=basis,
    )


def split_out_of_time(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    cut = int(round(len(ordered) * (1.0 - test_fraction)))
    return ordered.iloc[:cut].reset_index(drop=True), ordered.iloc[cut:].reset_index(drop=True)


def fit_fill_prob(train: pd.DataFrame, cfg: CalibrationSettings) -> FillProbModel:
    x = train.loc[:, list(FILL_PROB_FEATURES)].to_numpy(dtype="float64")
    y = train["label"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("fill-probability sample needs both filled and unfilled orders")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(C=cfg.logistic_c, max_iter=cfg.max_iter)
    model.fit((x - mean) / scale, y)
    return FillProbModel(
        features=list(FILL_PROB_FEATURES),
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]),
        horizon_ns=cfg.horizon_ns,
    )


def fit_slippage(train: pd.DataFrame) -> SlippageModel:
    x = train.loc[:, list(SLIPPAGE_FEATURES)].to_numpy(dtype="float64")
    y = train["slip_bps"].to_numpy(dtype="float64")
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    return SlippageModel(
        features=list(SLIPPAGE_FEATURES),
        intercept=float(beta[0]),
        coef=beta[1:].tolist(),
        resid_std_bps=float(np.sqrt(resid @ resid / dof)),
    )


def fit_latency(latency_ns: np.ndarray) -> LatencyModel:
    if len(latency_ns) == 0:
        raise ValueError("no taker fills to estimate latency from")
    return LatencyModel(
        quantile_levels=list(LATENCY_LEVELS),
        quantiles_ns=np.quantile(latency_ns, LATENCY_LEVELS).tolist(),
        mean_ns=float(np.mean(latency_ns)),
    )


def predict_fill_prob(model: FillProbModel, frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            model.predict(d, q, v)
            for d, q, v in zip(
                frame["distance_to_touch_bps"], frame["queue_ahead"], frame["vol_bps_sqrt_s"], strict=True
            )
        ]
    )


def predict_slippage(model: SlippageModel, frame: pd.DataFrame) -> np.ndarray:
    participation = frame["sqrt_participation"].to_numpy() ** 2
    return np.array(
        [
            model.predict_bps(h, p, v)
            for h, p, v in zip(frame["half_spread_bps"], participation, frame["vol_bps_sqrt_s"], strict=True)
        ]
    )


def evaluate_holdout(
    fill_prob: FillProbModel,
    slippage: SlippageModel,
    limit_test: pd.DataFrame,
    taker_test: pd.DataFrame,
    base_rate: float,
) -> dict[str, float]:
    """Held-out errors of the calibrated models and of naive assumptions.

    Naive baselines: "touch fills" (p = 1, the default of bar-level engines), a constant base rate,
    zero slippage, and half-spread-only slippage.
    """
    y = limit_test["label"].to_numpy(dtype="float64")
    p = predict_fill_prob(fill_prob, limit_test)
    slip = taker_test["slip_bps"].to_numpy()
    pred = predict_slippage(slippage, taker_test)
    out = {
        "n_limit_test": float(len(y)),
        "n_taker_test": float(len(slip)),
        "fill_rate_test": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_touch_fills": float(np.mean((1.0 - y) ** 2)),
        "brier_base_rate": float(np.mean((base_rate - y) ** 2)),
        "slippage_mae_bps": float(np.mean(np.abs(slip - pred))),
        "slippage_mae_zero_bps": float(np.mean(np.abs(slip))),
        "slippage_mae_half_spread_bps": float(
            np.mean(np.abs(slip - taker_test["half_spread_bps"].to_numpy()))
        ),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def calibrate(
    data: CalibrationInput,
    *,
    venue: str,
    symbol: str,
    adv: float,
    cfg: CalibrationSettings,
    version: int,
    fitted_at: datetime,
) -> FillModelBundle:
    """Fit all three models on the training split and evaluate them on the held-out split."""
    samples = build_samples(data, adv, cfg)
    if len(samples.limit) < cfg.min_samples or len(samples.taker) < cfg.min_samples:
        raise ValueError(
            f"need at least {cfg.min_samples} limit orders and taker fills, "
            f"got {len(samples.limit)} and {len(samples.taker)}"
        )
    limit_train, limit_test = split_out_of_time(samples.limit, cfg.test_fraction)
    taker_train, taker_test = split_out_of_time(samples.taker, cfg.test_fraction)
    fill_prob = fit_fill_prob(limit_train, cfg)
    slippage = fit_slippage(taker_train)
    evaluation = evaluate_holdout(
        fill_prob, slippage, limit_test, taker_test, float(limit_train["label"].mean())
    )
    ts_all = data.fills["ts_submit"]
    return FillModelBundle(
        version=version,
        venue=venue,
        symbol=symbol,
        fitted_at=fitted_at,
        sample_start=ns_to_datetime(int(ts_all.min())),
        sample_end=ns_to_datetime(int(ts_all.max())),
        data_fingerprint=data.fingerprint,
        label_basis="orders+fills" if samples.label_basis == "orders+fills" else "fills_only",
        n_limit_orders=len(samples.limit),
        n_taker_fills=len(samples.taker),
        reference_queue_ahead=float(samples.limit["queue_ahead"].median()),
        reference_half_spread_bps=float(samples.taker["half_spread_bps"].median()),
        reference_vol_bps_sqrt_s=float(samples.limit["vol_bps_sqrt_s"].median()),
        fill_prob=fill_prob,
        slippage=slippage,
        latency=fit_latency(samples.latency_ns),
        evaluation=evaluation,
    )

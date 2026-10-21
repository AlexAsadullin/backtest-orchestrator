"""Cross-engine parity: run one strategy on two engines with matched costs and fills, then explain
every difference in trades, positions and PnL.

Fills of each engine are first grouped into *executions* (all fills of one side within one bar), so an
order filled in several pieces still compares as one decision. Matched executions (same bar and side)
are compared field by field; unmatched ones are paired across nearby bars when possible.

Disagreement kinds
------------------
* ``partial_fill`` - same total quantity, split into a different number of fills;
* ``rounding`` - price within ``rounding_ticks`` ticks or quantity within ``qty_lots`` lots;
* ``fill_price`` - same bar and side, price differs by more than the rounding allowance;
* ``quantity`` - quantity differs by more than the rounding allowance;
* ``fee`` - the fee rate charged on the notional differs;
* ``fill_timing`` - the same execution lands in a different bar (within ``timing_window_bars``);
* ``missing_in_a`` / ``missing_in_b`` - an execution has no counterpart.

``rounding`` and ``partial_fill`` are benign: they change PnL by at most a tick or not at all.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from bt_orchestrator.domain import InstrumentSpec
from bt_orchestrator.engines.base import EngineResult
from bt_orchestrator.settings import ParitySettings

Kind = Literal[
    "partial_fill",
    "rounding",
    "fill_price",
    "quantity",
    "fee",
    "fill_timing",
    "missing_in_a",
    "missing_in_b",
]
BENIGN: frozenset[str] = frozenset({"partial_fill", "rounding"})
MAX_LISTED = 200


class Disagreement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bar: int
    side: int
    kinds: list[Kind]
    detail: dict[str, float]


class ParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    symbol: str
    engine_a: str
    engine_b: str
    n_exec_a: int
    n_exec_b: int
    n_matched: int
    n_identical: int
    counts: dict[str, int]
    material_disagreement_rate: float
    position_mismatch_bars: int
    pnl_a: float
    pnl_b: float
    pnl_diff_rel: float
    native_pnl_a: float
    native_pnl_b: float
    wall_time_a_s: float
    wall_time_b_s: float
    passed: bool
    disagreements: list[Disagreement]
    tolerances: dict[str, Any]


def executions(fills: pd.DataFrame, ts_close: np.ndarray) -> pd.DataFrame:
    """Group fills into per-(bar, side) executions with VWAP, total fee and fill count."""
    cols = ["bar", "side", "qty", "vwap", "fee", "n_fills"]
    if fills.empty:
        return pd.DataFrame(columns=cols)
    bar = np.minimum(np.searchsorted(ts_close, fills["ts"].to_numpy(), side="left"), len(ts_close) - 1)
    frame = fills.assign(bar=bar, notional=fills["qty"] * fills["price"])
    grouped = frame.groupby(["bar", "side"], sort=True)
    out = grouped.agg(qty=("qty", "sum"), notional=("notional", "sum"), fee=("fee", "sum"), n_fills=("qty", "size"))
    out["vwap"] = out["notional"] / out["qty"]
    return out.reset_index().loc[:, cols]


def _compare(ea: pd.Series, eb: pd.Series, inst: InstrumentSpec, cfg: ParitySettings) -> list[Kind]:
    kinds: list[Kind] = []
    dq = abs(float(ea["qty"]) - float(eb["qty"]))
    if dq > 1e-12:
        kinds.append("rounding" if dq <= cfg.qty_lots * inst.lot_size + 1e-12 else "quantity")
    elif int(ea["n_fills"]) != int(eb["n_fills"]):
        kinds.append("partial_fill")
    dp = abs(float(ea["vwap"]) - float(eb["vwap"]))
    if dp > cfg.price_rel_tol * float(ea["vwap"]):
        kinds.append("rounding" if dp <= cfg.rounding_ticks * inst.tick_size + 1e-9 else "fill_price")
    rate_a = float(ea["fee"]) / (float(ea["qty"]) * float(ea["vwap"]))
    rate_b = float(eb["fee"]) / (float(eb["qty"]) * float(eb["vwap"]))
    if abs(rate_a - rate_b) > cfg.fee_rel_tol * max(abs(rate_a), abs(rate_b), 1e-12):
        kinds.append("fee")
    return list(dict.fromkeys(kinds))


def compare_runs(
    a: EngineResult, b: EngineResult, inst: InstrumentSpec, cfg: ParitySettings, strategy: str
) -> ParityReport:
    """Compare two runs of the same strategy on the same bar grid."""
    if not np.array_equal(a.ledger["ts"].to_numpy(), b.ledger["ts"].to_numpy()):
        raise ValueError("parity needs both runs on the same bar grid")
    ts_close = a.ledger["ts_close"].to_numpy()
    xa, xb = executions(a.fills, ts_close), executions(b.fills, ts_close)
    merged = xa.merge(xb, on=["bar", "side"], how="outer", suffixes=("_a", "_b"), indicator=True)

    disagreements: list[Disagreement] = []
    counts: dict[str, int] = {}
    n_identical = 0
    material = 0

    def record(bar: int, side: int, kinds: list[Kind], detail: dict[str, float]) -> None:
        nonlocal material
        if not set(kinds) <= BENIGN:
            material += 1
        for k in kinds:
            counts[k] = counts.get(k, 0) + 1
        if len(disagreements) < MAX_LISTED:
            disagreements.append(Disagreement(bar=bar, side=side, kinds=kinds, detail=detail))

    both = merged[merged["_merge"] == "both"]
    for _, row in both.iterrows():
        ea = pd.Series({k: row[f"{k}_a"] for k in ("qty", "vwap", "fee", "n_fills")})
        eb = pd.Series({k: row[f"{k}_b"] for k in ("qty", "vwap", "fee", "n_fills")})
        kinds = _compare(ea, eb, inst, cfg)
        if not kinds:
            n_identical += 1
            continue
        detail = {
            "qty_a": float(ea["qty"]),
            "qty_b": float(eb["qty"]),
            "price_a": float(ea["vwap"]),
            "price_b": float(eb["vwap"]),
            "fee_a": float(ea["fee"]),
            "fee_b": float(eb["fee"]),
        }
        record(int(row["bar"]), int(row["side"]), kinds, detail)

    only_a = merged[merged["_merge"] == "left_only"][["bar", "side", "qty_a"]].to_numpy().tolist()
    only_b = merged[merged["_merge"] == "right_only"][["bar", "side", "qty_b"]].to_numpy().tolist()
    tol_qty = cfg.qty_lots * inst.lot_size + 1e-12
    for bar_a, side_a, qty_a in only_a:
        partner = next(
            (
                j
                for j, (bar_b, side_b, qty_b) in enumerate(only_b)
                if side_b == side_a
                and abs(bar_b - bar_a) <= cfg.timing_window_bars
                and abs(qty_b - qty_a) <= tol_qty
            ),
            None,
        )
        if partner is None:
            record(int(bar_a), int(side_a), ["missing_in_b"], {"qty_a": float(qty_a)})
            continue
        bar_b, _, qty_b = only_b.pop(partner)
        record(
            int(bar_a),
            int(side_a),
            ["fill_timing"],
            {"bar_shift": float(bar_b - bar_a), "qty_a": float(qty_a), "qty_b": float(qty_b)},
        )
    for bar_b, side_b, qty_b in only_b:
        record(int(bar_b), int(side_b), ["missing_in_a"], {"qty_b": float(qty_b)})

    denom = max(len(xa), len(xb), 1)
    pos_diff = np.abs(a.ledger["position"].to_numpy() - b.ledger["position"].to_numpy())
    pnl_diff_rel = abs(a.final_pnl - b.final_pnl) / a.initial_cash
    rate = material / denom
    return ParityReport(
        strategy=strategy,
        symbol=a.symbol,
        engine_a=a.engine,
        engine_b=b.engine,
        n_exec_a=len(xa),
        n_exec_b=len(xb),
        n_matched=len(both),
        n_identical=n_identical,
        counts=counts,
        material_disagreement_rate=rate,
        position_mismatch_bars=int(np.sum(pos_diff > inst.lot_size / 2)),
        pnl_a=a.final_pnl,
        pnl_b=b.final_pnl,
        pnl_diff_rel=pnl_diff_rel,
        native_pnl_a=a.native["pnl"],
        native_pnl_b=b.native["pnl"],
        wall_time_a_s=a.wall_time_s,
        wall_time_b_s=b.wall_time_s,
        passed=rate <= cfg.max_disagreement_rate and pnl_diff_rel <= cfg.pnl_rel_tol,
        disagreements=disagreements,
        tolerances=cfg.model_dump(),
    )

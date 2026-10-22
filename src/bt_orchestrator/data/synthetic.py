"""Synthetic market tape, bars and I6-style fills with a known generating process.

Market process
--------------
Quotes update every ``quote_interval``. The log mid follows a random walk with volatility
``vol_bps_per_sqrt_s`` (bps per square-root second) plus a drift that switches every ``regime_bars``
bars between up, flat and down (``trend_bps_per_bar``). The book has ``book_levels`` contiguous levels
per side with exponential sizes. Trades arrive as a Poisson process at the prevailing touch, with a
random aggressor side. Bars are aggregated from trades, so bar closes bounce between bid and ask.

Fills process (contract 6.4 columns)
------------------------------------
Limit orders rest at a distance ``d`` (bps) from the touch with ``q`` units queued ahead. Their time to
fill is exponential with hazard ``base * exp(-a d - b log1p(q) + c (vol - vol_ref))``; an order that
does not fill within ``order_ttl_s`` is cancelled and appears only in the orders extension file. Market
orders pay ``half_spread + k sqrt(qty / ADV)`` bps plus Gaussian noise. These are the ground truths the
calibration tests recover.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from bt_orchestrator.domain import InstrumentSpec, Tape
from bt_orchestrator.settings import ScenarioSettings, SyntheticFillsSettings, SyntheticSettings
from bt_orchestrator.timeutil import NANOS_PER_SECOND, to_ns

BPS = 1e4
CAPTURE_REGION = "synthetic"


@dataclass(frozen=True)
class MarketSpec:
    symbol: str
    venue: str
    start_ns: int
    n_bars: int
    bar_interval_ns: int
    quote_interval_ns: int
    initial_price: float
    vol_bps_per_sqrt_s: float
    regime_bars: int
    trend_bps_per_bar: float
    half_spread_ticks: float
    max_extra_spread_ticks: int
    book_levels: int
    level_size_mean: float
    trade_rate_per_s: float
    trade_size_mean: float
    feed_latency_ns: int
    seed: int

    @classmethod
    def from_settings(
        cls, cfg: SyntheticSettings, symbol: str, venue: str, bar_interval_ns: int
    ) -> MarketSpec:
        return cls(
            symbol=symbol,
            venue=venue,
            start_ns=to_ns(cfg.start),
            n_bars=cfg.n_bars,
            bar_interval_ns=bar_interval_ns,
            quote_interval_ns=cfg.quote_interval_ms * 1_000_000,
            initial_price=cfg.initial_price,
            vol_bps_per_sqrt_s=cfg.vol_bps_per_sqrt_s,
            regime_bars=cfg.regime_bars,
            trend_bps_per_bar=cfg.trend_bps_per_bar,
            half_spread_ticks=cfg.half_spread_ticks,
            max_extra_spread_ticks=cfg.max_extra_spread_ticks,
            book_levels=cfg.book_levels,
            level_size_mean=cfg.level_size_mean,
            trade_rate_per_s=cfg.trade_rate_per_s,
            trade_size_mean=cfg.trade_size_mean,
            feed_latency_ns=cfg.feed_latency_ns,
            seed=cfg.seed,
        )

    def with_scenario(self, scenario: ScenarioSettings, seed: int, n_bars: int) -> MarketSpec:
        """Apply a stress scenario's multipliers (used by the in-process S3 stand-in)."""
        return replace(
            self,
            vol_bps_per_sqrt_s=self.vol_bps_per_sqrt_s * scenario.vol_multiplier,
            half_spread_ticks=self.half_spread_ticks * scenario.spread_multiplier,
            level_size_mean=self.level_size_mean * scenario.depth_multiplier,
            trade_rate_per_s=self.trade_rate_per_s * scenario.trade_rate_multiplier,
            seed=seed,
            n_bars=n_bars,
        )

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.n_bars * self.bar_interval_ns


def _mid_path(spec: MarketSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    steps_per_bar = spec.bar_interval_ns // spec.quote_interval_ns
    n_steps = spec.n_bars * steps_per_bar
    dt_s = spec.quote_interval_ns / NANOS_PER_SECOND
    n_regimes = -(-spec.n_bars // spec.regime_bars)
    regime_sign = rng.choice([-1.0, 0.0, 1.0], size=n_regimes)
    bar_of_step = np.arange(n_steps) // steps_per_bar
    drift = regime_sign[bar_of_step // spec.regime_bars] * spec.trend_bps_per_bar / steps_per_bar
    shocks = rng.standard_normal(n_steps) * spec.vol_bps_per_sqrt_s * np.sqrt(dt_s)
    log_mid = np.log(spec.initial_price) + np.cumsum((drift + shocks) / BPS)
    ts = spec.start_ns + np.arange(n_steps, dtype="int64") * spec.quote_interval_ns
    return ts, np.exp(log_mid)


def _touch_ticks(
    spec: MarketSpec, inst: InstrumentSpec, mid: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    base_spread = max(1, round(2 * spec.half_spread_ticks))
    spread = base_spread + rng.integers(0, spec.max_extra_spread_ticks + 1, size=len(mid))
    bid = np.floor(mid / inst.tick_size - spread / 2).astype("int64")
    return bid, bid + spread


def _book_deltas(
    spec: MarketSpec,
    inst: InstrumentSpec,
    ts: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    levels = spec.book_levels
    n = len(ts)
    chunks: list[pd.DataFrame] = []
    for side, best, step in (("bid", bid, -1), ("ask", ask, 1)):
        prev = np.r_[best[0], best[:-1]]
        for j in range(levels):
            # Every update republishes the full ladder of sizes.
            sizes = rng.exponential(spec.level_size_mean, size=n) + inst.lot_size
            chunks.append(
                pd.DataFrame({"ts": ts, "side": side, "tick": best + step * j, "size": sizes, "order": 1})
            )
            # Levels of the previous ladder that fall outside the new one are deleted (size 0).
            old = prev + step * j
            lo = np.minimum(best, best + step * (levels - 1))
            hi = np.maximum(best, best + step * (levels - 1))
            gone = ((old < lo) | (old > hi)) & (np.arange(n) > 0)
            chunks.append(
                pd.DataFrame({"ts": ts[gone], "side": side, "tick": old[gone], "size": 0.0, "order": 0})
            )
    book = pd.concat(chunks, ignore_index=True)
    book = book.sort_values(["ts", "order"], kind="stable").reset_index(drop=True)
    seq = np.arange(1, len(book) + 1, dtype="int64")
    return pd.DataFrame(
        {
            "venue": spec.venue,
            "symbol": spec.symbol,
            "ts_exchange": book["ts"].to_numpy(dtype="int64"),
            "ts_local": book["ts"].to_numpy(dtype="int64") + spec.feed_latency_ns,
            "capture_region": CAPTURE_REGION,
            "connection_id": "synthetic-0",
            "seq": seq,
            "side": book["side"].to_numpy(),
            "price": np.round(book["tick"].to_numpy() * inst.tick_size, inst.price_precision),
            "size": np.round(book["size"].to_numpy(), inst.size_precision),
            "prev_seq": seq - 1,
        }
    )


def _trades(
    spec: MarketSpec,
    inst: InstrumentSpec,
    ts: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    dt_s = spec.quote_interval_ns / NANOS_PER_SECOND
    counts = rng.poisson(spec.trade_rate_per_s * dt_s, size=len(ts))
    step = np.repeat(np.arange(len(ts)), counts)
    # Trades happen strictly inside the quote interval, after the quote they hit.
    offsets = rng.uniform(0.05, 0.95, size=len(step)) * spec.quote_interval_ns
    t = ts[step] + offsets.astype("int64")
    order = np.lexsort((t, step))
    step, t = step[order], t[order]
    is_buy = rng.random(len(step)) < 0.5
    ticks = np.where(is_buy, ask[step], bid[step])
    size = np.maximum(
        np.round(rng.exponential(spec.trade_size_mean, len(step)), inst.size_precision),
        inst.lot_size,
    )
    return pd.DataFrame(
        {
            "venue": spec.venue,
            "symbol": spec.symbol,
            "ts_exchange": t,
            "ts_local": t + spec.feed_latency_ns,
            "capture_region": CAPTURE_REGION,
            "connection_id": "synthetic-0",
            "seq": np.arange(1, len(t) + 1, dtype="int64"),
            "trade_id": [f"T{i}" for i in range(len(t))],
            "price": np.round(ticks * inst.tick_size, inst.price_precision),
            "size": size,
            "side": np.where(is_buy, "buy", "sell"),
        }
    )


def generate_tape(spec: MarketSpec, inst: InstrumentSpec) -> Tape:
    """Generate an I1-schema tape (trades + book deltas) for one symbol."""
    if spec.bar_interval_ns % spec.quote_interval_ns:
        raise ValueError("bar interval must be a multiple of the quote interval")
    rng = np.random.default_rng(spec.seed)
    ts, mid = _mid_path(spec, rng)
    bid, ask = _touch_ticks(spec, inst, mid, rng)
    book = _book_deltas(spec, inst, ts, bid, ask, rng)
    trades = _trades(spec, inst, ts, bid, ask, rng)
    return Tape(symbol=spec.symbol, trades=trades, book=book)


def with_knowledge_times(frame: pd.DataFrame, valid: str, known: str) -> pd.DataFrame:
    """Add the I2 bitemporal columns to a tape channel: valid = exchange time, known = arrival."""
    out = frame.copy()
    out["valid_time"] = out[valid].astype("int64")
    out["knowledge_time"] = out[known].astype("int64")
    return out


def bars_with_knowledge_times(bars: pd.DataFrame, symbol: str, interval_ns: int) -> pd.DataFrame:
    """Bars as stored for I2-style queries: a bar is valid from its open and known at its close."""
    out = bars.copy()
    out.insert(0, "symbol", symbol)
    out["valid_time"] = out["ts"].astype("int64")
    out["knowledge_time"] = out["ts"].astype("int64") + interval_ns
    return out


@dataclass(frozen=True)
class SyntheticFills:
    fills: pd.DataFrame
    orders: pd.DataFrame


def generate_fills(
    cfg: SyntheticFillsSettings,
    *,
    symbol: str,
    start_ns: int,
    initial_price: float,
    vol_bps_per_sqrt_s: float,
    adv: float,
    maker_fee: float,
    taker_fee: float,
    seed: int,
) -> SyntheticFills:
    """Generate an I6-style fills export plus the optional orders extension (see module docstring)."""
    rng = np.random.default_rng(seed)
    n = cfg.n_orders
    gaps = rng.exponential(cfg.order_spacing_s, size=n)
    t_s = np.cumsum(gaps)
    ts_submit = start_ns + (t_s * NANOS_PER_SECOND).astype("int64")
    # Volatility regimes, so the fill model can identify the volatility effect.
    regime = np.arange(n) // max(1, n // 20)
    vol_levels = vol_bps_per_sqrt_s * rng.lognormal(0.0, 0.4, size=regime.max() + 1)
    vol = vol_levels[regime]
    dmid = rng.standard_normal(n) * vol * np.sqrt(np.r_[gaps[0], gaps[1:]])
    mid = initial_price * np.exp(np.cumsum(dmid) / BPS)

    is_limit = rng.random(n) < cfg.limit_share
    side = np.where(rng.random(n) < 0.5, 1, -1)
    qty = np.maximum(np.round(rng.exponential(cfg.qty_mean, size=n), 3), 0.001)
    latency = (cfg.latency_median_ns * rng.lognormal(0.0, cfg.latency_sigma, size=n)).astype("int64")

    distance = np.where(is_limit, rng.uniform(0.0, cfg.max_distance_bps, size=n), 0.0)
    queue = np.where(is_limit, rng.exponential(cfg.queue_mean, size=n), 0.0)
    hazard = cfg.base_hazard_per_s * np.exp(
        -cfg.distance_coef * distance
        - cfg.queue_coef * np.log1p(queue)
        + cfg.vol_coef * (vol - vol_bps_per_sqrt_s)
    )
    ttf_s = rng.exponential(1.0 / hazard)
    limit_filled = is_limit & (ttf_s <= cfg.order_ttl_s)

    half_spread = cfg.half_spread_bps * rng.lognormal(0.0, 0.3, size=n)
    slip_bps = (
        half_spread
        + cfg.impact_coef_bps * np.sqrt(qty / adv)
        + rng.standard_normal(n) * cfg.slippage_noise_bps
    )
    limit_px = mid * (1.0 - side * distance / BPS)
    market_px = mid * (1.0 + side * slip_bps / BPS)
    price = np.where(is_limit, limit_px, market_px)
    ts_fill = np.where(
        is_limit, ts_submit + latency + (ttf_s * NANOS_PER_SECOND).astype("int64"), ts_submit + latency
    )
    filled = limit_filled | ~is_limit
    liquidity = np.where(is_limit, "maker", "taker")
    fee = np.abs(price * qty) * np.where(is_limit, maker_fee, taker_fee)
    order_ids = np.array([f"O{i}" for i in range(n)])
    side_str = np.where(side > 0, "buy", "sell")
    order_type = np.where(is_limit, "limit", "market")
    touch_distance = np.where(is_limit, distance, half_spread)

    fills = pd.DataFrame(
        {
            "fill_id": [f"F{i}" for i in range(n)],
            "order_id": order_ids,
            "client_order_id": [f"C{i}" for i in range(n)],
            "account_id": cfg.account_id,
            "strategy_id": cfg.strategy_id,
            "venue": cfg.venue,
            "symbol": symbol,
            "side": side_str,
            "order_type": order_type,
            "ts_submit": ts_submit,
            "ts_fill": ts_fill,
            "price": price,
            "qty": qty,
            "fee": fee,
            "liquidity": liquidity,
            "arrival_mid": mid,
            "distance_to_touch_bps": touch_distance,
            "queue_ahead_at_entry": queue,
            "latency_ns": latency,
            "scenario_id": None,
        }
    )[filled].reset_index(drop=True)

    ttl_ns = int(cfg.order_ttl_s * NANOS_PER_SECOND)
    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "client_order_id": [f"C{i}" for i in range(n)],
            "account_id": cfg.account_id,
            "strategy_id": cfg.strategy_id,
            "venue": cfg.venue,
            "symbol": symbol,
            "side": side_str,
            "order_type": order_type,
            "ts_submit": ts_submit,
            "ts_done": np.where(limit_filled, ts_fill, ts_submit + ttl_ns),
            "status": np.where(limit_filled, "filled", "cancelled"),
            "qty": qty,
            "arrival_mid": mid,
            "distance_to_touch_bps": distance,
            "queue_ahead_at_entry": queue,
            "scenario_id": None,
        }
    )[is_limit].reset_index(drop=True)
    return SyntheticFills(fills=fills, orders=orders)


def write_date_partitioned(frame: pd.DataFrame, root: Path, ts_column: str) -> list[Path]:
    """Write ``root/date=<YYYY-MM-DD>/part-0.parquet`` partitions (the I6 export layout)."""
    dates = pd.to_datetime(frame[ts_column], unit="ns", utc=True).dt.strftime("%Y-%m-%d")
    paths: list[Path] = []
    for date, part in frame.groupby(dates.to_numpy(), sort=True):
        out = root / f"date={date}"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "part-0.parquet"
        part.reset_index(drop=True).to_parquet(path, index=False)
        paths.append(path)
    return paths

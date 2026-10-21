"""Contract 6.4: the I6 fills export consumed for calibration, plus the optional orders extension."""

from __future__ import annotations

FILL_EXPORT_COLUMNS: tuple[str, ...] = (
    "fill_id",
    "order_id",
    "client_order_id",
    "account_id",
    "strategy_id",
    "venue",
    "symbol",
    "side",
    "order_type",
    "ts_submit",
    "ts_fill",
    "price",
    "qty",
    "fee",
    "liquidity",
    "arrival_mid",
    "distance_to_touch_bps",
    "queue_ahead_at_entry",
    "latency_ns",
    "scenario_id",
)

# Optional extension (documented in the README): limit orders that never filled. Without it, the
# fill-probability model can only learn from the fill-time distribution of orders that did fill.
ORDER_EXPORT_COLUMNS: tuple[str, ...] = (
    "order_id",
    "client_order_id",
    "account_id",
    "strategy_id",
    "venue",
    "symbol",
    "side",
    "order_type",
    "ts_submit",
    "ts_done",
    "status",
    "qty",
    "arrival_mid",
    "distance_to_touch_bps",
    "queue_ahead_at_entry",
    "scenario_id",
)

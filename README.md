# Backtest Orchestrator (I5)

The orchestrator runs one engine-agnostic strategy on three existing engines, vectorbt (vectorised screen),
NautilusTrader (event-driven) and hftbacktest (tick replay with queue position and latency), plus the S3
reactive simulator. It joins them into a fidelity ladder with promotion gates, learns fill probability,
slippage and latency from Shadow Exchange fills, checks cross-engine parity, and logs every run to the
Research Ledger. It does not build a new engine. On synthetic data with a known fill process, the calibrated
fill probability missed the true probability by 0.021 on average (95% CI 0.018 to 0.025, 8 seeds), while the
touch-fills assumption of bar engines missed it by 0.752. Over 240 screened candidates the gated ladder made
no false promotion and used 24.5 s of engine time, against 140.5 s for running every candidate on
hftbacktest. Details are in [docs/report.md](docs/report.md).

Specification: `quant_platform_plan_full.md`, section "I5. Multi-fidelity backtest orchestrator".

## Quickstart

```bash
cp .env.example .env            # set TRADEQ_DATA_DIR=/Users/.../Tradeq/data
make setup                      # uv venv --python 3.13 .venv, uv sync --frozen
make lint typecheck test        # ruff, mypy --strict, pytest with coverage (offline, about 30 s)
make reproduce                  # synthetic market and fills, calibration, two ladders, parity, sweep (about 1 min)
make report                     # experiments E1 to E5 behind docs/report.md, 8 seeds (about 7 min)
make serve                      # HTTP service on 127.0.0.1:8606
```

CLI: `bt synth | calibrate [--synthetic] | run | ladder | parity | sweep | walk-forward | serve | reproduce | report`.
Example: `bt ladder --strategy limit_mean_reversion -p entry_z=2.0 --all-levels`.

## Configuration

Defaults live in `config/settings.yaml`. Environment variables override them with the prefix `BT_` and the
nested delimiter `__`, for example `BT_SERVICE__PORT=9606`. `TRADEQ_DATA_DIR` is the data root, and
`BT_CONFIG_FILE` selects another settings file. The settings that matter most:

| Setting | Default | Meaning |
|---|---|---|
| `ledger.mode` | auto | `http` to S6, `inbox` (JSONL under the data root), or HTTP with inbox fallback |
| `pit.mode` | local | `http` reads through I2 (contract 6.3); `local` reads the synthetic Parquet stand-in |
| `sim.mode` | fake | `http` calls S3 (contract 6.5); `fake` writes a stressed synthetic tape in process |
| `ladder.promotion.<rung>` | see file | gate per rung: minimum Sharpe and fills, maximum drawdown and Sharpe drop |
| `parity.*` | see file | tolerances: ticks and lots counted as rounding, fee rate, PnL, timing window |
| `calibration.horizon_ns` | 60 s | a limit order counts as filled if it fills within one bar |
| `costs.maker_fee`, `costs.taker_fee` | 2 bps, 5 bps | identical fees for every engine |

## Architecture

```
I2 (or local Parquet) --> MarketData --> Orchestrator.run_level --> engine per rung --> metrics --> TrialRecord --> S6
I6 fills export --> calibration (logistic fill prob, slippage regression, latency quantiles) --> models/fills/v000N.json
ladder: vectorbt --gate--> Nautilus --gate--> hftbacktest --gate--> S3 tape on hftbacktest
parity: same strategy on vectorbt and Nautilus, fills grouped per bar and side, every difference classified
```

`engines/` holds one adapter per engine behind the `Engine` protocol; the calibrated models are injected into
Nautilus as a `FillModel` subclass and into hftbacktest as order latency plus an impact overlay on taker fills.
`strategies/` declares parameters as pydantic models and implements a vectorised target function, event
callbacks, or both. `analytics/` computes metrics and stationary-bootstrap intervals (Politis and Romano
1994). `orchestrator.py` is the only path to an engine, so no run escapes trial logging. `experiments.py`
holds the report experiments.

## Contracts

Produced: contract 6.5 backtest service (conventions section 6.5), `POST /v1/backtests` with
`{strategy, params, universe, start, end, fidelity, known_at}` returning `{run_id, trial_id, metrics,
fidelity_report?}`. Optional extensions: `ladder: true` climbs every rung up to `fidelity` and fills
`fidelity_report`; `seed` and `actor` are accepted; `GET /v1/strategies` lists strategies and parameter
schemas. Consumed: 6.1 (trials to S6, leakage events), 6.3 (I2 queries, dataset fingerprint stored in every
trial; a refused query returns 409), 6.4 (I6 fills export; an optional orders file with unfilled limit
orders, which the fill-probability model needs to avoid a fills-only bias), 6.5 (S3 simulations).

## Key results

Synthetic data only (8 seeds of 1440 one-minute bars and 6000 synthetic orders each); see
[docs/report.md](docs/report.md).

| Result | Value (95% CI) | Sample |
|---|---|---|
| Mean error of fill probability vs the true process: calibrated / touch fills / fills-only | 0.021 (0.018 to 0.025) / 0.752 (0.729 to 0.776) / 0.105 (0.095 to 0.115) | 8 seeds, about 1262 held-out orders each |
| Mean error of expected taker slippage: calibrated / half spread | 0.013 bps (0.009 to 0.016) / 0.269 bps (0.263 to 0.275) | 8 seeds, about 538 held-out fills each |
| False promotions: gated ladder / vectorbt alone | 0/240 (0% to 2%) / 6/240 (1% to 5%) | 240 candidates |
| Engine time to screen the candidates: ladder / hftbacktest alone | 24.5 s / 140.5 s | 240 candidates |
| Planted engine bugs detected: parity classifier / PnL tolerance / fill count | 64/64 (94% to 100%) / 20/64 (21% to 43%) / 16/64 (16% to 37%) | 4 fault types, 16 runs each |

## Limitations

All evidence is synthetic, and the in-process S3 stand-in is not reactive. The ladder missed 22 of the 51
candidates that pass the tick gate, because the vectorised gate asks for 10 fills while the tick gate asks
for 5, and because the event gate limits the Sharpe drop to 2.0. Nautilus applies the calibrated fill
probability only when the market rests on the limit price, so with bar data it changes almost no fills.
Annualised Sharpe ratios of one-day samples of one-minute bars are large and noisy.

## Roadmap

1. Apply the calibrated fill probability to every touched limit order in the Nautilus adapter (a custom
   matching hook), because `prob_fill_on_limit` covers only the resting case.
2. Align the minimum-fill gates across rungs, and test the gate thresholds against the missed-candidate rate.
3. Connect the reactive rung to the real S3 service and calibrate on I6 fills once they exist.

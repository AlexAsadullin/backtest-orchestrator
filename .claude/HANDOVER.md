# HANDOVER: backtest-orchestrator (Tradeq I5)

Last updated 2026-10-23 by the I5 developer agent after the integration pass against the shipped I2 and S6
(earlier sections written 2026-10-22 by the orchestrating session).

## Goal

The project orchestrates existing backtest engines into a fidelity ladder; it does not build a new engine. The rungs
are vectorbt (vectorised screen), NautilusTrader (event-driven), hftbacktest (tick and queue), and the S3 reactive
simulator, with promotion gates between rungs. The orchestrator also learns reality-calibrated fill models from
Shadow Exchange fills (contract 6.4), checks cross-engine parity, and logs every run to the Research Ledger
(contract 6.1). The spec is in `../prompts/06-backtest-orchestrator.md`. The default port is 8606. The project
produces contract 6.5 (backtest service) and consumes 6.1, 6.3, 6.4 and 6.5 (sim). For the report, the choice that
needs evidence is the method or engine adapter, which means the ladder, the gates, calibrated vs uncalibrated fills,
and parity.

## Stack

Python 3.13, uv `.venv`, and pins from `../constraints.txt`. The engines are vectorbt 1.2.0, nautilus-trader 1.231.0
and hftbacktest 2.4.2. The project also uses pandas 3, numpy, scipy, scikit-learn, numba, pyarrow, FastAPI and
uvicorn, httpx, typer (the `bt` CLI), pydantic-settings (`BT_*` env vars and `config/settings.yaml`) and structlog.

## Architecture and data flow

`wiring.py` is the composition root. It builds an `Orchestrator` (`orchestrator.py`) that loads point-in-time data
(`data/source.py`, with an I2 HTTP client or an offline Parquet stand-in), runs a strategy (`strategies/`) on the
engine for each fidelity level (`engines/factory.py`), and turns each run into a TrialRecord (`trials.py`) for S6
(`clients/ledger.py`). `ladder.py` chains the rungs with promotion gates. `calibration/fit.py` learns fill
probability, slippage and latency from I6 fills and stores them as versioned JSON (`calibration/store.py`).
`parity.py` runs one strategy on two engines and explains the difference. `sweeps.py` runs grid and random sweeps
and walk-forward. `analytics/` holds the metrics and stationary-bootstrap CIs. `reproduce.py` runs
synthetic data, then calibration, ladders, parity, a sweep and a report.

## Files

| Path | Purpose |
|---|---|
| src/bt_orchestrator/engines/{vectorbt,nautilus,hftbacktest}_engine.py, reactive_sim.py | Engine adapters behind engines/base.py |
| src/bt_orchestrator/ladder.py, orchestrator.py, parity.py, sweeps.py | Core orchestration |
| src/bt_orchestrator/calibration/{fit,models,store}.py | Calibrated fill models |
| src/bt_orchestrator/data/{source,market,tape,synthetic,synthetic_store}.py | PIT data access, tape helpers, and a synthetic market with a known generating process |
| src/bt_orchestrator/contracts/*.py | Wire models for contracts 6.1, 6.3, 6.4 and 6.5 |
| src/bt_orchestrator/analytics/{metrics,bootstrap}.py | Metrics and stationary bootstrap (Politis and Romano 1994) |
| src/bt_orchestrator/strategies/{sma_cross,limit_mean_reversion,registry,base}.py | Engine-agnostic strategies |
| src/bt_orchestrator/api.py, cli.py, reporting.py, reproduce.py | `POST /v1/backtests`, the `bt` CLI, report writers, `make reproduce` |

## State at handover

1. Commit e0e93d3 (by the stopped agent): `.gitignore` had an unanchored `data/` pattern that hid the source package `src/bt_orchestrator/data/`; it is anchored to `/data/` now.
2. Commit 5f74116 added `src/bt_orchestrator/experiments.py` (E1 to E5, 8 fixed seeds 101 to 108, 1440 bars each, Student t and Wilson intervals, paired t-tests) and replaced the unused `experiments.speed_repeats` and `experiments.docs_tables` settings with `fault_fee_bps`, `fault_price_ticks` and `fault_share`. `make report` works: about 7.5 minutes, outputs `tables.md` and `experiments.json` in `$TRADEQ_DATA_DIR/backtest-orchestrator/reports/`, per-seed data in `$TRADEQ_DATA_DIR/backtest-orchestrator/experiments/seed-*`, trials in a private inbox there.
3. Commit 227532a added 72 tests (unit and integration, offline, about 30 s); coverage is 98%. `make lint typecheck test` is clean.
4. Commit 865d578 wrote the lean README. `docs/report.md` text was returned to the orchestrator (subagents cannot write report files), so the orchestrator writes and commits it.
5. Headline results (synthetic): calibrated fill-probability error 0.021 against 0.752 for touch fills; the gated ladder made 0 false promotions in 240 candidates at 24.5 s of engine time against 140.5 s for hftbacktest alone; the parity classifier detected 64 of 64 planted bugs against 20 for a PnL tolerance.
6. Open findings: NautilusTrader applies `prob_fill_on_limit` only when the market rests on the limit price, so the calibrated probability barely changes bar-mode fills (E4 shows no significant gain); the ladder missed 22 of 51 tick-accepted candidates because the vectorised gate asks for 10 fills while the tick gate asks for 5, and the event gate caps the Sharpe drop at 2.0. Both are in the README roadmap; no threshold was tuned.

## Integration pass with the shipped I2, S6 and I9 (2026-10-23)

1. Commit 1696fe2 aligned the I2 client with `point-in-time-store` (`models.py`, `api.py`, `wire.py`). Filters are now I2's typed list `[{column, op, value}]` (they were a dict, which I2 rejects with 422). Requests send `format` and `strategy_id`. The JSON format is supported, and its ISO-8601 time columns (`pit.timestamp_columns`) are turned back into int64 ns. `X-Leakage-Flag` becomes `QueryResult.leakage_flags` and trial tags. The local dataset names are now I2's `tape.trades` and `tape.book_delta`. I2 has no bars dataset, so `pit.bars_source: trades` builds bars from `tape.trades`.
2. Trials store I2's own fingerprint of the primary dataset (the bars dataset, or `tape.trades`). The derived hash over several fingerprints was removed, because I2 could not resolve it. Every dataset fingerprint is also a tag `fingerprint:<symbol>:<dataset>=<fp>`.
3. A 409 from the real I2 raises `LeakageError(reported=True)`, and the orchestrator no longer reposts the event to S6 (I2 already does). The local stand-in still reports its own events.
4. TrialRecord against `research_ledger/sdk/wire.py`: the fields match. S6 rejects NaN in returns and strict JSON has no NaN, so non-finite metrics are now sent as `null` (`trials.finite_metrics`, also in the API response). S6's batch endpoint returns a list of TrialAck, which the client already handled.
5. Commit 96c996e added `make integration` (marker `integration`, deselected by default; `tests/live/test_real_services.py`). It starts `pit serve` and `ledger serve` from the sibling venvs on free loopback ports with a scratch data root under `tmp/`, seeds I2 with `pit ingest tape.trades|tape.book_delta`, climbs vectorised and event, checks `GET /v1/fingerprints/{fp}` on I2 and a 409 for a leaking query, then reads S6's `trials` table with S6's own Python. The run on 2026-10-23 passed (1 passed in 5.3 s).
6. Commit 243a0f9: I9 (`mcp_gateway/clients/research.py`, `gateway.py`) sends exactly the contract request fields and accepts the response, but it maps an agent-scoped strategy id that it never sends, so trials landed under `i5-<strategy>-<universe>` while I9 asks S6 for its own id. I5 now accepts an optional `strategy_id` (and already accepted `actor`); I9 still has to send both. Calibration also counts each `fill_id` once, because I6's export used to repeat fills.
7. The report numbers do not change: the experiments read their own synthetic fills under `experiments/seed-*`, never I6's shared export, and none of the changes touch engine results.
8. State: `make lint typecheck test` is clean, 76 tests pass with 98% coverage, and the integration test passes.

## Mistakes and lessons

1. An unanchored ignore pattern silently excluded source code from git. When an ignore pattern names a directory, anchor it to the repo root.
2. A CLI command was wired to a module before the module was written, which left `make report` broken. Check entry points with mypy before committing.
3. The project was marked as built without a test suite. Tests go before the report.
4. `make reproduce` with the default `ledger.mode: auto` falls back to the shared Research Ledger inbox under the data root; for local trial runs set `BT_LEDGER__MODE=inbox` and `BT_LEDGER__INBOX_SUBDIR` to a private folder. The report experiments already do this.
5. `parse_iso` truncates to microseconds, so tests that build timestamps must use whole seconds, not nanosecond offsets.
6. Timings in the report depend on CPU load, so run `make report` alone, not in parallel with the tests.
7. These lessons apply platform-wide. Compound shell commands were denied for subagents, so use `make -C`, `git -C` and `.venv/bin/python`. Subagents may not write report files, so return the text to the orchestrator. Commit early, because of the rate limit.
8. A dict-shaped filter that I had assumed for I2 was wrong: I2 takes a typed list and rejects extra shapes. Read the producer's wire models before writing a client, and keep a live test against the real service.
9. A combined fingerprint looked tidy but broke reproducibility, because only the producer can resolve its fingerprints. Store producer identifiers verbatim.
10. When a producer reports an event itself (I2 posts leakage to S6), the consumer must not report it again.

## MOEX pivot (2026-10-29)

`config/settings.yaml` gained the MOEX instruments SBER, GAZP and LKOH (`market.instruments`, lot sizes in shares,
ticks from the MOEX securities table), so I9's real-service test can run a backtest on the MOEX tape in the
real I2 (`BT_MARKET__VENUE=moex-tbank`). The crypto instruments stay, because the report experiments use a
synthetic BTC market. `make test` unchanged (76 passed). Next for I5 (not done): calibrate fills from I6 MOEX
fills and the I9 mirror, and run the ladder on a recorded MOEX day.

# HANDOVER: backtest-orchestrator (Tradeq I5)

Last updated 2026-10-22 by the orchestrating session, from the transcript of the agent that stopped at the
session rate limit.

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

## Mistakes and lessons

1. An unanchored ignore pattern silently excluded source code from git. When an ignore pattern names a directory, anchor it to the repo root.
2. A CLI command was wired to a module before the module was written, which left `make report` broken. Check entry points with mypy before committing.
3. The project was marked as built without a test suite. Tests go before the report.
4. `make reproduce` with the default `ledger.mode: auto` falls back to the shared Research Ledger inbox under the data root; for local trial runs set `BT_LEDGER__MODE=inbox` and `BT_LEDGER__INBOX_SUBDIR` to a private folder. The report experiments already do this.
5. `parse_iso` truncates to microseconds, so tests that build timestamps must use whole seconds, not nanosecond offsets.
6. Timings in the report depend on CPU load, so run `make report` alone, not in parallel with the tests.
7. These lessons apply platform-wide. Compound shell commands were denied for subagents, so use `make -C`, `git -C` and `.venv/bin/python`. Subagents may not write report files, so return the text to the orchestrator. Commit early, because of the rate limit.

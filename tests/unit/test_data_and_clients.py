from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import respx

from bt_orchestrator.clients.ledger import LedgerClient
from bt_orchestrator.clients.sim import FakeSimService, SimHttpClient
from bt_orchestrator.contracts.ledger import Actor, LeakageEvent, ReturnSeries, TrialRecord
from bt_orchestrator.contracts.pit import FINGERPRINT_HEADER, LEAKAGE_FLAG_HEADER, PitFilter, PitQuery
from bt_orchestrator.contracts.sim import SimulationRequest
from bt_orchestrator.data.market import MarketData
from bt_orchestrator.data.source import LeakageError, LocalParquetSource, PitHttpSource
from bt_orchestrator.data.synthetic import MarketSpec, generate_tape
from bt_orchestrator.data.tape import bars_from_trades, event_time, read_i1_tape, write_i1_tape
from bt_orchestrator.orchestrator import instrument_spec
from bt_orchestrator.settings import Settings
from bt_orchestrator.trials import TrialBuilder, TrialInputs, git_sha

BASE = "http://ledger.test"
NOW = datetime(2026, 1, 5, tzinfo=UTC)
S = 1_000_000_000


def record(**kw: object) -> TrialRecord:
    fields: dict[str, object] = {
        "family": "f",
        "strategy_id": "s",
        "source": "i5",
        "actor": Actor(kind="robot", id="t"),
        "params": {},
        "dataset_fingerprint": None,
        "code_version": None,
        "seed": 1,
        "started_at": NOW,
        "finished_at": NOW,
        "periods_per_year": 1.0,
        "returns": ReturnSeries(ts=["2026-01-05T00:00:00Z"], values=[0.0]),
        "metrics": {},
        "tags": [],
    }
    return TrialRecord.model_validate(fields | kw)


def test_trial_record_needs_exactly_one_return_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        record(returns=None)
    with pytest.raises(ValueError, match="same length"):
        ReturnSeries(ts=["a"], values=[])


def test_ledger_inbox_mode(tmp_path: Path) -> None:
    client = LedgerClient(mode="inbox", base_url=BASE, timeout_s=1, inbox_dir=tmp_path, source="i5")
    assert client.log_trials([]) == []
    ids = client.log_trials([record(), record()])
    assert len(ids) == 2
    lines = next(tmp_path.glob("i5-*.jsonl")).read_text().splitlines()
    assert len(lines) == 2
    assert client.last_sink == "inbox"
    event = LeakageEvent(
        source="i2", strategy_id=None, trial_id=None, severity="block", kind="k", detail={}, observed_at=NOW
    )
    assert client.report_leakage(event) is False


@respx.mock
def test_ledger_http_and_fallback(tmp_path: Path) -> None:
    respx.post(f"{BASE}/v1/trials").mock(return_value=httpx.Response(200, json={"trial_id": "x"}))
    respx.post(f"{BASE}/v1/trials/batch").mock(
        return_value=httpx.Response(200, json={"trial_ids": ["a", {"trial_id": "b"}]})
    )
    respx.post(f"{BASE}/v1/leakage").mock(return_value=httpx.Response(500))
    client = LedgerClient(mode="http", base_url=BASE, timeout_s=1, inbox_dir=tmp_path, source="i5")
    assert client.log_trials([record()]) == ["x"]
    assert client.log_trials([record(), record()]) == ["a", "b"]
    assert client.last_sink == "http"
    event = LeakageEvent(
        source="i2", strategy_id=None, trial_id=None, severity="block", kind="k", detail={}, observed_at=NOW
    )
    assert client.report_leakage(event) is False


@respx.mock
def test_ledger_auto_falls_back_on_transport_error(tmp_path: Path) -> None:
    respx.post(f"{BASE}/v1/trials").mock(side_effect=httpx.ConnectError("down"))
    auto = LedgerClient(mode="auto", base_url=BASE, timeout_s=1, inbox_dir=tmp_path, source="i5")
    assert len(auto.log_trials([record()])) == 1
    assert auto.last_sink == "inbox"
    strict = LedgerClient(mode="http", base_url=BASE, timeout_s=1, inbox_dir=tmp_path, source="i5")
    with pytest.raises(httpx.ConnectError):
        strict.log_trials([record()])


def test_trial_builder_inline_and_parquet(settings: Settings, tmp_path: Path) -> None:
    inputs = TrialInputs(
        strategy="sma_cross",
        universe=["BTC-USDT-PERP"],
        params={"fast": 1},
        returns=pd.Series([0.01, -0.01], index=[60_000_000_000, 120_000_000_000]),
        metrics={"sharpe": 1.0},
        dataset_fingerprint="sha256:x",
        seed=1,
        started_at=NOW,
        finished_at=NOW,
        periods_per_year=525600.0,
        tags=["t"],
    )
    rec = TrialBuilder(settings.ledger, tmp_path, "abc").build(inputs)
    assert rec.family == "i5/sma_cross/BTC-USDT-PERP"
    assert rec.returns is not None
    assert rec.returns.ts[0] == "1970-01-01T00:01:00Z"
    assert rec.code_version == "abc"
    small = settings.ledger.model_copy(update={"max_inline_returns": 1})
    rec2 = TrialBuilder(small, tmp_path, None).build(inputs)
    assert rec2.returns is None
    assert rec2.returns_path is not None
    assert len(pd.read_parquet(rec2.returns_path)) == 2
    assert git_sha(tmp_path) is None


def _local(settings: Settings, tmp_path: Path) -> LocalParquetSource:
    path = tmp_path / "bars.parquet"
    pd.DataFrame(
        {
            "symbol": ["X", "X", "X", "Y"],
            "ts": [0, 0, 60 * S, 0],
            "valid_time": [0, 0, 60 * S, 0],
            "knowledge_time": [60 * S, 90 * S, 120 * S, 60 * S],
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(path)
    cfg = settings.local_source.model_copy(
        update={
            "datasets": {
                "bars": settings.local_source.datasets["bars_1m"].model_copy(update={"path": "bars.parquet"})
            }
        }
    )
    return LocalParquetSource(tmp_path, cfg)


def iso(seconds: int) -> str:
    return pd.Timestamp(seconds, unit="s", tz="UTC").isoformat()


def test_local_source_point_in_time(settings: Settings, tmp_path: Path) -> None:
    src = _local(settings, tmp_path)
    q = PitQuery(
        dataset="bars",
        start=iso(0),
        end=iso(120),
        known_at=iso(200),
        filters=[PitFilter(column="symbol", op="==", value="X")],
    )
    res = src.query(q)
    assert res.frame["close"].tolist() == [2.0, 3.0]  # the later revision wins
    early = src.query(q.model_copy(update={"known_at": iso(130), "end": iso(120)}))
    assert early.fingerprint != res.fingerprint
    cols = src.query(
        q.model_copy(
            update={"columns": ["close"], "filters": [PitFilter(column="symbol", op="in", value=["X", "Y"])]}
        )
    )
    assert list(cols.frame.columns) == ["close"]
    with pytest.raises(LeakageError, match="end_after_known_at"):
        src.query(q.model_copy(update={"known_at": iso(60)}))
    with pytest.raises(KeyError, match="not configured"):
        src.query(q.model_copy(update={"dataset": "nope"}))


def test_local_source_missing_files_and_knowledge_time(settings: Settings, tmp_path: Path) -> None:
    src = _local(settings, tmp_path)
    pd.DataFrame({"symbol": ["X"], "ts": [0], "valid_time": [0], "close": [1.0]}).to_parquet(
        tmp_path / "bars.parquet"
    )
    q = PitQuery(dataset="bars", start=iso(0), end=iso(60), known_at=iso(60))
    with pytest.raises(LeakageError, match="missing_knowledge_time"):
        src.query(q)
    assert len(src.query(q.model_copy(update={"mode": "live"})).frame) == 1
    (tmp_path / "bars.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        src.query(q)


def _arrow(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


LEAK_BODY = {
    "schema_version": "1",
    "source": "i2",
    "strategy_id": "i5-x",
    "trial_id": None,
    "severity": "block",
    "kind": "end_after_known_at",
    "detail": {"dataset": "tape.trades"},
    "observed_at": "2026-01-05T00:00:00Z",
}


@respx.mock
def test_pit_http_source_arrow() -> None:
    body = _arrow(pa.table({"ts": [0], "close": [1.0]}))
    route = respx.post("http://pit.test/v1/query")
    route.side_effect = [
        httpx.Response(200, content=body, headers={FINGERPRINT_HEADER: "sha256:f"}),
        httpx.Response(
            200, content=body, headers={FINGERPRINT_HEADER: "sha256:g", LEAKAGE_FLAG_HEADER: "a, b"}
        ),
        httpx.Response(200, content=body),
        httpx.Response(409, json=LEAK_BODY),
    ]
    src = PitHttpSource("http://pit.test", 1.0)
    q = PitQuery(
        dataset="bars",
        start=iso(0),
        end=iso(60),
        known_at=iso(60),
        filters=[PitFilter(column="symbol", op="==", value="X")],
        strategy_id="i5-x",
    )
    res = src.query(q)
    assert res.fingerprint == "sha256:f"
    assert res.leakage_flags == ()
    assert res.frame["close"].tolist() == [1.0]
    sent = json.loads(route.calls[0].request.content)
    assert sent["format"] == "arrow"
    assert sent["filters"] == [{"column": "symbol", "op": "==", "value": "X"}]
    assert sent["strategy_id"] == "i5-x"
    assert src.query(q).leakage_flags == ("a", "b")
    with pytest.raises(ValueError, match="lacks"):
        src.query(q)
    with pytest.raises(LeakageError) as err:
        src.query(q)
    assert err.value.event.kind == "end_after_known_at"
    src.close()


@respx.mock
def test_pit_http_source_json_times() -> None:
    respx.post("http://pit.test/v1/query").mock(
        return_value=httpx.Response(
            200,
            json={
                "schema_version": "1",
                "fingerprint": "sha256:j",
                "columns": ["ts_local", "price", "knowledge_time"],
                "rows": [
                    {"ts_local": "2026-01-05T00:00:01.000000002Z", "price": 1.5, "knowledge_time": None},
                ],
                "flags": [],
            },
            headers={FINGERPRINT_HEADER: "sha256:j"},
        )
    )
    src = PitHttpSource("http://pit.test", 1.0, fmt="json", timestamp_columns=("ts_local", "knowledge_time"))
    res = src.query(PitQuery(dataset="tape.trades", start=iso(0), end=iso(60), known_at=iso(60)))
    expected = pd.Timestamp("2026-01-05T00:00:01.000000002Z").value
    assert int(res.frame["ts_local"].iloc[0]) == expected
    assert pd.isna(res.frame["knowledge_time"].iloc[0])
    assert res.fingerprint == "sha256:j"


def test_pit_filter_validation() -> None:
    with pytest.raises(ValueError, match="list"):
        PitFilter(column="symbol", op="in", value="X")
    with pytest.raises(ValueError, match="scalar"):
        PitFilter(column="symbol", op="==", value=["X"])


def test_local_filter_operators(settings: Settings, tmp_path: Path) -> None:
    from bt_orchestrator.data.source import filter_mask

    frame = pd.DataFrame({"x": [1, 2, 3]})
    cases = {"==": [2], "!=": [1, 3], "<": [1], "<=": [1, 2], ">": [3], ">=": [2, 3]}
    for op, expected in cases.items():
        mask = filter_mask(frame, [PitFilter(column="x", op=op, value=2)])  # type: ignore[arg-type]
        assert frame.loc[mask, "x"].tolist() == expected


def test_market_data_errors(settings: Settings, tmp_path: Path) -> None:
    class Empty:
        def query(self, q: PitQuery) -> object:
            from bt_orchestrator.data.source import QueryResult

            return QueryResult(pd.DataFrame(), "f")

    md = MarketData(Empty(), settings.pit, venue="SYNTH", bar_interval_ns=60 * S)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no bars"):
        md.load("X", iso(0), iso(60), iso(60), with_tape=False)
    trades_cfg = settings.pit.model_copy(update={"bars_source": "trades"})
    md = MarketData(Empty(), trades_cfg, venue="SYNTH", bar_interval_ns=60 * S)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no trades"):
        md.load("X", iso(0), iso(60), iso(60), with_tape=False)


def test_tape_round_trip_and_bars(settings: Settings, tmp_path: Path) -> None:
    inst = instrument_spec(settings, "BTC-USDT-PERP")
    spec = MarketSpec.from_settings(settings.synthetic, "BTC-USDT-PERP", "SYNTH", settings.bar_interval_ns)
    spec = MarketSpec(**{**spec.__dict__, "n_bars": 5})
    tape = generate_tape(spec, inst)
    assert len(write_i1_tape(tape, tmp_path, "SYNTH")) == 2
    back = read_i1_tape(tmp_path, "BTC-USDT-PERP")
    assert len(back.trades) == len(tape.trades)
    with pytest.raises(FileNotFoundError):
        read_i1_tape(tmp_path, "ETH-USDT-PERP")
    bars = bars_from_trades(back.trades, spec.start_ns, spec.end_ns, settings.bar_interval_ns)
    assert len(bars) == 5
    assert (bars["high"] >= bars["low"]).all()
    with pytest.raises(ValueError, match="after"):
        bars_from_trades(back.trades, 10, 10, 1)
    local_only = back.trades.drop(columns=["ts_exchange"])
    assert np.array_equal(event_time(local_only), back.trades["ts_local"].to_numpy())
    with pytest.raises(ValueError, match="multiple"):
        generate_tape(MarketSpec(**{**spec.__dict__, "quote_interval_ns": 7_000_000_000}), inst)


@respx.mock
def test_sim_clients(settings: Settings, tmp_path: Path) -> None:
    respx.get("http://sim.test/v1/scenarios").mock(
        return_value=httpx.Response(200, json={"scenarios": ["a", {"name": "b"}]})
    )
    respx.post("http://sim.test/v1/simulations").mock(
        return_value=httpx.Response(200, json={"simulation_id": "s", "tape_path": "/x"})
    )
    http = SimHttpClient("http://sim.test", 1.0)
    req = SimulationRequest(scenario="a", seed=1, duration_s=120, venue="SYNTH", symbol="BTC-USDT-PERP")
    assert http.scenarios() == ["a", "b"]
    assert http.simulate(req).tape_path == "/x"

    base = MarketSpec.from_settings(settings.synthetic, "BTC-USDT-PERP", "SYNTH", settings.bar_interval_ns)
    req = req.model_copy(update={"scenario": "baseline"})
    fake = FakeSimService(
        tmp_path,
        base,
        {"BTC-USDT-PERP": instrument_spec(settings, "BTC-USDT-PERP")},
        settings.sim.fake_scenarios,
    )
    assert fake.scenarios() == ["baseline", "thin-book-stress"]
    resp = fake.simulate(req)
    assert read_i1_tape(Path(resp.tape_path), "BTC-USDT-PERP").trades.shape[0] > 0
    assert fake.simulate(req) == resp
    with pytest.raises(ValueError, match="unknown"):
        fake.simulate(req.model_copy(update={"scenario": "nope"}))
    assert json.loads(req.model_dump_json())["seed"] == 1

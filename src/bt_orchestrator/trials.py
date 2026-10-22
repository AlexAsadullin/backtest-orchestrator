"""Turns a finished run into a contract 6.1 :class:`TrialRecord`."""

from __future__ import annotations

import math
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from bt_orchestrator.contracts.ledger import Actor, ReturnSeries, TrialRecord
from bt_orchestrator.settings import REPO_ROOT, LedgerSettings
from bt_orchestrator.timeutil import ns_to_iso


@lru_cache(maxsize=1)
def git_sha(repo: Path = REPO_ROOT) -> str | None:
    """The repository's HEAD sha, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def finite_metrics(metrics: dict[str, float | int | None]) -> dict[str, float | int | None]:
    """Replace NaN and infinities by ``None``: JSON has no NaN, and S6 and I9 parse strict JSON."""
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in metrics.items()}


@dataclass(frozen=True)
class TrialInputs:
    strategy: str
    universe: list[str]
    params: dict[str, Any]
    returns: pd.Series  # index: bar close ts (ns), values: net returns
    metrics: dict[str, float | int | None]
    dataset_fingerprint: str | None
    seed: int | None
    started_at: datetime
    finished_at: datetime
    periods_per_year: float
    tags: list[str]
    actor: Actor | None = None


class TrialBuilder:
    def __init__(self, cfg: LedgerSettings, trials_dir: Path, code_version: str | None) -> None:
        self._cfg = cfg
        self._dir = trials_dir
        self._code = code_version if code_version is not None else git_sha()

    def strategy_id(self, strategy: str, universe: list[str]) -> str:
        return self._cfg.strategy_id_template.format(strategy=strategy, universe="+".join(universe))

    def build(self, x: TrialInputs) -> TrialRecord:
        universe = "+".join(x.universe)
        trial_id = str(uuid.uuid4())
        ts = [ns_to_iso(int(t)) for t in x.returns.index]
        values = [float(v) for v in x.returns.to_numpy()]
        returns: ReturnSeries | None = None
        returns_path: str | None = None
        if len(values) <= self._cfg.max_inline_returns:
            returns = ReturnSeries(ts=ts, values=values)
        else:
            path = self._dir / trial_id / "returns.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"ts": pd.to_datetime(x.returns.index, unit="ns", utc=True), "ret": values}
            ).to_parquet(path, index=False)
            returns_path = str(path)
        actor = x.actor or Actor(kind=self._cfg.actor.kind, id=self._cfg.actor.id)
        return TrialRecord(
            trial_id=trial_id,
            family=self._cfg.family_template.format(strategy=x.strategy, universe=universe),
            strategy_id=self.strategy_id(x.strategy, x.universe),
            source=self._cfg.source,
            actor=actor,
            params=x.params,
            dataset_fingerprint=x.dataset_fingerprint,
            code_version=self._code,
            seed=x.seed,
            started_at=x.started_at,
            finished_at=x.finished_at,
            periods_per_year=x.periods_per_year,
            returns=returns,
            returns_path=returns_path,
            metrics=finite_metrics(x.metrics),
            tags=[*self._cfg.tags, *x.tags],
        )

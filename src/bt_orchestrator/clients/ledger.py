"""Contract 6.1 Research Ledger client: HTTP with the JSONL inbox as the offline fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import httpx
import structlog

from bt_orchestrator.contracts.ledger import LeakageEvent, TrialAck, TrialRecord
from bt_orchestrator.timeutil import utc_now

log = structlog.get_logger(__name__)


class LedgerClient:
    """Logs trials to S6. ``auto`` mode falls back to the inbox when S6 cannot be reached."""

    def __init__(
        self,
        *,
        mode: Literal["http", "inbox", "auto"],
        base_url: str,
        timeout_s: float,
        inbox_dir: Path,
        source: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._mode = mode
        self._inbox = inbox_dir
        self._source = source
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self.last_sink: str | None = None

    def _inbox_path(self) -> Path:
        return self._inbox / f"{self._source}-{utc_now():%Y%m%d}.jsonl"

    def _to_inbox(self, records: list[TrialRecord]) -> list[str]:
        self._inbox.mkdir(parents=True, exist_ok=True)
        with self._inbox_path().open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(record.model_dump_json() + "\n")
        self.last_sink = "inbox"
        return [r.trial_id for r in records]

    def _to_http(self, records: list[TrialRecord]) -> list[str]:
        if len(records) == 1:
            response = self._client.post("/v1/trials", json=records[0].model_dump(mode="json"))
            response.raise_for_status()
            ids = [TrialAck.model_validate(response.json()).trial_id]
        else:
            payload = [r.model_dump(mode="json") for r in records]
            response = self._client.post("/v1/trials/batch", json=payload)
            response.raise_for_status()
            body = response.json()
            items = body.get("trial_ids", body) if isinstance(body, dict) else body
            ids = [i if isinstance(i, str) else TrialAck.model_validate(i).trial_id for i in items]
        self.last_sink = "http"
        return ids

    def log_trials(self, records: list[TrialRecord]) -> list[str]:
        """Send trials; returns the trial ids the ledger acknowledged (or wrote to the inbox)."""
        if not records:
            return []
        if self._mode == "inbox":
            return self._to_inbox(records)
        try:
            return self._to_http(records)
        except httpx.TransportError as exc:
            if self._mode == "http":
                raise
            log.warning("ledger_unreachable_using_inbox", error=str(exc), n=len(records))
            return self._to_inbox(records)

    def report_leakage(self, event: LeakageEvent) -> bool:
        """Best-effort ``POST /v1/leakage``; returns whether S6 accepted it."""
        if self._mode == "inbox":
            log.warning("leakage_event_not_sent_offline", kind=event.kind, detail=event.detail)
            return False
        try:
            response = self._client.post("/v1/leakage", json=event.model_dump(mode="json"))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("leakage_event_not_sent", error=str(exc), kind=event.kind)
            return False
        return True

"""Versioned storage of calibrated fill models under the data root.

Layout: ``<models_dir>/venue=<v>/symbol=<s>/v<NNNN>.json`` plus ``latest.json`` naming the current
version. Versions are never overwritten, so every backtest can cite the exact model it used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bt_orchestrator.calibration.models import FillModelBundle

_VERSION_FILE = re.compile(r"^v(\d{4,})\.json$")


class FillModelStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _dir(self, venue: str, symbol: str) -> Path:
        return self._root / f"venue={venue}" / f"symbol={symbol}"

    def versions(self, venue: str, symbol: str) -> list[int]:
        d = self._dir(venue, symbol)
        if not d.exists():
            return []
        found = (_VERSION_FILE.match(p.name) for p in d.iterdir())
        return sorted(int(m.group(1)) for m in found if m)

    def next_version(self, venue: str, symbol: str) -> int:
        existing = self.versions(venue, symbol)
        return (existing[-1] + 1) if existing else 1

    def save(self, bundle: FillModelBundle) -> Path:
        d = self._dir(bundle.venue, bundle.symbol)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"v{bundle.version:04d}.json"
        if path.exists():
            raise FileExistsError(f"model version {bundle.version} already exists at {path}")
        path.write_text(bundle.model_dump_json(indent=2))
        (d / "latest.json").write_text(json.dumps({"version": bundle.version}))
        return path

    def load(self, venue: str, symbol: str, version: int | None = None) -> FillModelBundle | None:
        """Load a version (default: latest); ``None`` when no model was calibrated yet."""
        d = self._dir(venue, symbol)
        if version is None:
            pointer = d / "latest.json"
            if not pointer.exists():
                return None
            version = int(json.loads(pointer.read_text())["version"])
        path = d / f"v{version:04d}.json"
        if not path.exists():
            return None
        return FillModelBundle.model_validate_json(path.read_text())

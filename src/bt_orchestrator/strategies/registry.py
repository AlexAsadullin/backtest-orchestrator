"""Name -> strategy lookup used by the CLI, the service and the orchestrator."""

from __future__ import annotations

from collections.abc import Iterable

from bt_orchestrator.strategies.base import Strategy
from bt_orchestrator.strategies.limit_mean_reversion import LimitMeanReversion
from bt_orchestrator.strategies.sma_cross import SmaCross


class StrategyRegistry:
    def __init__(self, strategies: Iterable[Strategy]) -> None:
        self._by_name = {s.name: s for s in strategies}

    def get(self, name: str) -> Strategy:
        try:
            return self._by_name[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._by_name))
            raise KeyError(f"unknown strategy {name!r}; known: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name


def default_registry() -> StrategyRegistry:
    return StrategyRegistry([SmaCross(), LimitMeanReversion()])

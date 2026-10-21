"""Typed settings: defaults from ``config/settings.yaml``, overridden by ``BT_*`` environment variables.

The platform data root comes from ``TRADEQ_DATA_DIR`` (environment or ``.env``). Nothing in the package
hardcodes paths, URLs, ports, symbols, fees or thresholds; they all live here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from bt_orchestrator.contracts.ledger import TrialSource
from bt_orchestrator.domain import Fidelity
from bt_orchestrator.timeutil import NANOS_PER_SECOND

CONFIG_ENV_VAR = "BT_CONFIG_FILE"
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_config_file() -> Path:
    """Return the YAML file holding the defaults (``BT_CONFIG_FILE`` overrides the repo default)."""
    override = os.environ.get(CONFIG_ENV_VAR)
    return Path(override) if override else REPO_ROOT / "config" / "settings.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceSettings(_Section):
    host: str
    port: int


class LoggingSettings(_Section):
    level: str
    json_output: bool


class PathSettings(_Section):
    subtree: str
    reports: str
    models: str
    synthetic: str
    trials: str
    sim_fake: str


class ActorSettings(_Section):
    kind: Literal["human", "agent", "robot"]
    id: str


class LedgerSettings(_Section):
    base_url: str
    mode: Literal["http", "inbox", "auto"]
    timeout_s: float
    inbox_subdir: str
    source: TrialSource
    family_template: str
    strategy_id_template: str
    actor: ActorSettings
    max_inline_returns: int
    tags: list[str]


class PitDatasets(_Section):
    bars: str
    trades: str
    book: str


class PitSettings(_Section):
    base_url: str
    mode: Literal["http", "local"]
    timeout_s: float
    query_mode: Literal["research", "live"]
    datasets: PitDatasets


class LocalDataset(_Section):
    path: str
    keys: list[str]


class LocalSourceSettings(_Section):
    datasets: dict[str, LocalDataset]
    valid_time_column: str
    knowledge_time_column: str


class ScenarioSettings(_Section):
    vol_multiplier: float
    spread_multiplier: float
    depth_multiplier: float
    trade_rate_multiplier: float


class SimSettings(_Section):
    base_url: str
    mode: Literal["http", "fake"]
    timeout_s: float
    scenario: str
    seed: int
    max_duration_s: int
    fake_scenarios: dict[str, ScenarioSettings]


class FillsSettings(_Section):
    root_subdir: str
    orders_subdir: str


class InstrumentSettings(_Section):
    base: str
    quote: str
    tick_size: float
    lot_size: float
    price_precision: int
    size_precision: int


class MarketSettings(_Section):
    venue: str
    bar_interval_s: int
    seconds_per_year: float
    seconds_per_day: float
    default_universe: list[str]
    instruments: dict[str, InstrumentSettings]


class CostSettings(_Section):
    maker_fee: float
    taker_fee: float


class AccountSettings(_Section):
    initial_cash: float
    leverage: float


class NautilusSettings(_Section):
    data_mode: Literal["bars", "trades"]
    trader_id: str
    log_level: str


class HftBacktestSettings(_Section):
    exchange: Literal["no_partial_fill", "partial_fill"]
    queue_model: Literal["power_prob", "risk_adverse"]
    queue_power: float
    entry_latency_ns: int
    response_latency_ns: int
    feed_latency_ns: int
    contract_size: float


class VectorbtSettings(_Section):
    direction: Literal["both", "longonly", "shortonly"]


class EngineSettings(_Section):
    nautilus: NautilusSettings
    hftbacktest: HftBacktestSettings
    vectorbt: VectorbtSettings


class CalibrationSettings(_Section):
    horizon_ns: int
    vol_window: int
    test_fraction: float
    seed: int
    min_samples: int
    logistic_c: float
    latency_quantile: float
    max_iter: int


class MetricsSettings(_Section):
    ci_level: float
    n_boot: int
    block_len: int
    seed: int


class PromotionCriteria(_Section):
    min_sharpe: float
    min_fills: int
    max_drawdown: float
    max_sharpe_drop: float | None = None


class LadderSettings(_Section):
    levels: list[Fidelity]
    run_all_levels: bool
    causality_cuts: int
    promotion: dict[Fidelity, PromotionCriteria]


class ParitySettings(_Section):
    price_rel_tol: float
    rounding_ticks: float
    qty_lots: float
    fee_rel_tol: float
    pnl_rel_tol: float
    timing_window_bars: int
    max_disagreement_rate: float


class SweepSettings(_Section):
    objective: str
    random_samples: int
    seed: int


class WalkForwardSettings(_Section):
    train_bars: int
    test_bars: int
    step_bars: int
    fidelity: Fidelity


class StrategySettings(_Section):
    defaults: dict[str, Any]
    grid: dict[str, list[Any]]
    ranges: dict[str, list[float]]


class SyntheticFillsSettings(_Section):
    n_orders: int
    limit_share: float
    venue: str
    account_id: str
    strategy_id: str
    base_hazard_per_s: float
    distance_coef: float
    queue_coef: float
    vol_coef: float
    order_ttl_s: float
    max_distance_bps: float
    queue_mean: float
    half_spread_bps: float
    impact_coef_bps: float
    slippage_noise_bps: float
    latency_median_ns: float
    latency_sigma: float
    qty_mean: float
    order_spacing_s: float


class SyntheticSettings(_Section):
    seed: int
    start: str
    n_bars: int
    quote_interval_ms: int
    initial_price: float
    vol_bps_per_sqrt_s: float
    regime_bars: int
    trend_bps_per_bar: float
    half_spread_ticks: int
    max_extra_spread_ticks: int
    book_levels: int
    level_size_mean: float
    trade_rate_per_s: float
    trade_size_mean: float
    feed_latency_ns: int
    fills: SyntheticFillsSettings


class ReproduceSettings(_Section):
    strategies: list[str]
    parity_strategies: list[str]
    run_all_levels: bool
    sweep_strategy: str


class ExperimentSettings(_Section):
    subtree: str
    ledger_inbox_subdir: str
    seeds: list[int]
    n_bars: int
    speed_repeats: int
    strategies: list[str]
    docs_tables: str


class Settings(BaseSettings):
    """All runtime configuration of the orchestrator."""

    model_config = SettingsConfigDict(
        env_prefix="BT_",
        env_nested_delimiter="__",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
        frozen=True,
    )

    data_dir: Path = Field(validation_alias=AliasChoices("TRADEQ_DATA_DIR", "BT_DATA_DIR"))
    code_version: str | None = None

    service: ServiceSettings
    logging: LoggingSettings
    paths: PathSettings
    ledger: LedgerSettings
    pit: PitSettings
    local_source: LocalSourceSettings
    sim: SimSettings
    fills: FillsSettings
    market: MarketSettings
    costs: CostSettings
    account: AccountSettings
    engines: EngineSettings
    calibration: CalibrationSettings
    metrics: MetricsSettings
    ladder: LadderSettings
    parity: ParitySettings
    sweep: SweepSettings
    walk_forward: WalkForwardSettings
    strategies: dict[str, StrategySettings]
    synthetic: SyntheticSettings
    reproduce: ReproduceSettings
    experiments: ExperimentSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=default_config_file())
        return init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings

    @property
    def project_dir(self) -> Path:
        """This project's data subtree, ``$TRADEQ_DATA_DIR/backtest-orchestrator``."""
        return self.data_dir / self.paths.subtree

    @property
    def reports_dir(self) -> Path:
        return self.project_dir / self.paths.reports

    @property
    def models_dir(self) -> Path:
        return self.project_dir / self.paths.models

    @property
    def synthetic_dir(self) -> Path:
        return self.project_dir / self.paths.synthetic

    @property
    def trials_dir(self) -> Path:
        return self.project_dir / self.paths.trials

    @property
    def sim_fake_dir(self) -> Path:
        return self.project_dir / self.paths.sim_fake

    @property
    def ledger_inbox_dir(self) -> Path:
        return self.data_dir / self.ledger.inbox_subdir

    @property
    def fills_root(self) -> Path:
        return self.data_dir / self.fills.root_subdir

    @property
    def orders_root(self) -> Path:
        return self.data_dir / self.fills.orders_subdir

    @property
    def bar_interval_ns(self) -> int:
        return self.market.bar_interval_s * NANOS_PER_SECOND

    @property
    def periods_per_year(self) -> float:
        return self.market.seconds_per_year / self.market.bar_interval_s

    def strategy_settings(self, name: str) -> StrategySettings:
        try:
            return self.strategies[name]
        except KeyError as exc:
            raise KeyError(f"no settings for strategy {name!r}") from exc

    def instrument(self, symbol: str) -> InstrumentSettings:
        try:
            return self.market.instruments[symbol]
        except KeyError as exc:
            raise KeyError(f"symbol {symbol!r} is not configured under market.instruments") from exc


def load_settings(**overrides: Any) -> Settings:
    """Build settings from YAML + env; keyword overrides take precedence (used by tests)."""
    return Settings(**overrides)

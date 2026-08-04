"""Typed application configuration.

Everything is environment-driven. There are no secrets, paths, or tunables
hard-coded anywhere else in the package -- if you find one, it is a bug.

Local development defaults to SQLite so the project runs with zero external
services. Production points ``FORECASTER_DATABASE_URL`` at Postgres; the ORM
layer adapts its column types automatically (see ``db.types``).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: .../backend/src/forecaster/config.py -> up 4 -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORECASTER_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ── Runtime ────────────────────────────────────────────────────────────
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE
    random_seed: int = 42

    # ── Storage ────────────────────────────────────────────────────────────
    database_url: str = f"sqlite+aiosqlite:///{(REPO_ROOT / 'data' / 'forecaster.db').as_posix()}"
    database_echo: bool = False
    redis_url: str | None = None

    #: Root of the cold-tier Parquet lake.
    lake_root: Path = REPO_ROOT / "data" / "lake"

    #: Where trained model artifacts are persisted.
    artifact_root: Path = REPO_ROOT / "data" / "artifacts"

    # ── Universe / tiering ─────────────────────────────────────────────────
    #: Hard cap on symbols promoted to the Postgres hot tier. Free Postgres is
    #: 0.5 GB; ~800 symbols x 5y x ~80 bytes/row keeps us comfortably inside it.
    hot_tier_max_symbols: int = 800
    default_history_years: int = 5

    # ── Provider credentials (all optional; router skips unconfigured) ─────
    alphavantage_api_key: SecretStr | None = None
    finnhub_api_key: SecretStr | None = None
    tiingo_api_key: SecretStr | None = None
    fmp_api_key: SecretStr | None = None
    fred_api_key: SecretStr | None = None
    huggingface_api_token: SecretStr | None = None

    #: SEC requires a descriptive User-Agent with contact info on every request.
    #: Requests without one are throttled or blocked outright.
    sec_user_agent: str = "StockForecaster research contact@example.com"

    # ── HTTP behaviour ─────────────────────────────────────────────────────
    http_timeout_seconds: float = 20.0
    http_max_retries: int = 3
    provider_circuit_breaker_threshold: int = 5
    provider_circuit_breaker_cooldown_seconds: int = 300

    # ── API service ────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_rate_limit: str = "120/minute"

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        # Allow FORECASTER_API_CORS_ORIGINS="http://a.com,http://b.com"
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def _guard_production(self) -> Settings:
        if self.environment is Environment.PRODUCTION:
            if self.database_url.startswith("sqlite"):
                raise ValueError("SQLite is not permitted in production; point at Postgres.")
            if self.debug:
                raise ValueError("debug must be False in production.")
        return self

    # ── Derived helpers ────────────────────────────────────────────────────
    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgresql", "postgres"))

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def sync_database_url(self) -> str:
        """Alembic and DuckDB attachments need a sync driver."""
        return self.database_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")

    def ensure_directories(self) -> None:
        """Create the on-disk trees the app assumes exist."""
        for path in (self.lake_root, self.artifact_root):
            path.mkdir(parents=True, exist_ok=True)
        if self.is_sqlite:
            # sqlite+aiosqlite:////abs/path/forecaster.db -> /abs/path
            db_file = self.database_url.split("///", 1)[-1]
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that config is read once. Tests override via
    ``get_settings.cache_clear()`` after monkeypatching the environment.
    """
    return Settings()

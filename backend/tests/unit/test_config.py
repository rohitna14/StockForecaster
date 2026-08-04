"""Configuration tests."""

from __future__ import annotations

import pytest

from forecaster.config import Environment, Settings


def test_cors_allows_both_localhost_spellings() -> None:
    """localhost and 127.0.0.1 are different origins to a browser.

    Allowing only one silently blocks every client-side fetch from the other.
    The failure is nasty to debug because the server is healthy, curl succeeds
    (curl ignores CORS), and the browser reports it as a generic network error
    rather than anything mentioning CORS.
    """
    origins = Settings().api_cors_origins

    for port in (3000, 8501):
        assert f"http://localhost:{port}" in origins, f"localhost:{port} not allowed"
        assert f"http://127.0.0.1:{port}" in origins, f"127.0.0.1:{port} not allowed"


def test_cors_origins_accept_comma_separated_env() -> None:
    """FORECASTER_API_CORS_ORIGINS="a,b" must parse into a list."""
    settings = Settings(api_cors_origins="https://a.com, https://b.com")  # type: ignore[arg-type]
    assert settings.api_cors_origins == ["https://a.com", "https://b.com"]


def test_production_rejects_sqlite() -> None:
    """A production deploy pointed at SQLite is a misconfiguration, not a choice."""
    with pytest.raises(ValueError, match="SQLite is not permitted in production"):
        Settings(
            environment=Environment.PRODUCTION,
            database_url="sqlite+aiosqlite:///./x.db",
        )


def test_production_rejects_debug() -> None:
    with pytest.raises(ValueError, match="debug must be False"):
        Settings(
            environment=Environment.PRODUCTION,
            database_url="postgresql+asyncpg://u:p@h/db",
            debug=True,
        )


def test_log_level_is_validated_and_normalised() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValueError, match="log_level must be one of"):
        Settings(log_level="chatty")


def test_dialect_helpers_agree_with_url() -> None:
    sqlite = Settings(database_url="sqlite+aiosqlite:///./x.db")
    assert sqlite.is_sqlite and not sqlite.is_postgres

    postgres = Settings(database_url="postgresql+asyncpg://u:p@h/db")
    assert postgres.is_postgres and not postgres.is_sqlite
    # Alembic and DuckDB need a sync driver.
    assert "+asyncpg" not in postgres.sync_database_url

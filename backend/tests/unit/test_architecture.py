"""Architectural constraints, enforced as tests.

Layering rules decay the moment they are only written down. These assert them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_SRC = REPO_ROOT / "backend" / "src" / "forecaster"
STREAMLIT_APP = REPO_ROOT / "streamlit_app"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


# ═════════════════════ clients must speak HTTP only ════════════════════════
@pytest.mark.skipif(not STREAMLIT_APP.exists(), reason="streamlit app not present")
def test_streamlit_client_contains_no_domain_logic() -> None:
    """The Streamlit app must not import the forecasting package.

    The project claims the API is client-agnostic and that two independent
    frontends share one backend. That claim is only true while the clients
    cannot reach into the domain layer. A single `from forecaster... import`
    here would silently make the architecture diagram a lie.
    """
    forbidden = {
        "forecaster",
        "sklearn",
        "lightgbm",
        "xgboost",
        "catboost",
        "tensorflow",
        "torch",
        "statsmodels",
        "numpy",
        "pandas",
    }

    offenders: dict[str, set[str]] = {}
    for path in STREAMLIT_APP.rglob("*.py"):
        violations = _imported_modules(path) & forbidden
        if violations:
            offenders[str(path.relative_to(REPO_ROOT))] = violations

    assert not offenders, (
        f"Streamlit client imports domain/ML packages: {offenders}. "
        f"It must call the API over HTTP instead — see streamlit_app/api_client.py."
    )


# ═════════════════════ backend layering ════════════════════════════════════
def test_domain_layer_does_not_import_the_api() -> None:
    """Features, models, validation and backtest must not depend on FastAPI.

    Everything the API exposes has to be reachable from the CLI and the tests
    without HTTP. If the domain imported the web layer, the library would only
    be usable through a running server.
    """
    domain_packages = [
        "features",
        "models",
        "validation",
        "backtest",
        "explain",
        "ingestion",
        "lake",
    ]

    offenders: dict[str, set[str]] = {}
    for package in domain_packages:
        directory = BACKEND_SRC / package
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            imports = _imported_modules(path)
            if "fastapi" in imports or "starlette" in imports:
                offenders[str(path.relative_to(BACKEND_SRC))] = imports & {"fastapi", "starlette"}

    assert not offenders, f"Domain modules import the web layer: {offenders}"


def test_core_package_does_not_import_heavy_ml_at_module_scope() -> None:
    """`import forecaster` must not drag in TensorFlow or the boosting libraries.

    The prototype imported TensorFlow, XGBoost, LightGBM, CatBoost and
    statsmodels at module scope, which pushed cold start past a minute and made
    the container enormous. Heavy imports belong inside functions or behind the
    lazy model registry.
    """
    heavy = {
        "tensorflow",
        "torch",
        "keras",
        "lightgbm",
        "xgboost",
        "catboost",
        "statsmodels",
        "shap",
        "optuna",
    }

    always_loaded = [
        BACKEND_SRC / "config.py",
        BACKEND_SRC / "logging.py",
        BACKEND_SRC / "exceptions.py",
        BACKEND_SRC / "db" / "models.py",
        BACKEND_SRC / "db" / "session.py",
        BACKEND_SRC / "models" / "registry.py",
        BACKEND_SRC / "models" / "base.py",
        BACKEND_SRC / "models" / "baselines.py",
    ]

    offenders: dict[str, set[str]] = {}
    for path in always_loaded:
        if not path.exists():
            continue
        violations = _imported_modules(path) & heavy
        if violations:
            offenders[str(path.relative_to(BACKEND_SRC))] = violations

    assert not offenders, (
        f"Heavy ML libraries imported at module scope: {offenders}. "
        f"Move them inside functions or behind the lazy registry."
    )


def test_baselines_are_registered_and_cannot_be_dropped() -> None:
    """The reference baseline must always survive model-set resolution."""
    from forecaster.models.baselines import REFERENCE_BASELINE
    from forecaster.models.registry import model_set

    for name in ("fast", "linear", "trees", "standard"):
        assert REFERENCE_BASELINE in model_set(name), (
            f"model_set({name!r}) omitted the reference baseline. Every "
            f"leaderboard must carry it — a result without a baseline is not a result."
        )


def test_no_print_statements_in_domain_code() -> None:
    """Domain code must log, not print.

    The prototype reported every failure with print() inside a bare except,
    which made errors invisible in deployment and unsearchable locally.
    """
    offenders: list[str] = []
    skip = {"cli.py", "cli_evaluate.py"}  # CLI output is legitimate

    for path in BACKEND_SRC.rglob("*.py"):
        if path.name in skip:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.relative_to(BACKEND_SRC)}:{node.lineno}")

    assert not offenders, f"print() found in domain code: {offenders}. Use structlog."


def test_no_bare_except_in_domain_code() -> None:
    """`except:` swallows KeyboardInterrupt and SystemExit along with everything else."""
    offenders: list[str] = []

    for path in BACKEND_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                offenders.append(f"{path.relative_to(BACKEND_SRC)}:{node.lineno}")

    assert not offenders, f"Bare `except:` found: {offenders}"

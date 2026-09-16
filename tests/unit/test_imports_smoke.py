"""Import-surface smoke test (task 1.2): full dependency import works in this env."""

from __future__ import annotations


def test_core_package_imports() -> None:
    import fastapi  # noqa: F401
    import lightgbm  # noqa: F401
    import optuna  # noqa: F401
    import polars  # noqa: F401
    import pydantic  # noqa: F401
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401


def test_project_package_imports() -> None:
    import telco_churn  # noqa: F401
    from telco_churn import config  # noqa: F401

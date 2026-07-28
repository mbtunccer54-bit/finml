"""Enforce that the domain layer stays dependency-free.

The Clean Architecture rule this project is built on — the domain imports
nothing external — is easy to state and easy to break. One ``import numpy`` for
a quick mean, and the business core is no longer testable or reviewable in
isolation, and nothing fails to tell you.

This script parses every module under ``src/domain`` and rejects any import
outside the standard library and the domain package itself. It runs as a
pre-commit hook and as a unit test.

Exit codes:
    0: the domain is clean.
    1: a forbidden import was found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

#: Distribution packages the domain must never import.
FORBIDDEN_PREFIXES: frozenset[str] = frozenset(
    {
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "catboost",
        "xgboost",
        "lightgbm",
        "shap",
        "lime",
        "dice_ml",
        "mlflow",
        "optuna",
        "imblearn",
        "pandera",
        "statsmodels",
        "torch",
        "transformers",
        "bertopic",
        "spacy",
        "fastapi",
        "pydantic",
        "streamlit",
        "plotly",
        "hydra",
        "omegaconf",
        "structlog",
        "httpx",
        "requests",
        "matplotlib",
        "sqlalchemy",
        "prometheus_client",
        "infrastructure",
        "application",
        "presentation",
    }
)


def module_roots(tree: ast.Module) -> set[str]:
    """Collect the root package of every import in a module.

    Args:
        tree: Parsed module AST.

    Returns:
        The distinct root package names imported.
    """
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        # level > 0 is a relative import, which stays inside the package.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def check_file(path: Path) -> list[str]:
    """Check one module for forbidden imports.

    Args:
        path: Module path.

    Returns:
        Human-readable violation messages; empty when the module is clean.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: could not parse ({exc})"]

    return [
        f"{path}: forbidden import '{root}'"
        for root in sorted(module_roots(tree) & FORBIDDEN_PREFIXES)
    ]


def check_domain(domain_dir: Path) -> list[str]:
    """Check every module in the domain package.

    Args:
        domain_dir: Path to ``src/domain``.

    Returns:
        All violation messages found.
    """
    if not domain_dir.is_dir():
        return [f"{domain_dir}: domain directory not found"]
    violations: list[str] = []
    for path in sorted(domain_dir.rglob("*.py")):
        violations.extend(check_file(path))
    return violations


def main() -> int:
    """Run the check against ``src/domain``.

    Returns:
        ``0`` when clean, ``1`` when a violation was found.
    """
    root = Path(__file__).resolve().parents[1]
    violations = check_domain(root / "src" / "domain")

    if violations:
        sys.stderr.write("Domain purity violations:\n")
        for violation in violations:
            sys.stderr.write(f"  - {violation}\n")
        sys.stderr.write(
            "\nThe domain layer must depend only on the standard library.\n"
            "Move the offending logic to infrastructure/ or application/.\n"
        )
        return 1

    sys.stdout.write("Domain layer is clean: standard library imports only.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
